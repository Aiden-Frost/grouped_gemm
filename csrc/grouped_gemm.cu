#include "grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "cublas_v2.h"

namespace grouped_gemm {

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

#define CUBLAS_CALL(code)					  \
  do {								  \
    cublasStatus_t status = code;				  \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "CuBLAS Error"); \
  } while (0)

#define GROUPED_GEMM_STRINGIFY_HELPER(x) #x
#define GROUPED_GEMM_STRINGIFY(x) \
  GROUPED_GEMM_STRINGIFY_HELPER(x)


template <typename T>
torch::Tensor CopyToDevice(const std::vector<T> &x, const torch::Device &device) {
  size_t bytes = x.size() * sizeof(T);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(device);
  torch::Tensor out = torch::empty(bytes, options);

  CUDA_CALL(cudaMemcpyAsync(out.data_ptr(),
			    x.data(), bytes,
			    cudaMemcpyHostToDevice,
			    c10::cuda::getCurrentCUDAStream()));
  return out;
}


// void CublasGemm(c10::Half *a, int64_t a_rows, int64_t a_cols, bool trans_a,
// 		c10::Half *b, int64_t b_rows, int64_t b_cols, bool trans_b,
// 		c10::Half *c, int64_t c_rows, int64_t c_cols) {
//   int m = trans_b ? b_rows : b_cols;
//   int k = trans_b ? b_cols : b_rows;
//   int n = trans_a ? a_cols : a_rows;
//
//   int lda = trans_a ? n : k;
//   int ldb = trans_b ? k : m;
//   cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
//   cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
//
//   float alpha = 1.0, beta = 0.0;
//   CUBLAS_CALL(cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(),
// 			   transpose_b, transpose_a,
// 			   m, n, k, &alpha,
// 			   b, CUDA_R_16F, ldb,
// 			   a, CUDA_R_16F, lda,
// 			   &beta,
// 			   c, CUDA_R_16F, c_cols, CUDA_R_32F,
// 			   CUBLAS_GEMM_DEFAULT));
// }

void CublasGemm(c10::Half *a, int64_t a_rows, int64_t a_cols, bool trans_a,
		c10::Half *b, int64_t b_rows, int64_t b_cols, bool trans_b,
		c10::Half *c, int64_t c_rows, int64_t c_cols,
        cublasHandle_t handle, int group_count, const int group_size[]) {
  int m = trans_b ? b_rows : b_cols;
  int k = trans_b ? b_cols : b_rows;
  int n = trans_a ? a_cols : a_rows;

  int lda = trans_a ? n : k;
  int ldb = trans_b ? k : m;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  float alpha = 1.0, beta = 0.0;

  // Adjust the dimensions for grouped batched gemm
  int m_array[group_count], n_array[group_count], k_array[group_count];
  const float alpha_array[group_count] = {alpha}; // Assuming alpha is same for all batches
  const float beta_array[group_count] = {beta};   // Assuming beta is same for all batches
  const int lda_array[group_count] = {lda};      // Assuming lda is same for all batches
  const int ldb_array[group_count] = {ldb};      // Assuming ldb is same for all batches
  const int ldc_array[group_count] = {n};        // Assuming ldc is same for all batches

  for (int i = 0; i < group_count; ++i) {
      m_array[i] = m;
      n_array[i] = n;
      k_array[i] = k;
  }

  // Call cublasSgemmGroupedBatched
  CUBLAS_CALL(cublasSgemmGroupedBatched(at::cuda::getCurrentCUDABlasHandle(),
                                         &transpose_b, &transpose_a,
                                         m_array, n_array, k_array,
                                         alpha_array,
                                         (const float *const*)&b, ldb_array,
                                         (const float *const*)&a, lda_array,
                                         beta_array,
                                         (float *const*)&c, ldc_array,
                                         group_count, group_size));
}


// void CublasGroupedBatchedGemm(torch::Tensor a,
//            torch::Tensor b,
//            torch::Tensor c,
//            torch::Tensor batch_sizes,
//            bool trans_b) {
//
//
//     int64_t bs = batch_sizes.size(0), k = a.size(1);
//     int64_t n = trans_b ? b.size(1) : b.size(2);
//     int64_t b_rows = b.size(1), b_cols = b.size(2);
//     c10::Half* a_ptr = a.data_ptr<c10::Half>();
//     c10::Half* b_ptr = b.data_ptr<c10::Half>();
//     c10::Half* c_ptr = c.data_ptr<c10::Half>();
//     for (int i = 0; i < bs; ++i) {
//     int64_t m = batch_sizes.data_ptr<int64_t>()[i];
//     CublasGemm(a_ptr, m, k, /*trans_a=*/false,
//        b_ptr, b_rows, b_cols, trans_b,
//        c_ptr, m, n);
//         a_ptr += m * k;
//         b_ptr += b_rows * b_cols;
//         c_ptr += m * n;
//     }
//
// }

// void CublasGroupedGemm(torch::Tensor a,
// 		       torch::Tensor b,
// 		       torch::Tensor c,
// 		       torch::Tensor batch_sizes,
// 		       bool trans_b) {
//   int64_t bs = batch_sizes.size(0), k = a.size(1);
//   int64_t n = trans_b ? b.size(1) : b.size(2);
//   int64_t b_rows = b.size(1), b_cols = b.size(2);
//   c10::Half* a_ptr = a.data_ptr<c10::Half>();
//   c10::Half* b_ptr = b.data_ptr<c10::Half>();
//   c10::Half* c_ptr = c.data_ptr<c10::Half>();
//   for (int i = 0; i < bs; ++i) {
//     int64_t m = batch_sizes.data_ptr<int64_t>()[i];
//     CublasGemm(a_ptr, m, k, /*trans_a=*/false,
// 	       b_ptr, b_rows, b_cols, trans_b,
// 	       c_ptr, m, n);
//     a_ptr += m * k;
//     b_ptr += b_rows * b_cols;
//     c_ptr += m * n;
//   }
// }

// void CublasGroupedGemmVariableK(torch::Tensor a,
// 				torch::Tensor b,
// 				torch::Tensor c,
// 				torch::Tensor batch_sizes) {
//   int64_t bs = batch_sizes.size(0), m = a.size(1), n = b.size(1);
//   c10::Half* a_ptr = a.data_ptr<c10::Half>();
//   c10::Half* b_ptr = b.data_ptr<c10::Half>();
//   c10::Half* c_ptr = c.data_ptr<c10::Half>();
//   for (int i = 0; i < bs; ++i) {
//     int64_t k = batch_sizes.data_ptr<int64_t>()[i];
//     CublasGemm(a_ptr, k, m, /*trans_a=*/true,
// 	       b_ptr, k, n, /*trans_b=*/false,
// 	       c_ptr, m, n);
//     a_ptr += k * m;
//     b_ptr += k * n;
//     c_ptr += m * n;
//   }
// }


// NOTE: We only support dynamic group sizes for the 'a' tensor. Tensor 'b' is
// assumed to be batched with fixed sized batches.
//
// TODO(tgale): Validate alignment is true for every batch element.
// void GroupedGemm(torch::Tensor a,
// 		 torch::Tensor b,
// 		 torch::Tensor c,
// 		 torch::Tensor batch_sizes,
// 		 bool trans_a, bool trans_b) {
//
//   // Defer to the variable 'k' helper for the rest of the op.
//   if (trans_a) {
//     CublasGroupedGemmVariableK(a, b, c, batch_sizes);
//     return;
//   }
//
//   CublasGroupedGemm(a, b, c, batch_sizes, trans_b);
//   return;
//
// }
//
// }

void CublasGroupedBatchedGemm(torch::Tensor a,
           torch::Tensor b,
           torch::Tensor c,
           torch::Tensor batch_sizes,
           bool trans_b,
           cublasHandle_t handle) {

    int64_t bs = batch_sizes.size(0), k = a.size(1);
    int64_t n = trans_b ? b.size(1) : b.size(2);
    int64_t b_rows = b.size(1), b_cols = b.size(2);
    c10::Half* a_ptr = a.data_ptr<c10::Half>();
    c10::Half* b_ptr = b.data_ptr<c10::Half>();
    c10::Half* c_ptr = c.data_ptr<c10::Half>();

    // Assuming group_count is equal to batch_sizes.size(0)
    int group_count = bs;
    int group_size[group_count];

    for (int i = 0; i < bs; ++i) {
        group_size[i] = batch_sizes.data_ptr<int64_t>()[i];
    }

    CublasGemm(a_ptr, bs, k, /*trans_a=*/false,
               b_ptr, b_rows, b_cols, trans_b,
               c_ptr, bs, n, handle, group_count, group_size);
}

void CublasGroupedGemm(torch::Tensor a,
		       torch::Tensor b,
		       torch::Tensor c,
		       torch::Tensor batch_sizes,
		       bool trans_b,
		       cublasHandle_t handle) {
  int64_t bs = batch_sizes.size(0), k = a.size(1);
  int64_t n = trans_b ? b.size(1) : b.size(2);
  int64_t b_rows = b.size(1), b_cols = b.size(2);
  c10::Half* a_ptr = a.data_ptr<c10::Half>();
  c10::Half* b_ptr = b.data_ptr<c10::Half>();
  c10::Half* c_ptr = c.data_ptr<c10::Half>();

  // Assuming group_count is equal to batch_sizes.size(0)
  int group_count = bs;
  int group_size[group_count];

  for (int i = 0; i < bs; ++i) {
      group_size[i] = batch_sizes.data_ptr<int64_t>()[i];
  }

  CublasGemm(a_ptr, bs, k, /*trans_a=*/false,
             b_ptr, b_rows, b_cols, trans_b,
             c_ptr, bs, n, handle, group_count, group_size);
}

void CublasGroupedGemmVariableK(torch::Tensor a,
				torch::Tensor b,
				torch::Tensor c,
				torch::Tensor batch_sizes,
				cublasHandle_t handle) {
  int64_t bs = batch_sizes.size(0), m = a.size(1), n = b.size(1);
  c10::Half* a_ptr = a.data_ptr<c10::Half>();
  c10::Half* b_ptr = b.data_ptr<c10::Half>();
  c10::Half* c_ptr = c.data_ptr<c10::Half>();

  // Assuming group_count is equal to batch_sizes.size(0)
  int group_count = bs;
  int group_size[group_count];

  for (int i = 0; i < bs; ++i) {
      group_size[i] = batch_sizes.data_ptr<int64_t>()[i];
  }

  CublasGemm(a_ptr, m, bs, /*trans_a=*/true,
             b_ptr, m, n, /*trans_b=*/false,
             c_ptr, bs, n, handle, group_count, group_size);
}

void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b,
		 cublasHandle_t handle) {

  // Defer to the variable 'k' helper for the rest of the op.
  if (trans_a) {
    CublasGroupedGemmVariableK(a, b, c, batch_sizes, handle);
    return;
  }

  CublasGroupedGemm(a, b, c, batch_sizes, trans_b, handle);
  return;

}

}