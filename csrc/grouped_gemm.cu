#include "grouped_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace grouped_gemm {

#define CUDA_CALL(code)                                     \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);                \
  } while (0)

#define CUBLAS_CALL(code)                                         \
  do {                                                            \
    cublasStatus_t status = code;                                 \
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


void CublasGemm(float *a, int64_t a_rows, int64_t a_cols, bool trans_a,
                float *b, int64_t b_rows, int64_t b_cols, bool trans_b,
                float *c, int64_t c_rows, int64_t c_cols) {
  int m = trans_b ? b_rows : b_cols;
  int k = trans_b ? b_cols : b_rows;
  int n = trans_a ? a_cols : a_rows;

  int lda = trans_a ? n : k;
  int ldb = trans_b ? k : m;
  cublasOperation_t transpose_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transpose_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  float alpha = 1.0, beta = 0.0;
  CUBLAS_CALL(cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(),
                           transpose_b, transpose_a,
                           m, n, k, &alpha,
                           b, CUDA_R_32F, ldb,
                           a, CUDA_R_32F, lda,
                           &beta,
                           c, CUDA_R_32F, c_cols, CUDA_R_32F,
                           CUBLAS_GEMM_DEFAULT));
}

void CublasGroupedBatchedGemm(torch::Tensor a,
                              torch::Tensor b,
                              torch::Tensor c,
                              torch::Tensor batch_sizes,
                              bool trans_b) {
    bool trans_a = false;
    int64_t bs = batch_sizes.size(0);
    int64_t k = a.size(1);
    int64_t n = trans_b ? b.size(1) : b.size(2);
    int64_t b_rows = b.size(1), b_cols = b.size(2);

    auto A = a.contiguous().cpu();
    auto B = b.contiguous().cpu();
    auto C = c.contiguous().cpu();

    const int gemm_count = bs;
    std::vector<float*> d_A(gemm_count, nullptr);
    std::vector<float*> d_B(gemm_count, nullptr);
    std::vector<float*> d_C(gemm_count, nullptr);

    float **d_A_array = nullptr;
    float **d_B_array = nullptr;
    float **d_C_array = nullptr;

    cublasHandle_t cublasH;
    cudaStream_t stream;

    CUBLAS_CALL(cublasCreate(&cublasH));
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CALL(cublasSetStream(cublasH, stream));

    for (int i = 0; i < gemm_count; ++i) {
        int64_t m = batch_sizes.data_ptr<int64_t>()[i];
        CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(float) * m * k));
        CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(float) * b_rows * b_cols));
        CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(float) * m * n));
    }

    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(float*) * gemm_count));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(float*) * gemm_count));
    CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(float*) * gemm_count));

    for (int i = 0; i < gemm_count; ++i) {
        int64_t m = batch_sizes.data_ptr<int64_t>()[i];
        CUDA_CALL(cudaMemcpyAsync(d_A[i], a.data_ptr<float>() + (i * m * k), sizeof(float) * (m * k), cudaMemcpyHostToDevice, stream));
        CUDA_CALL(cudaMemcpyAsync(d_B[i], b.data_ptr<float>() + (i * b_rows * b_cols), sizeof(float) * (b_rows * b_cols), cudaMemcpyHostToDevice, stream));
    }

    CUDA_CALL(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(float*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(float*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(float*) * gemm_count, cudaMemcpyHostToDevice, stream));




    cublasOperation_t* transa_array = new cublasOperation_t[bs];
    cublasOperation_t* transb_array = new cublasOperation_t[bs];

    int* m_array = new int[bs];
    int* n_array = new int[bs];
    int* k_array = new int[bs];

    float* alpha_array = new float[bs];
    float* beta_array = new float[bs];

    int* lda_array = new int[bs];
    int* ldb_array = new int[bs];
    int* ldc_array = new int[bs];

    int* group_size = new int[bs];

    for (int i = 0; i < bs; ++i) {
        int64_t m = batch_sizes.data_ptr<int64_t>()[i];

        m_array[i] = m;
        n_array[i] = n;
        k_array[i] = k;

        alpha_array[i] = 1.0;
        beta_array[i] = 0.0;

        lda_array[i] = k;
        ldb_array[i] = b_cols;
        ldc_array[i] = n;

        group_size[i] = 1;

        transa_array[i] = CUBLAS_OP_N;
        transb_array[i] = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    }

    CUBLAS_CALL(cublasSgemmGroupedBatched(
        at::cuda::getCurrentCUDABlasHandle(),
        transa_array, transb_array,
        m_array, n_array, k_array,
        alpha_array, d_A_array, lda_array,
        d_B_array, ldb_array, beta_array,
        d_C_array, ldc_array,
        bs, group_size
    ));

    // Step 4: copy data back to host
    for (int i = 0; i < gemm_count; ++i) {
        int64_t m = batch_sizes.data_ptr<int64_t>()[i];
        CUDA_CALL(cudaMemcpyAsync(c.data_ptr<float>() + (i * m * n), d_C[i], sizeof(float) * (m * n), cudaMemcpyDeviceToHost, stream));
    }

    // Synchronize the stream to ensure all operations are completed
    CUDA_CALL(cudaStreamSynchronize(stream));

    // Clean up
    for (int i = 0; i < gemm_count; ++i) {
        CUDA_CALL(cudaFree(d_A[i]));
        CUDA_CALL(cudaFree(d_B[i]));
        CUDA_CALL(cudaFree(d_C[i]));
    }

    CUDA_CALL(cudaFree(d_A_array));
    CUDA_CALL(cudaFree(d_B_array));
    CUDA_CALL(cudaFree(d_C_array));

    CUBLAS_CALL(cublasDestroy(cublasH));
    CUDA_CALL(cudaStreamDestroy(stream));

    delete transa_array;
    delete transb_array;
    delete m_array;
    delete n_array;
    delete k_array;
    delete alpha_array;
    delete beta_array;
    delete lda_array;
    delete ldb_array;
    delete ldc_array;
    delete group_size;
}

void CublasGroupedGemm(torch::Tensor a,
                       torch::Tensor b,
                       torch::Tensor c,
                       torch::Tensor batch_sizes,
                       bool trans_b) {
  int64_t bs = batch_sizes.size(0), k = a.size(1);
  int64_t n = trans_b ? b.size(1) : b.size(2);
  int64_t b_rows = b.size(1), b_cols = b.size(2);
  float* a_ptr = a.data_ptr<float>();
  float * b_ptr = b.data_ptr<float>();
  float* c_ptr = c.data_ptr<float>();
  for (int i = 0; i < bs; ++i) {
    int64_t m = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(a_ptr, m, k, /*trans_a=*/false,
               b_ptr, b_rows, b_cols, trans_b,
               c_ptr, m, n);
    a_ptr += m * k;
    b_ptr += b_rows * b_cols;
    c_ptr += m * n;
  }
}

void CublasGroupedGemmVariableK(torch::Tensor a,
                                torch::Tensor b,
                                torch::Tensor c,
                                torch::Tensor batch_sizes) {
  int64_t bs = batch_sizes.size(0), m = a.size(1), n = b.size(1);
  float* a_ptr = a.data_ptr<float>();
  float* b_ptr = b.data_ptr<float>();
  float* c_ptr = c.data_ptr<float>();
  for (int i = 0; i < bs; ++i) {
    int64_t k = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(a_ptr, k, m, /*trans_a=*/true,
               b_ptr, k, n, /*trans_b=*/false,
               c_ptr, m, n);
    a_ptr += k * m;
    b_ptr += k * n;
    c_ptr += m * n;
  }
}


// NOTE: We only support dynamic group sizes for the 'a' tensor. Tensor 'b' is
// assumed to be batched with fixed sized batches.
//
// TODO(tgale): Validate alignment is true for every batch element.
void GroupedGemm(torch::Tensor a,
                 torch::Tensor b,
                 torch::Tensor c,
                 torch::Tensor batch_sizes,
                 bool trans_a, bool trans_b) {
    // Defer to the variable 'k' helper for the rest of the op.
    if (trans_a) {
        CublasGroupedGemmVariableK(a, b, c, batch_sizes);
        return;
    }


    CublasGroupedGemm(a, b, c, batch_sizes, trans_b);
    return;
}}
