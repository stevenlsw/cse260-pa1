/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */
#include <string.h>
#include <immintrin.h>
#include <avx2intrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define L1_BLOCK_SIZE 36
#define L2_BLOCK_SIZE 104
#define L3_BLOCK_SIZE 1144
#define AVX_BLOCK_SIZE 4
// #define BLOCK_SIZE 719
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

static void do_block_l1 (int lda, int M_L1, int N_L1, int K_L1, double* A, double* B, double* C)
{
    double *buffer_A = (double*) _mm_malloc(L1_BLOCK_SIZE * L1_BLOCK_SIZE * sizeof(double), 64);
    double *buffer_B = (double*) _mm_malloc(L1_BLOCK_SIZE * L1_BLOCK_SIZE * sizeof(double), 64);
    memset(buffer_A, 0, L1_BLOCK_SIZE * L1_BLOCK_SIZE * sizeof(double));
    memset(buffer_B, 0, L1_BLOCK_SIZE * L1_BLOCK_SIZE * sizeof(double));
    /* Matrix padding and buffering */
    for (int i = 0; i < M_L1; ++i)
        for (int k = 0; k < K_L1; ++k)
            buffer_A[i*L1_BLOCK_SIZE+k] = A[i*lda+k];
            
    for (int k = 0; k < K_L1; ++k)
        for (int j = 0; j < N_L1; ++j)
        {
                #ifdef TRANSPOSE
            buffer_B[k*L1_BLOCK_SIZE+j] =  B[j*lda+k];
                #else
            buffer_B[k*L1_BLOCK_SIZE+j] =  B[k*lda+j];
                #endif
        }
    
  /* For each row i of A */
    for (int i = 0; i < M_L1; i+=AVX_BLOCK_SIZE)
      /* For each column j of B */
      for (int j = 0; j < N_L1; j+=AVX_BLOCK_SIZE)
      {
          /* c: AVX_BLOCK_SIZE * 4 */
          register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C+i*lda+j);
          register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+(i+1)*lda+j);
          register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+(i+2)*lda+j);
          register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C+(i+3)*lda+j);
          
          for (int k = 0; k < K_L1; k+=4)
           /*4 here 256/sizeof(double)/8=4 */
              for (int kk=0; kk<AVX_BLOCK_SIZE;kk++)
              {
                  register __m256d a0x = _mm256_broadcast_sd(buffer_A+i*L1_BLOCK_SIZE+k+kk);
                  register __m256d a1x = _mm256_broadcast_sd(buffer_A+(i+1)*L1_BLOCK_SIZE+k+kk);
                  register __m256d a2x = _mm256_broadcast_sd(buffer_A+(i+2)*L1_BLOCK_SIZE+k+kk);
                  register __m256d a3x = _mm256_broadcast_sd(buffer_A+(i+3)*L1_BLOCK_SIZE+k+kk);
                  
                  register __m256d b = _mm256_loadu_pd(buffer_B+(k+kk)*L1_BLOCK_SIZE+j);

                  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
                  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
                  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
                  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
              }
          
          _mm256_storeu_pd(C+i*lda+j, c00_c01_c02_c03);
          _mm256_storeu_pd(C+(i+1)*lda+j, c10_c11_c12_c13);
          _mm256_storeu_pd(C+(i+2)*lda+j, c20_c21_c22_c23);
          _mm256_storeu_pd(C+(i+3)*lda+j, c30_c31_c32_c33);
          
      }
    
    _mm_free(buffer_A);
    _mm_free(buffer_B);
}

static void do_block_l2 (int lda, int M_L2, int N_L2, int K_L2, double* A, double* B, double* C)
{
    /* For each block-row of A */
      for (int i = 0; i < M_L2; i += L1_BLOCK_SIZE)
        /* For each block-column of B */
        for (int j = 0; j < N_L2; j += L1_BLOCK_SIZE)
          /* Accumulate block dgemms into block of C */
          for (int k = 0; k < K_L2; k += L1_BLOCK_SIZE)
          {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M_L1 = min (L1_BLOCK_SIZE, M_L2-i);
        int N_L1 = min (L1_BLOCK_SIZE, N_L2-j);
        int K_L1 = min (L1_BLOCK_SIZE, K_L2-k);

        /* Perform individual block dgemm */
    #ifdef TRANSPOSE
        do_block_l1(lda, M_L1, N_L1, K_L1, A + i*lda + k, B + j*lda + k, C + i*lda + j);
    #else
        do_block_l1(lda, M_L1, N_L1, K_L1, A + i*lda + k, B + k*lda + j, C + i*lda + j);
    #endif
          }
}

static void do_block_l3 (int lda, int M_L3, int N_L3, int K_L3, double* A, double* B, double* C)
{
    /* For each block-row of A */
    for (int i = 0; i < M_L3; i += L2_BLOCK_SIZE)
      /* For each block-column of B */
      for (int j = 0; j < N_L3; j += L2_BLOCK_SIZE)
        /* Accumulate block dgemms into block of C */
        for (int k = 0; k < K_L3; k += L2_BLOCK_SIZE)
        {
      /* Correct block dimensions if block "goes off edge of" the matrix */
      int M_L2 = min (L2_BLOCK_SIZE, M_L3-i);
      int N_L2 = min (L2_BLOCK_SIZE, N_L3-j);
      int K_L2 = min (L2_BLOCK_SIZE, K_L3-k);

      /* Perform individual block dgemm */
  #ifdef TRANSPOSE
      do_block_l2(lda, M_L2, N_L2, K_L2, A + i*lda + k, B + j*lda + k, C + i*lda + j);
  #else
      do_block_l2(lda, M_L2, N_L2, K_L2, A + i*lda + k, B + k*lda + j, C + i*lda + j);
  #endif
        }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
#ifdef TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
    }
#endif
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += L3_BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += L3_BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += L3_BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M_L3 = min (L3_BLOCK_SIZE, lda-i);
	int N_L3 = min (L3_BLOCK_SIZE, lda-j);
	int K_L3 = min (L3_BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
#ifdef TRANSPOSE
	do_block_l3(lda, M_L3, N_L3, K_L3, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
	do_block_l3(lda, M_L3, N_L3, K_L3, A + i*lda + k, B + k*lda + j, C + i*lda + j);
#endif
      }
#if TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
    }
#endif
}
