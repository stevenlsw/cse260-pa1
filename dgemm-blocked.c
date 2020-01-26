/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <immintrin.h>
#include <avx2intrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define L1_BLOCK_SIZE 36
#define L2_BLOCK_SIZE 104
#define L3_BLOCK_SIZE 1144
#define AVX_BLOCK_SIZE 8
// #define BLOCK_SIZE 719
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

static void do_block_l1 (int lda, int M_L1, int N_L1, int K_L1, double* A, double* B, double* C)
{

  /* For each row i of A */
    for (int i = 0; i < M_L1; i+=AVX_BLOCK_SIZE)
      /* For each column j of B */
      for (int j = 0; j < N_L1; j+=AVX_BLOCK_SIZE)
      {
          register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C+i*lda+j);
          register __m256d c04_c05_c06_c07 = _mm256_loadu_pd(C+i*lda+j+4);
          register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+(i+1)*lda+j);
          register __m256d c14_c15_c16_c17 = _mm256_loadu_pd(C+(i+1)*lda+j+4);
          register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+(i+2)*lda+j);
          register __m256d c24_c25_c26_c27 = _mm256_loadu_pd(C+(i+2)*lda+j+4);
          register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C+(i+3)*lda+j);
          register __m256d c34_c35_c36_c37 = _mm256_loadu_pd(C+(i+3)*lda+j+4);
          
          register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(C+(i+4)*lda+j);
          register __m256d c44_c45_c46_c47 = _mm256_loadu_pd(C+(i+4)*lda+j+4);
          register __m256d c50_c51_c52_c53 = _mm256_loadu_pd(C+(i+5)*lda+j);
          register __m256d c54_c55_c56_c57 = _mm256_loadu_pd(C+(i+5)*lda+j+4);
          register __m256d c60_c61_c62_c63 = _mm256_loadu_pd(C+(i+6)*lda+j);
          register __m256d c64_c65_c66_c67 = _mm256_loadu_pd(C+(i+6)*lda+j+4);
          register __m256d c70_c71_c72_c73 = _mm256_loadu_pd(C+(i+7)*lda+j);
          register __m256d c74_c75_c76_c77 = _mm256_loadu_pd(C+(i+7)*lda+j+4);
          
          for (int k = 0; k < K_L1; k+=4)
           /*4 here 256/sizeof(double)/8=4 */
              for (int kk=0; kk<AVX_BLOCK_SIZE;kk++)
              {
                  register __m256d a0x = _mm256_broadcast_sd(A+i*lda+k+kk);
                  register __m256d a1x = _mm256_broadcast_sd(A+(i+1)*lda+k+kk);
                  register __m256d a2x = _mm256_broadcast_sd(A+(i+2)*lda+k+kk);
                  register __m256d a3x = _mm256_broadcast_sd(A+(i+3)*lda+k+kk);
                  
                  register __m256d a4x = _mm256_broadcast_sd(A+(i+4)*lda+k+kk);
                  register __m256d a5x = _mm256_broadcast_sd(A+(i+5)*lda+k+kk);
                  register __m256d a6x = _mm256_broadcast_sd(A+(i+6)*lda+k+kk);
                  register __m256d a7x = _mm256_broadcast_sd(A+(i+7)*lda+k+kk);
                  
                  register __m256d b0123 = _mm256_loadu_pd(B+(k+kk)*lda+j);
                  register __m256d b4567 = _mm256_loadu_pd(B+(k+kk)*lda+j+4);
                  /* boundary padding */
                  if (j+AVX_BLOCK_SIZE>N_L1)
                  {
                      int e = N_L1 - j;
                      b0123[0] = b0123[0] * (e>0);
                      b0123[1] = b0123[1] * (e>1);
                      b0123[2] = b0123[2] * (e>2);
                      b0123[3] = b0123[3] * (e>3);
                      b4567[0] = b4567[0] * (e>4);
                      b4567[1] = b4567[1] * (e>5);
                      b4567[2] = b4567[2] * (e>6);
                      b4567[3] = b4567[3] * (e>7);
                  }

                  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0123, c00_c01_c02_c03);
                  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b0123, c10_c11_c12_c13);
                  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b0123, c20_c21_c22_c23);
                  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b0123, c30_c31_c32_c33);
                  
                  c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b0123, c40_c41_c42_c43);
                  c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b0123, c50_c51_c52_c53);
                  c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b0123, c60_c61_c62_c63);
                  c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b0123, c70_c71_c72_c73);
                  
                  c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b4567, c04_c05_c06_c07);
                  c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, b4567, c14_c15_c16_c17);
                  c24_c25_c26_c27 = _mm256_fmadd_pd(a2x, b4567, c24_c25_c26_c27);
                  c34_c35_c36_c37 = _mm256_fmadd_pd(a3x, b4567, c34_c35_c36_c37);
                  
                  c44_c45_c46_c47 = _mm256_fmadd_pd(a4x, b4567, c44_c45_c46_c47);
                  c54_c55_c56_c57 = _mm256_fmadd_pd(a5x, b4567, c54_c55_c56_c57);
                  c64_c65_c66_c67 = _mm256_fmadd_pd(a6x, b4567, c64_c65_c66_c67);
                  c74_c75_c76_c77 = _mm256_fmadd_pd(a7x, b4567, c74_c75_c76_c77);
                  
              }

          _mm256_storeu_pd(C+i*lda+j, c00_c01_c02_c03);
          _mm256_storeu_pd(C+i*lda+j+4, c04_c05_c06_c07);
          _mm256_storeu_pd(C+(i+1)*lda+j, c10_c11_c12_c13);
          _mm256_storeu_pd(C+(i+1)*lda+j+4, c14_c15_c16_c17);
          _mm256_storeu_pd(C+(i+2)*lda+j, c20_c21_c22_c23);
          _mm256_storeu_pd(C+(i+2)*lda+j+4, c24_c25_c26_c27);
          _mm256_storeu_pd(C+(i+3)*lda+j, c30_c31_c32_c33);
          _mm256_storeu_pd(C+(i+3)*lda+j+4, c34_c35_c36_c37);
          
          _mm256_storeu_pd(C+(i+4)*lda+j, c40_c41_c42_c43);
          _mm256_storeu_pd(C+(i+4)*lda+j+4, c44_c45_c46_c47);
          _mm256_storeu_pd(C+(i+5)*lda+j, c50_c51_c52_c53);
          _mm256_storeu_pd(C+(i+5)*lda+j+4, c54_c55_c56_c57);
          _mm256_storeu_pd(C+(i+6)*lda+j, c60_c61_c62_c63);
          _mm256_storeu_pd(C+(i+6)*lda+j+4, c64_c65_c66_c67);
          _mm256_storeu_pd(C+(i+7)*lda+j, c70_c71_c72_c73);
          _mm256_storeu_pd(C+(i+7)*lda+j+4, c74_c75_c76_c77);
          
      }
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
