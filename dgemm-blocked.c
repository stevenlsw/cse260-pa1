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
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define L1_BLOCK_SIZE 36
#define L2_BLOCK_SIZE 108
#define L3_BLOCK_SIZE 1140
#define AVX_BLOCK_SIZE_W 12
#define AVX_BLOCK_SIZE_H 3
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

static void do_block_l1 (int lda, int M_L1, int N_L1, int K_L1, double* A, double* B, double* C)
{
    /* For each row i of A */
    for (int i = 0; i < M_L1; i+=AVX_BLOCK_SIZE_H)
    /* For each column j of B */
        for (int j = 0; j < N_L1; j+=AVX_BLOCK_SIZE_W)
        {
                  /* AVX_BLOCK_SIZE_H * AVX_BLOCK_SIZE_W */
            register __m256d c00_c01_c02_c03 = _mm256_load_pd(C+i*lda+j);
            register __m256d c04_c05_c06_c07 = _mm256_loadu_pd(C+i*lda+j+4);
            register __m256d c08_c09_c00_c01 = _mm256_loadu_pd(C+i*lda+j+8);
            register __m256d c10_c11_c12_c13 = _mm256_load_pd(C+(i+1)*lda+j);
            register __m256d c14_c15_c16_c17 = _mm256_loadu_pd(C+(i+1)*lda+j+4);
            register __m256d c18_c19_c10_c11 = _mm256_loadu_pd(C+(i+1)*lda+j+8);
            register __m256d c20_c21_c22_c23 = _mm256_load_pd(C+(i+2)*lda+j);
            register __m256d c24_c25_c26_c27 = _mm256_loadu_pd(C+(i+2)*lda+j+4);
            register __m256d c28_c29_c20_c21 = _mm256_loadu_pd(C+(i+2)*lda+j+8);
            
            for (int k = 0; k < K_L1; k+=1)
            {
                    register __m256d a0x = _mm256_broadcast_sd(A+i*lda+k);
                    register __m256d a1x = _mm256_broadcast_sd(A+(i+1)*lda+k);
                    register __m256d a2x = _mm256_broadcast_sd(A+(i+2)*lda+k);
                    
                    register __m256d b0123 = _mm256_load_pd(B+k*lda+j);
                    register __m256d b4567 = _mm256_loadu_pd(B+k*lda+j+4);
                    register __m256d b8901 = _mm256_loadu_pd(B+k*lda+j+8);
                    
                    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b0123, c00_c01_c02_c03);
                    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b0123, c10_c11_c12_c13);
                    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b0123, c20_c21_c22_c23);
                    
                    c04_c05_c06_c07 = _mm256_fmadd_pd(a0x, b4567, c04_c05_c06_c07);
                    c14_c15_c16_c17 = _mm256_fmadd_pd(a1x, b4567, c14_c15_c16_c17);
                    c24_c25_c26_c27 = _mm256_fmadd_pd(a2x, b4567, c24_c25_c26_c27);
                
                    c08_c09_c00_c01 = _mm256_fmadd_pd(a0x, b8901, c08_c09_c00_c01);
                    c18_c19_c10_c11 = _mm256_fmadd_pd(a1x, b8901, c18_c19_c10_c11);
                    c28_c29_c20_c21 = _mm256_fmadd_pd(a2x, b8901, c28_c29_c20_c21);
                
                
            }
            
            _mm256_store_pd(C+i*lda+j, c00_c01_c02_c03);
            _mm256_store_pd(C+(i+1)*lda+j, c10_c11_c12_c13);
            _mm256_store_pd(C+(i+2)*lda+j, c20_c21_c22_c23);
            
            _mm256_store_pd(C+i*lda+j+4, c04_c05_c06_c07);
            _mm256_store_pd(C+(i+1)*lda+j+4, c14_c15_c16_c17);
            _mm256_store_pd(C+(i+2)*lda+j+4, c24_c25_c26_c27);
            
            _mm256_store_pd(C+i*lda+j+8, c08_c09_c00_c01);
            _mm256_store_pd(C+(i+1)*lda+j+8, c18_c19_c10_c11);
            _mm256_store_pd(C+(i+2)*lda+j+8, c28_c29_c20_c21);
            
        }
}

static void do_block_l2 (int lda, int M_L2, int N_L2, int K_L2, double* restrict A, double* restrict B, double* restrict C)
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

static void do_block_l3 (int lda, int M_L3, int N_L3, int K_L3, double* restrict A, double* restrict B, double* restrict C)
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
void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{
#ifdef TRANSPOSE
    for (int i = 0; i < lda; ++i)
        for (int j = i+1; j < lda; ++j) {
            double t = B[i*lda+j];
            B[i*lda+j] = B[j*lda+i];
            B[j*lda+i] = t;
        }
#endif
    
    /* Matrix padding and buffering */
    /* 12 = AVX_SIZE_H */
    int SIZE = lda + 12 - lda % 12;
    
    double *buffer_A = (double*) _mm_malloc(SIZE * SIZE * sizeof(double), 64);
    double *buffer_B = (double*) _mm_malloc(SIZE * SIZE * sizeof(double), 64);
    double *buffer_C = (double*) _mm_malloc(SIZE * SIZE * sizeof(double), 64);
    memset(buffer_A, 0, SIZE * SIZE * sizeof(double));
    memset(buffer_B, 0, SIZE * SIZE * sizeof(double));
    memset(buffer_C, 0, SIZE * SIZE * sizeof(double));
    
    for (int i = 0; i < lda; ++i)
        for (int j = 0; j < lda; ++j) {
            buffer_A[i*SIZE+j] = A[i*lda+j];
            buffer_B[i*SIZE+j] = B[i*lda+j];
            buffer_C[i*SIZE+j] = C[i*lda+j];
        }
    
    /* For each block-row of A */
    for (int i = 0; i < SIZE; i += L3_BLOCK_SIZE)
    /* For each block-column of B */
        for (int j = 0; j < SIZE; j += L3_BLOCK_SIZE)
        /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += L3_BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M_L3 = min (L3_BLOCK_SIZE, SIZE-i);
                int N_L3 = min (L3_BLOCK_SIZE, SIZE-j);
                int K_L3 = min (L3_BLOCK_SIZE, SIZE-k);
                
                /* Perform individual block dgemm */
#ifdef TRANSPOSE
                do_block_l3(SIZE, M_L3, N_L3, K_L3, buffer_A + i*SIZE + k, buffer_B + j*SIZE + k, buffer_C + i*SIZE + j);
#else
                do_block_l3(SIZE, M_L3, N_L3, K_L3, buffer_A + i*SIZE + k, buffer_B + k*SIZE + j, buffer_C + i*SIZE + j);
#endif
            }
    
    for (int i = 0; i < lda; ++i)
        for (int j = 0; j < lda; ++j)
            C[i*lda+j] = buffer_C[i*SIZE+j];
    
    _mm_free(buffer_A);
    _mm_free(buffer_B);
    _mm_free(buffer_C);
    
#if TRANSPOSE
    for (int i = 0; i < lda; ++i)
        for (int j = i+1; j < lda; ++j) {
            double t = B[i*lda+j];
            B[i*lda+j] = B[j*lda+i];
            B[j*lda+i] = t;
        }
#endif
}
