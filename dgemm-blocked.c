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

#if defined(AVX512)
#define L1_BLOCK_SIZE 20
#define L2_BLOCK_SIZE 200
#define L3_BLOCK_SIZE 1240
#define AVX_BLOCK_SIZE_W 20
#define AVX_BLOCK_SIZE_H 4
#else
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

static void do_block_l1 (int buffer_size, int M_L1, int N_L1, int K_L1, double* buffer_A, double* buffer_B, double* buffer_C)
{
    /* For each row i of A */
    for (int i = 0; i < M_L1; i+=AVX_BLOCK_SIZE_H)
    /* For each column j of B */
        for (int j = 0; j < N_L1; j+=AVX_BLOCK_SIZE_W)
        {
                #ifdef AVX512
            /* AVX_BLOCK_SIZE_H * AVX_BLOCK_SIZE_W */
               register __m512d c00 = _mm512_load_pd(buffer_C+i*buffer_size+j);
               register __m512d c08 = _mm512_load_pd(buffer_C+i*buffer_size+j+8);
               register __m512d c016 = _mm512_load_pd(buffer_C+i*buffer_size+j+16);
               register __m512d c024 = _mm512_load_pd(buffer_C+i*buffer_size+j+24);
               register __m512d c032 = _mm512_load_pd(buffer_C+i*buffer_size+j+32);
               
               register __m512d c10 = _mm512_load_pd(buffer_C+(i+1)*buffer_size+j);
               register __m512d c18 = _mm512_load_pd(buffer_C+(i+1)*buffer_size+j+8);
               register __m512d c116 = _mm512_load_pd(buffer_C+(i+1)*buffer_size+j+16);
               register __m512d c124 = _mm512_load_pd(buffer_C+(i+1)*buffer_size+j+24);
               register __m512d c132 = _mm512_load_pd(buffer_C+(i+1)*buffer_size+j+32);
            
               register __m512d c20 = _mm512_load_pd(buffer_C+(i+2)*buffer_size+j);
               register __m512d c28 = _mm512_load_pd(buffer_C+(i+2)*buffer_size+j+8);
               register __m512d c216 = _mm512_load_pd(buffer_C+(i+2)*buffer_size+j+16);
               register __m512d c224 = _mm512_load_pd(buffer_C+(i+2)*buffer_size+j+24);
               register __m512d c232 = _mm512_load_pd(buffer_C+(i+2)*buffer_size+j+32);
                
            
               register __m512d c30 = _mm512_load_pd(buffer_C+(i+3)*buffer_size+j);
               register __m512d c38 = _mm512_load_pd(buffer_C+(i+3)*buffer_size+j+8);
               register __m512d c316 = _mm512_load_pd(buffer_C+(i+3)*buffer_size+j+16);
               register __m512d c324 = _mm512_load_pd(buffer_C+(i+3)*buffer_size+j+24);
               register __m512d c332 = _mm512_load_pd(buffer_C+(i+3)*buffer_size+j+32);
               
               for (int k = 0; k < K_L1; k+=1)
               {
                       register __m512d a0x = _mm512_broadcast_f64x4(_mm256_broadcast_sd(buffer_A+i*buffer_size+k));
                       register __m512d a1x = _mm512_broadcast_f64x4(_mm256_broadcast_sd(buffer_A+(i+1)*buffer_size+k));
                       register __m512d a2x = _mm512_broadcast_f64x4(_mm256_broadcast_sd(buffer_A+(i+2)*buffer_size+k));
                       register __m512d a3x = _mm512_broadcast_f64x4(_mm256_broadcast_sd(buffer_A+(i+3)*buffer_size+k));
                       
                       register __m512d b0 = _mm512_load_pd(buffer_B+k*buffer_size+j);
                       register __m512d b8 = _mm512_load_pd(buffer_B+k*buffer_size+j+8);
                       register __m512d b16 = _mm512_load_pd(buffer_B+k*buffer_size+j+16);
                       register __m512d b24 = _mm512_load_pd(buffer_B+k*buffer_size+j+24);
                       register __m512d b32 = _mm512_load_pd(buffer_B+k*buffer_size+j+32);
                       
                       c00 = _mm512_fmadd_pd(a0x, b0, c00);
                       c10 = _mm512_fmadd_pd(a1x, b0, c10);
                       c20 = _mm512_fmadd_pd(a2x, b0, c20);
                       c30 = _mm512_fmadd_pd(a3x, b0, c30);
                       
                       c08 = _mm512_fmadd_pd(a0x, b8, c08);
                       c18 = _mm512_fmadd_pd(a1x, b8, c18);
                       c28 = _mm512_fmadd_pd(a2x, b8, c28);
                       c38 = _mm512_fmadd_pd(a3x, b8, c38);
                   
                       c016 = _mm512_fmadd_pd(a0x, b16, c016);
                       c116 = _mm512_fmadd_pd(a1x, b16, c116);
                       c216 = _mm512_fmadd_pd(a2x, b16, c216);
                       c316 = _mm512_fmadd_pd(a3x, b16, c316);
                   
                       c024 = _mm512_fmadd_pd(a0x, b24, c024);
                       c124 = _mm512_fmadd_pd(a1x, b24, c124);
                       c224 = _mm512_fmadd_pd(a2x, b24, c224);
                       c324 = _mm512_fmadd_pd(a3x, b24, c324);
                     
                       c032 = _mm512_fmadd_pd(a0x, b32, c032);
                       c132 = _mm512_fmadd_pd(a1x, b32, c132);
                       c232 = _mm512_fmadd_pd(a2x, b32, c232);
                       c332 = _mm512_fmadd_pd(a3x, b32, c332);
                   
               }
               
               _mm512_store_pd(buffer_C+i*buffer_size+j, c00);
               _mm512_store_pd(buffer_C+(i+1)*buffer_size+j, c10);
               _mm512_store_pd(buffer_C+(i+2)*buffer_size+j, c20);
               _mm512_store_pd(buffer_C+(i+3)*buffer_size+j, c30);
               
               _mm512_store_pd(buffer_C+i*buffer_size+j+8, c08);
               _mm512_store_pd(buffer_C+(i+1)*buffer_size+j+8, c18);
               _mm512_store_pd(buffer_C+(i+2)*buffer_size+j+8, c28);
               _mm512_store_pd(buffer_C+(i+3)*buffer_size+j+8, c38);
               
               _mm512_store_pd(buffer_C+i*buffer_size+j+16, c016);
               _mm512_store_pd(buffer_C+(i+1)*buffer_size+j+16, c116);
               _mm512_store_pd(buffer_C+(i+2)*buffer_size+j+16, c216);
               _mm512_store_pd(buffer_C+(i+3)*buffer_size+j+16, c316);
            
               _mm512_store_pd(buffer_C+i*buffer_size+j+24, c024);
               _mm512_store_pd(buffer_C+(i+1)*buffer_size+j+24, c124);
               _mm512_store_pd(buffer_C+(i+2)*buffer_size+j+24, c224);
               _mm512_store_pd(buffer_C+(i+3)*buffer_size+j+24, c324);
             
               _mm512_store_pd(buffer_C+i*buffer_size+j+32, c032);
               _mm512_store_pd(buffer_C+(i+1)*buffer_size+j+32, c132);
               _mm512_store_pd(buffer_C+(i+2)*buffer_size+j+32, c232);
               _mm512_store_pd(buffer_C+(i+3)*buffer_size+j+32, c332);
            
                #else
                  /* AVX_BLOCK_SIZE_H * AVX_BLOCK_SIZE_W */
            register __m256d c00_c01_c02_c03 = _mm256_load_pd(buffer_C+i*buffer_size+j);
            register __m256d c04_c05_c06_c07 = _mm256_load_pd(buffer_C+i*buffer_size+j+4);
            register __m256d c08_c09_c00_c01 = _mm256_load_pd(buffer_C+i*buffer_size+j+8);
            register __m256d c10_c11_c12_c13 = _mm256_load_pd(buffer_C+(i+1)*buffer_size+j);
            register __m256d c14_c15_c16_c17 = _mm256_load_pd(buffer_C+(i+1)*buffer_size+j+4);
            register __m256d c18_c19_c10_c11 = _mm256_load_pd(buffer_C+(i+1)*buffer_size+j+8);
            register __m256d c20_c21_c22_c23 = _mm256_load_pd(buffer_C+(i+2)*buffer_size+j);
            register __m256d c24_c25_c26_c27 = _mm256_load_pd(buffer_C+(i+2)*buffer_size+j+4);
            register __m256d c28_c29_c20_c21 = _mm256_load_pd(buffer_C+(i+2)*buffer_size+j+8);
            
            for (int k = 0; k < K_L1; k+=1)
            {
                    register __m256d a0x = _mm256_broadcast_sd(buffer_A+i*buffer_size+k);
                    register __m256d a1x = _mm256_broadcast_sd(buffer_A+(i+1)*buffer_size+k);
                    register __m256d a2x = _mm256_broadcast_sd(buffer_A+(i+2)*buffer_size+k);
                    
                    register __m256d b0123 = _mm256_load_pd(buffer_B+k*buffer_size+j);
                    register __m256d b4567 = _mm256_load_pd(buffer_B+k*buffer_size+j+4);
                    register __m256d b8901 = _mm256_load_pd(buffer_B+k*buffer_size+j+8);
                    
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
            
            _mm256_store_pd(buffer_C+i*buffer_size+j, c00_c01_c02_c03);
            _mm256_store_pd(buffer_C+(i+1)*buffer_size+j, c10_c11_c12_c13);
            _mm256_store_pd(buffer_C+(i+2)*buffer_size+j, c20_c21_c22_c23);
            
            _mm256_store_pd(buffer_C+i*buffer_size+j+4, c04_c05_c06_c07);
            _mm256_store_pd(buffer_C+(i+1)*buffer_size+j+4, c14_c15_c16_c17);
            _mm256_store_pd(buffer_C+(i+2)*buffer_size+j+4, c24_c25_c26_c27);
            
            _mm256_store_pd(buffer_C+i*buffer_size+j+8, c08_c09_c00_c01);
            _mm256_store_pd(buffer_C+(i+1)*buffer_size+j+8, c18_c19_c10_c11);
            _mm256_store_pd(buffer_C+(i+2)*buffer_size+j+8, c28_c29_c20_c21);
                #endif
        }
}

static void do_block_l2 (int buffer_size, int M_L2, int N_L2, int K_L2, double* restrict buffer_A, double* restrict buffer_B, double* restrict buffer_C)
{
    /* For each block-row of A */
    for (int i = 0; i < M_L2; i += L1_BLOCK_SIZE)
    {
        int M_L1 = min (L1_BLOCK_SIZE, M_L2-i);
    /* For each block-column of B */
        for (int j = 0; j < N_L2; j += L1_BLOCK_SIZE)
        {
            int N_L1 = min (L1_BLOCK_SIZE, N_L2-j);
        /* Accumulate block dgemms into block of C */
            for (int k = 0; k < K_L2; k += L1_BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int K_L1 = min (L1_BLOCK_SIZE, K_L2-k);
                /* Perform individual block dgemm */
                do_block_l1(buffer_size, M_L1, N_L1, K_L1, buffer_A + i*buffer_size + k, buffer_B + k*buffer_size + j, buffer_C + i*buffer_size + j);

            }
        }
    }
}

static void do_block_l3 (int buffer_size, int M_L3, int N_L3, int K_L3, double* restrict buffer_A, double* restrict buffer_B, double* restrict buffer_C)
{
    /* For each block-row of A */
    for (int i = 0; i < M_L3; i += L2_BLOCK_SIZE)
    {
        int M_L2 = min (L2_BLOCK_SIZE, M_L3-i);
    /* For each block-column of B */
        for (int j = 0; j < N_L3; j += L2_BLOCK_SIZE)
        {
            int N_L2 = min (L2_BLOCK_SIZE, N_L3-j);
        /* Accumulate block dgemms into block of C */
            for (int k = 0; k < K_L3; k += L2_BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int K_L2_0 = min (L2_BLOCK_SIZE, K_L3-k);
                /* Perform individual block dgemm */
                do_block_l2(buffer_size, M_L2, N_L2, K_L2_0, buffer_A + i*buffer_size + k, buffer_B + k*buffer_size + j, buffer_C + i*buffer_size + j);
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{
    /* Matrix padding and buffering */
    int  buffer_size = lda + AVX_BLOCK_SIZE_W - lda % AVX_BLOCK_SIZE_W;
    
    double *buffer_A = (double*) _mm_malloc(buffer_size * buffer_size * sizeof(double), 1024);
    double *buffer_B = (double*) _mm_malloc(buffer_size * buffer_size * sizeof(double), 1024);
    double *buffer_C = (double*) _mm_malloc(buffer_size * buffer_size * sizeof(double), 1024);
   
    
    for (int i = 0; i < lda; ++i)
        for (int j = 0; j < lda; ++j) {
            buffer_A[i*buffer_size+j] = A[i*lda+j];
            buffer_B[i*buffer_size+j] = B[i*lda+j];
            buffer_C[i*buffer_size+j] = 0.0;
        }
    
    /* For each block-row of A */
    for (int i = 0; i < lda; i += L3_BLOCK_SIZE)
    {
        int M_L3 = min (L3_BLOCK_SIZE, lda-i);
    /* For each block-column of B */
        for (int j = 0; j < lda; j += L3_BLOCK_SIZE)
        {
            int N_L3 = min (L3_BLOCK_SIZE, lda-j);
        /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += L3_BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int K_L3 = min (L3_BLOCK_SIZE, lda-k);
                /* Perform individual block dgemm */
                do_block_l3(buffer_size, M_L3, N_L3, K_L3, buffer_A + i*buffer_size + k, buffer_B + k*buffer_size + j, buffer_C + i*buffer_size + j);
                
            }
        }
    }
    
    for (int i = 0; i < lda; ++i)
        for (int j = 0; j < lda; ++j)
            C[i*lda+j] = buffer_C[i*buffer_size+j];
    
    _mm_free(buffer_A);
    _mm_free(buffer_B);
    _mm_free(buffer_C);

}
