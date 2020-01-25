/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define L1_BLOCK_SIZE 36
#define L2_BLOCK_SIZE 104
#define L3_BLOCK_SIZE 1144
// #define BLOCK_SIZE 719
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */

static void do_block_l1 (int M_L1, int N_L1, int K_L1, double* A, double* B, double* C)
{
  /* For each row i of A */
    for (int i = 0; i < M_L1; ++i)
      /* For each column j of B */
      for (int j = 0; j < N_L1; ++j)
      {
        /* Compute C(i,j) */
        double cij = C[i*L2_BLOCK_SIZE+j];
        for (int k = 0; k < K_L1; ++k)
  #ifdef TRANSPOSE
      cij += A[i*L2_BLOCK_SIZE+k] * B[j*L2_BLOCK_SIZE+k];
  #else
      cij += A[i*L2_BLOCK_SIZE+k] * B[k*L2_BLOCK_SIZE+j];
  #endif
        C[i*L2_BLOCK_SIZE+j] = cij;
      }
}

static void do_block_l2 (int M_L2, int N_L2, int K_L2, double* A, double* B, double* C)
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
        do_block_l1(M_L1, N_L1, K_L1, A + i*L2_BLOCK_SIZE + k, B + j*L2_BLOCK_SIZE + k, C + i*L2_BLOCK_SIZE + j);
    #else
        do_block_l1(M_L1, N_L1, K_L1, A + i*L2_BLOCK_SIZE + k, B + k*L2_BLOCK_SIZE + j, C + i*L2_BLOCK_SIZE + j);
    #endif
          }
}

static void do_block_l3 (int M_L3, int N_L3, int K_L3, double* A, double* B, double* C)
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
      do_block_l2(M_L2, N_L2, K_L2, A + i*L3_BLOCK_SIZE + k, B + j*L3_BLOCK_SIZE + k, C + i*L3_BLOCK_SIZE + j);
  #else
      do_block_l2(M_L2, N_L2, K_L2, A + i*L3_BLOCK_SIZE + k, B + k*L3_BLOCK_SIZE + j, C + i*L3_BLOCK_SIZE + j);
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
	do_block_l3(M_L3, N_L3, K_L3, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
	do_block_l3(M_L3, N_L3, K_L3, A + i*lda + k, B + k*lda + j, C + i*lda + j);
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
