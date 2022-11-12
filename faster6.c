//DGEMM matrix multiplication in C
//improved using multithreading, specifically openMP

//for small enough matrices (those that fit in the L1 cache, such as 64x64), the 48-thread version is about half as fast as in faster5.c
//about 6x to 9x as performant for intermediate-sized matrices (320x320 - 960-960) when compared to the previous dgemm C implementation in faster5.c
//about 16x as performant for large matrices (4096x4096) when compared to the previous dgemm C implementation in faster5.c

//#include <x86intrin.h>
#include <immintrin.h>
#define UNROLL (4)
#define BLOCKSIZE 32

void do_block (int n, int si, int sj, int sk, double *A, double *B, double *C)
{
    for (int i = si; i < si + BLOCKSIZE; i += UNROLL*8)
		for (int j = sj; j < sj + BLOCKSIZE; ++j)
		{
            __m512d c[UNROLL];
            for (int r=0;r<UNROLL;r++)
                c[r] = _mm512_load_pd(C+i+r*8+j*n); //load array c with values

			for(int k = sk; k < sk + BLOCKSIZE; k++)
                for (int r=0;r<UNROLL;r++)
                    c[r] = _mm512_fmadd_pd(_mm512_load_pd(A+n*k+r*8+i), _mm512_broadcastsd_pd(_mm_load_sd(B+j*n+k)), c[r]);

            for (int r=0;r<UNROLL;r++)
                _mm512_store_pd(C+i+r*8+j*n, c[r]); //store output C with values in array c
		}
}
void dgemm (int n, double* A, double* B, double* C)
{
#pragma omp parallel for //openMP multithreading
	for ( int sj = 0; sj < n; sj += BLOCKSIZE )
        for ( int si = 0; si < n; si += BLOCKSIZE )
            for ( int sk = 0; sk < n; sk += BLOCKSIZE )
                do_block(n, si, sj, sk, A, B, C);
}