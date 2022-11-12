//DGEMM matrix multiplication in C
//improved using cache blocking (memory level), using spatial and temporal locality

//do_block function is essentially the dgemm function previously, with new parameters to specify the starting positions of the submatrices
//The two inner loops of the do_block now compute in steps of size BLOCKSIZE rather than the full length of B and C.
//computing on submatrices ensures elemnets being accessed can fit in the cache and reduce cache misses
//gcc compiler can remove the function overhead by inlining the function call

//for small enough matrices (those that fit in the L1 cache), it does not offer much optimization, about the same performance as in faster4.c
//about 1.5x to 1.7x as performant for intermediate-sized matrices (320x320 - 960-960) when compared to the previous dgemm C implementation in faster4.c
//about 10x as performant for large matrices (4096x4096) when compared to the previous dgemm C implementation in faster4.c

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
	for ( int sj = 0; sj < n; sj += BLOCKSIZE )
        for ( int si = 0; si < n; si += BLOCKSIZE )
            for ( int sk = 0; sk < n; sk += BLOCKSIZE )
                do_block(n, si, sj, sk, A, B, C);
}