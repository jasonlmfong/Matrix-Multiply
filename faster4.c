//DGEMM matrix multiplication in C
//improved using loop unrolling (instruction level)

//about 2x as performant as the previous dgemm C implementation in faster3.c
//despite unrolling 4 times, the instruction count only doubles

//#include <x86intrin.h>
#include <immintrin.h>
#define UNROLL (4)

void dgemm (int n, double* A, double* B, double* C)
{
	for (int i = 0; i < n; i += UNROLL*8)
		for (int j = 0; j < n; ++j)
		{
            __m512d c[UNROLL];
            for (int r=0;r<UNROLL;r++)
                c[r] = _mm512_load_pd(C+i+r*8+j*n); //load array c with values

			for(int k = 0; k < n; k++)
                for (int r=0;r<UNROLL;r++)
                    c[r] = _mm512_fmadd_pd(_mm512_load_pd(A+n*k+r*8+i), _mm512_broadcastsd_pd(_mm_load_sd(B+j*n+k)), c[r]);

            for (int r=0;r<UNROLL;r++)
                _mm512_store_pd(C+i+r*8+j*n, c[r]); //store output C with values in array c
		}
}