//DGEMM matrix multiplication in C
//improved using AVX (advanced vector extension) and subword parallelism (data level)

//about 8x as performant as the previous dgemm C implementation in faster2.c
//because it computes 8 operations at the same time using parallelism

//#include <x86intrin.h>
#include <immintrin.h>

void dgemm (int n, double* A, double* B, double* C)
{
	for (int i = 0; i < n; i += 8) //increment by 8 since each load have 8 numbers
		for (int j = 0; j < n; ++j)
		{
            __m512d c0 = _mm512_load_pd(C+i+j*n); /* c0 = C[i][j] */
            // __m512d data type is a variable that holds 8 x 64 bits double precision floating point values
            // _mm512_load_pd() loads 8 values in parallel from matrix C to c0
            // C+i+j*n represents C[i+j*n]
			for(int k = 0; k < n; k++)
				c0 = _mm512_add_pd(c0, /* c0 += A[i][k] * B[k][j] */
                                    _mm512_mul_pd(_mm512_load_pd(A+i+k*n), _mm512_broadcastsd_pd(_mm_load_sd(B+j*n+k))));
                //broadcast makes 8 exact copies of the double precision number in the ZMM registers
                //then we multiply and add
            _mm512_store_pd(C+i+j*n, c0); /* C[i][j] = c0 */
            //parallelly store 8 values back from c0 to C
		}
}