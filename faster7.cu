#include <stdio.h>

#define N  4096

__global__ void matrixMulGPU( int * a, int * b, int * c )
{
  /*
   * Build out this kernel.
   */
  
  int val = 0;
  
  int rowIndexWithinTheGrid = threadIdx.x + blockIdx.x * blockDim.x;
  int rowGridStride = gridDim.x * blockDim.x;
  
  int colIndexWithinTheGrid = threadIdx.y + blockIdx.y * blockDim.y;
  int colGridStride = gridDim.y * blockDim.y;
  
  for(int row = rowIndexWithinTheGrid; row < N; row += rowGridStride)
  {
    for(int col = colIndexWithinTheGrid; col < N; col += colGridStride)
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
  }
}

/*
 * This CPU function already works, and will run to create a solution matrix
 * against which to verify your work building out the matrixMulGPU kernel.
 */

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  int *a, *b, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory; create sample 2D matrices
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_gpu[row*N + col] = 0;
    }

  // prefecth to GPU memory for operation
  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c_gpu, size, deviceId);
  
  /*
   * Assign `threads_per_block` and `number_of_blocks` 2D values
   * that can be used in matrixMulGPU above.
   */

  dim3 number_of_blocks((numberOfSMs * 32), (numberOfSMs * 32), 1);
  dim3 threads_per_block(32, 32, 1);
  
  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu );

  cudaDeviceSynchronize(); // wait for it tofinish

  printf("Success!\n");

  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_gpu );
}
