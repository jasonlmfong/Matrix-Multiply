# Matrix-Multiply

The code goes through many different versions of matrix multiplication to see how different types of parallelism offers more performance enhancements.

The table below illustrates the speedup across different implementations and sizes of matrices in terms of billion floating point operations per second (GFLOPS/second). 

| Sizes     | faster1 | faster2 | faster3 | faster4 | faster5 | faster6 (4 threads) | faster6 (16 threads) | faster6 (64 threads) |
|-----------|---------|---------|---------|---------|---------|---------------------|----------------------|----------------------|
| 64x64     | 0.007   | 2.0     | 7.5     | 27      | 28      | 25                  | 21                   | 14                   |
| 320x320   | 0.007   | 1.5     | 6.7     | 17      | 30      | 78                  | 179                  | 157                  |
| 960x960   | 0.007   | 1.5     | 10.9    | 22      | 32      | 79                  | 267                  | 298                  |
| 4096x4096 | 0.007   | 0.6     | 1.4     | 1.9     | 22      | 33                  | 129                  | 329                  |

To put things into perspective, in faster1, 4096x4096 matrices would take almost 6 hours, while in faster6, 4096x4096 matrices would take less than a second.
