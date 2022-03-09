#import <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#import "gpu_kernels.h"


__global__
void kernel(){
    printf("hello from the kernel!\n");
}

void gpuf(){
    printf("hello from the gpu file!\n");
    cudaDeviceSynchronize();
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();
}