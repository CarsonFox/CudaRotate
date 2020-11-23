#include <iostream>
#include <cassert>

#include "Image.hpp"

__global__ void rotateNaive(Pixel *in, Pixel *out) {
    const auto i = blockIdx.x * 32 + threadIdx.x;
    const auto j = blockIdx.y * 32 + threadIdx.y;
    out[j * 32 + i] = in[i * 32 + j];
}

void checkErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    //Read image from stdin
    Image hostImageIn, hostImageOut;
    std::cin >> hostImageIn;

    //Allocate space for input and output images on device
    Image *devImageIn, *devImageOut;
    checkErrors(cudaMalloc(&devImageIn, sizeof(Image)));
    checkErrors(cudaMalloc(&devImageOut, sizeof(Image)));

    //Send input image to device
    checkErrors(cudaMemcpy(devImageIn, &hostImageIn, sizeof(Image), cudaMemcpyHostToDevice));

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(32, 32, 1);

    //Call kernel to rotate image
    rotateNaive<<<gridDim, blockDim>>>((Pixel *)devImageIn, (Pixel *)devImageOut);
    checkErrors(cudaPeekAtLastError());

    //Send rotated image back to host
    checkErrors(cudaMemcpy(&hostImageOut, devImageOut, sizeof(Image), cudaMemcpyDeviceToHost));

    checkErrors(cudaDeviceSynchronize());

    //Check that the image was rotated
    assert(isRotated(hostImageIn, hostImageOut));

    //Write rotated image to stdout
//    std::cout << hostImageOut;

    //Cleanup
    cudaFree(devImageIn);
    cudaFree(devImageOut);

    return 0;
}
