#include <iostream>

#include "Image.hpp"

__global__ void rotateNaive(Pixel *in, Pixel *out) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        out[i * IMAGE_SIZE + threadIdx.x] = in[threadIdx.x * IMAGE_SIZE + i];
    }
}

void checkErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    auto hostImage = new Image();
    std::cin >> *hostImage;

    Image *devImageIn, *devImageOut;
    checkErrors(cudaMalloc(&devImageIn, sizeof(Image)));
    checkErrors(cudaMalloc(&devImageOut, sizeof(Image)));

    checkErrors(cudaMemcpy(devImageIn, hostImage, sizeof(Image), cudaMemcpyHostToDevice));

    rotateNaive<<<1, IMAGE_SIZE>>>((Pixel *)devImageIn, (Pixel *)devImageOut);
    checkErrors(cudaPeekAtLastError());

    checkErrors(cudaMemcpy(hostImage, devImageIn, sizeof(Image), cudaMemcpyDeviceToHost));

    checkErrors(cudaDeviceSynchronize());

    std::cout << *hostImage;

    delete hostImage;
    cudaFree(devImageIn);

    return 0;
}
