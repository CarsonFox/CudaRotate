#include <iostream>

#include "Image.hpp"

__global__ void doNothing() {}

void checkErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    auto inputImage = new Image(), outputImage = new Image();
    Image *d_image;
    checkErrors(cudaMalloc(&d_image, sizeof(Image)));

    std::cin >> *inputImage;

    checkErrors(cudaMemcpy(d_image, inputImage, sizeof(Image), cudaMemcpyHostToDevice));
    doNothing<<<1,1>>>();
    checkErrors(cudaMemcpy(outputImage, d_image, sizeof(Image), cudaMemcpyDeviceToHost));

    checkErrors(cudaDeviceSynchronize());

    std::cout << *outputImage;

    delete inputImage;
    delete outputImage;
    cudaFree(d_image);
    return 0;
}
