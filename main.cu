#include <iostream>
#include <cassert>

#include "Image.hpp"

void naive();
void tiled();

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
    naive();
}

void tiled() {
}

void naive() {
    //Read image from stdin
    Image hostImageIn, hostImageOut;
    std::cin >> hostImageIn;

    //Allocate space for input and output images on device
    Image *devImageIn, *devImageOut;
    checkErrors(cudaMalloc(&devImageIn, sizeof(Image)));
    checkErrors(cudaMalloc(&devImageOut, sizeof(Image)));

    //Send input image to device
    checkErrors(cudaMemcpy(devImageIn, &hostImageIn, sizeof(Image), cudaMemcpyHostToDevice));

    //Call kernel to rotate image
    rotateNaive<<<1, IMAGE_SIZE>>>((Pixel *)devImageIn, (Pixel *)devImageOut);
    checkErrors(cudaPeekAtLastError());

    //Send rotated image back to host
    checkErrors(cudaMemcpy(&hostImageOut, devImageOut, sizeof(Image), cudaMemcpyDeviceToHost));

    checkErrors(cudaDeviceSynchronize());

    //Check that the image was rotated
    assert(isRotated(hostImageIn, hostImageOut));

    //Write rotated image to stdout
    std::cout << hostImageOut;

    //Cleanup
    cudaFree(devImageIn);
    cudaFree(devImageOut);
}
