#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>

#include "Image.hpp"

void naive();
void tiled();
void checkErrors(cudaError_t err);

//Size of a tile in pixels (square root of image size)
constexpr int tileSize = 32;

int main() {
    tiled();
    //naive();
}

__global__ void rotateNaive(Pixel *in, Pixel *out) {
    out[blockIdx.x * IMAGE_SIZE + threadIdx.x] = in[threadIdx.x * IMAGE_SIZE + blockIdx.x];
}

void tiled() {
    //Read image from stdin
    Image hostImageIn, hostImageOut;
    std::cin >> hostImageIn;

    //Allocate space for input and output images on device
    Image *devImageIn, *devImageOut;
    checkErrors(cudaMalloc(&devImageIn, sizeof(Image)));
    checkErrors(cudaMalloc(&devImageOut, sizeof(Image)));

    //Allocate tiles for each block.
    //Although memory will be allocated contiguously, treating it like
    //an array of tiles will result in smaller strides.
    Pixel *tiles;
    checkErrors(cudaMalloc(&tiles, sizeof(Image)));

    //Dimensions of grid and tiles
    const dim3 dim(tileSize, tileSize, 1);

    //Send input image to device
    checkErrors(cudaMemcpy(devImageIn, &hostImageIn, sizeof(Image), cudaMemcpyHostToDevice));

    //Call kernel to rotate image
    rotateNaive<<<IMAGE_SIZE, IMAGE_SIZE>>>((Pixel *) devImageIn, (Pixel *) devImageOut);
    checkErrors(cudaPeekAtLastError());

    //Send rotated image back to host
    checkErrors(cudaMemcpy(&hostImageOut, devImageOut, sizeof(Image), cudaMemcpyDeviceToHost));

    checkErrors(cudaDeviceSynchronize());

    //Check that the image was rotated
    assert(isRotated(hostImageIn, hostImageOut));

    //Write rotated image to stdout
    //std::cout << hostImageOut;

    //Cleanup
    cudaFree(devImageIn);
    cudaFree(devImageOut);
    cudaFree(tiles);
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
    rotateNaive<<<IMAGE_SIZE, IMAGE_SIZE>>>((Pixel *) devImageIn, (Pixel *) devImageOut);
    checkErrors(cudaPeekAtLastError());

    //Send rotated image back to host
    checkErrors(cudaMemcpy(&hostImageOut, devImageOut, sizeof(Image), cudaMemcpyDeviceToHost));

    checkErrors(cudaDeviceSynchronize());

    //Check that the image was rotated
    assert(isRotated(hostImageIn, hostImageOut));

    //Write rotated image to stdout
    //std::cout << hostImageOut;

    //Cleanup
    cudaFree(devImageIn);
    cudaFree(devImageOut);
}

void checkErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
