#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>

#include "Image.hpp"

void naive();

void tiled();

//Size of a tile in pixels (square root of image size)
constexpr int tileSize = 32;

int main() {
    tiled();
    //naive();
}

__global__ void rotateNaive(Pixel *in, Pixel *out) {
    out[blockIdx.x * IMAGE_SIZE + threadIdx.x] = in[threadIdx.x * IMAGE_SIZE + blockIdx.x];
}

void checkErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void tiled() {
    //Read image from stdin
    Image hostImageIn, hostImageOut;
    std::cin >> hostImageIn;

    //Allocate space for input and output images on device
    Image *devImageIn, *devImageOut;
    checkErrors(cudaMalloc(&devImageIn, sizeof(Image)));
    checkErrors(cudaMalloc(&devImageOut, sizeof(Image)));

    //Allocate tiles for each block
    const int numBlocks = IMAGE_SIZE / tileSize;
    std::vector<Pixel *> tiles(numBlocks, nullptr);
    for (Pixel *tile : tiles)
        checkErrors(cudaMalloc(&tile, tileSize * tileSize));

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
    for (Pixel *tile : tiles)
        cudaFree(tile);
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
