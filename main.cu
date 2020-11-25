#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>

#include "Image.hpp"

void naive();
void tiled();
void checkErrors(cudaError_t err);

int main() {
    tiled();
    naive();
}

//Literal transpose operation
__global__ void rotateNaive(Pixel *in, Pixel *out) {
    out[blockIdx.x * IMAGE_SIZE + threadIdx.x] = in[threadIdx.x * IMAGE_SIZE + blockIdx.x];
}

//Transpose using tiles to decrease distance between strides.
//First transpose the pixels within a tile,
//then transpose the tiles themselves.
__global__ void writeToTiles(Pixel *in, Pixel *tiles) {
     auto in_i = blockIdx.x * blockDim.x + threadIdx.x;
     auto in_j = blockIdx.y * blockDim.y + threadIdx.y;
     auto in_index = in_i + in_j * IMAGE_SIZE;

     auto tile_i = blockIdx.x * blockDim.x + threadIdx.y;
     auto tile_j = blockIdx.y * blockDim.y + threadIdx.x;
     auto tile_index = tile_i + tile_j * IMAGE_SIZE;

     tiles[tile_index] = in[in_index];
}

__global__ void readFromTiles(Pixel *tiles, Pixel *out) {
    auto tile_i = blockIdx.x * blockDim.x + threadIdx.x;
    auto tile_j = blockIdx.y * blockDim.y + threadIdx.y;
    auto tile_index = tile_i + tile_j * IMAGE_SIZE;

    auto out_i = blockIdx.y * blockDim.x + threadIdx.x;
    auto out_j = blockIdx.x * blockDim.y + threadIdx.y;
    auto out_index = out_i + out_j * IMAGE_SIZE;

    out[out_index] = tiles[tile_index];
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
    const dim3 dim(32, 32, 1);

    //Send input image to device
    checkErrors(cudaMemcpy(devImageIn, &hostImageIn, sizeof(Image), cudaMemcpyHostToDevice));

    //Call kernels to rotate image
    writeToTiles<<<dim, dim>>>((Pixel *) devImageIn, tiles);
    readFromTiles<<<dim, dim>>>(tiles, (Pixel *) devImageOut);
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
