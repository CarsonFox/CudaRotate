#include <iostream>

#include "Image.hpp"

int main() {
    auto inputImage = new Image(), outputImage = new Image();
    Image *d_image;
    cudaMalloc(&d_image, sizeof(Image));

    std::cin >> *inputImage;

    cudaMemcpy(d_image, inputImage, sizeof(Image), cudaMemcpyHostToDevice);

    //Kernel

    cudaMemcpy(outputImage, d_image, sizeof(Image), cudaMemcpyDeviceToHost);

    std::cout << *outputImage;

    delete inputImage;
    delete outputImage;
    cudaFree(d_image);
    return 0;
}
