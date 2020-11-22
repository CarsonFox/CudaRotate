#include <iostream>

#include "Image.hpp"

__global__ void kernel() {}

int main() {
    auto inputImage = new Image(), outputImage = new Image();

    std::cin >> *inputImage;

    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            (*outputImage)[i][j] = (*inputImage)[j][i];
        }
    }

    std::cout << *outputImage;

    return 0;
}
