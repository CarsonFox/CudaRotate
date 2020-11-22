#include <iostream>

#include "Image.hpp"

__global__ void add(int *a, int *b, int *out) {
    *out = *a + *b;
}

void cudaHello() {
    int a = 0, b = 0, out = 0;
    int *d_a, *d_b, *d_out;

    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_out, sizeof(int));

    a = 5, b = 11;

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    add<<<1,1>>>(d_a, d_b, d_out);

    cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a), cudaFree(d_b), cudaFree(d_out);

    std::cout << a << " + " << b << " = " << out << std::endl;
}

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
