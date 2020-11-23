#include "Image.hpp"

std::istream& operator>>(std::istream &is, Image &image) {
    for (auto &&row : image) {
        for (auto &&pixel : row) {
            is >> pixel;
        }
    }
    return is;
}

std::ostream& operator<<(std::ostream &os, const Image &image) {
    for (auto &&row : image) {
        for (auto &&pixel : row) {
            os << pixel;
        }
    }
    return os;
}

bool isRotated(const Image &in, const Image &out) {
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            if (in[i][j] != out[j][i]) {
                return false;
            }
        }
    }
    return true;
}
