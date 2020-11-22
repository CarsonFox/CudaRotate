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
