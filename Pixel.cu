#include "Pixel.hpp"

std::istream &operator>>(std::istream &is, Pixel &pixel) {
    pixel.r = is.get();
    pixel.g = is.get();
    pixel.b = is.get();
    return is;
}

std::ostream &operator<<(std::ostream &os, const Pixel &pixel) {
    os.put(pixel.r);
    os.put(pixel.g);
    os.put(pixel.b);
    return os;
}
