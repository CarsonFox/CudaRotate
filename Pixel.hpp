#pragma once

#include <iostream>

struct Pixel {
    char r, g, b;
};

std::istream& operator>>(std::istream &is, Pixel &pixel);
std::ostream& operator<<(std::ostream &os, const Pixel &pixel);

bool operator==(const Pixel &lhs, const Pixel &rhs);
bool operator!=(const Pixel &lhs, const Pixel &rhs);
