#pragma once

#include <array>

#include "Pixel.hpp"

const int IMAGE_SIZE = 1024;
using Image = std::array<std::array<Pixel, IMAGE_SIZE>, IMAGE_SIZE>;

std::istream& operator>>(std::istream &is, Image &image);
std::ostream& operator<<(std::ostream &os, const Image &image);

bool isRotated(const Image &in, const Image &out);