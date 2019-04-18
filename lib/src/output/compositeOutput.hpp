// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <stdint.h>

namespace VideoStitch {
namespace Output {

template <int pixelSize>
void writeHalfBufferHorizontalInter(char* dst, const char* src, const int64_t height, const int64_t dstWidth,
                                    const int64_t srcWidth, const int64_t xOffset);

}
}  // namespace VideoStitch
