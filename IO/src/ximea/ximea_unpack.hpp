// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <cuda_runtime.h>
#include <stdint.h>

void unpackMono12p(unsigned char* dst, const unsigned char* src, int64_t width, int64_t height, cudaStream_t s);
