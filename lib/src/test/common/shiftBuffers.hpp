// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef TESTING_SHIFT_BUFFERS_HPP_
#define TESTING_SHIFT_BUFFERS_HPP_
namespace VideoStitch {
namespace Testing {
namespace {

inline void shiftHostRight(uint32_t* data, int64_t width, int64_t height, int64_t s) {
  uint32_t* shifted = new uint32_t[size_t(width * height)];
  for (int64_t y = 0; y < height; ++y) {
    for (int64_t x = 0; x < width; ++x) {
      shifted[y * width + ((x + s) % width)] = data[y * width + x];
    }
  }
  for (int64_t i = 0; i < width * height; ++i) {
    data[i] = shifted[i];
  }
  delete[] shifted;
}

inline void shiftHostLeft(uint32_t* data, int64_t width, int64_t height, int64_t s) {
  uint32_t* shifted = new uint32_t[size_t(width * height)];
  for (int64_t y = 0; y < height; ++y) {
    for (int64_t x = 0; x < width; ++x) {
      shifted[y * width + x] = data[y * width + ((x + s) % width)];
    }
  }
  for (int64_t i = 0; i < width * height; ++i) {
    data[i] = shifted[i];
  }
  delete[] shifted;
}

inline void shiftDevLeft(uint32_t* devData, int64_t width, int64_t height, int64_t s) {
  uint32_t* data = new uint32_t[size_t(width * height)];
  ENSURE(cudaSuccess == cudaMemcpy(data, devData, size_t(width * height * 4), cudaMemcpyDeviceToHost));
  shiftHostLeft(data, width, height, s);
  ENSURE(cudaSuccess == cudaMemcpy(devData, data, size_t(width * height * 4), cudaMemcpyHostToDevice));
  delete[] data;
}

}  // namespace
}  // namespace Testing
}  // namespace VideoStitch

#endif
