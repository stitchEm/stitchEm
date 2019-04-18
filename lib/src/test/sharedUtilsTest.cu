// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Basic input unpacking tests.

#include "testing.hpp"
#include "util.hpp"
#include "../image/kernels/sharedUtils.hpp"

#include <memory>

namespace VideoStitch {
namespace Testing {

template <typename T, unsigned left, unsigned right, unsigned top, unsigned bottom, typename Getter>
__global__ void dumpSharedKernel(T* __restrict__ sharedDst, const unsigned sharedWidth, const unsigned sharedHeight,
                                 const T* __restrict__ src, unsigned srcWidth, unsigned srcHeight, unsigned srcOffsetX,
                                 unsigned srcOffsetY) {
  Image::loadToSharedMemory<T, left, right, top, bottom, Getter>(sharedDst, sharedWidth, sharedHeight, src, srcWidth,
                                                                 srcHeight, srcOffsetX, srcOffsetY);
  __syncthreads();
}

template <typename T, unsigned left, unsigned right, unsigned top, unsigned bottom>
void groundTruthExtend(std::vector<T>& sharedDst, const int sharedWidth, const int sharedHeight,
                       const std::vector<T>& src, int srcWidth, int srcHeight, int srcOffsetX, int srcOffsetY) {
  const int realSharedWidth = sharedWidth + left + right;
  const int realSharedHeight = sharedHeight + top + bottom;
  sharedDst.clear();
  // Top rows:
  for (int sharedY = 0; sharedY < realSharedHeight; ++sharedY) {
    int srcY = srcOffsetY + sharedY - top;
    if (srcY < 0) {
      srcY = 0;
    }
    if (srcY >= srcHeight) {
      srcY = srcHeight - 1;
    }
    for (int sharedX = 0; sharedX < realSharedWidth; ++sharedX) {
      int srcX = srcOffsetX + sharedX - left;
      if (srcX < 0) {
        srcX = 0;
      }
      if (srcX >= srcWidth) {
        srcX = srcWidth - 1;
      }
      sharedDst.push_back(src[srcWidth * srcY + srcX]);
    }
  }
}

template <typename T, unsigned left, unsigned right, unsigned top, unsigned bottom>
void groundTruthZero(std::vector<T>& sharedDst, const int sharedWidth, const int sharedHeight,
                     const std::vector<T>& src, int srcWidth, int srcHeight, int srcOffsetX, int srcOffsetY) {
  const int realSharedWidth = sharedWidth + left + right;
  const int realSharedHeight = sharedHeight + top + bottom;
  sharedDst.clear();
  // Top rows:
  for (int sharedY = 0; sharedY < realSharedHeight; ++sharedY) {
    const int srcY = srcOffsetY + sharedY - top;
    for (int sharedX = 0; sharedX < realSharedWidth; ++sharedX) {
      const int srcX = srcOffsetX + sharedX - left;
      if (srcY < 0 || srcY >= srcHeight || srcX < 0 || srcX >= srcWidth) {
        sharedDst.push_back(0);
      } else {
        sharedDst.push_back(src[srcWidth * srcY + srcX]);
      }
    }
  }
}

template <typename T, unsigned left, unsigned right, unsigned top, unsigned bottom, int blockWidth, int blockHeight>
void runTest(const int width, const int height, const std::vector<T>& input, const int sharedWidth,
             const int sharedHeight, const int srcOffsetX, const int srcOffsetY, bool extend) {
  // std::cout << "sharedWidth=" << sharedWidth << " sharedHeight=" << sharedHeight << " srcOffsetX=" << srcOffsetX << "
  // srcOffsetY=" << srcOffsetY << std::endl;

  std::vector<T> output;
  {
    DeviceBuffer<T> inputBuffer(width, height);
    inputBuffer.fill(input);
    const dim3 dimBlock2D(blockWidth, blockHeight, 1);
    const dim3 dimGrid2D(1, 1, 1);
    DeviceBuffer<T> outputBuffer(sharedWidth + left + right, sharedHeight + top + bottom);
    outputBuffer.fill((T)99);
    if (extend) {
      dumpSharedKernel<T, left, right, top, bottom, Image::ExtendBoundary<T>><<<dimGrid2D, dimBlock2D, 0, 0>>>(
          outputBuffer.ptr(), sharedWidth, sharedHeight, inputBuffer.ptr(), width, height, srcOffsetX, srcOffsetY);
    } else {
      dumpSharedKernel<T, left, right, top, bottom, Image::ZeroBoundary<T>><<<dimGrid2D, dimBlock2D, 0, 0>>>(
          outputBuffer.ptr(), sharedWidth, sharedHeight, inputBuffer.ptr(), width, height, srcOffsetX, srcOffsetY);
    }

    outputBuffer.readback(output);
  }

  std::vector<T> groundTruthOutput;
  if (extend) {
    groundTruthExtend<T, left, right, top, bottom>(groundTruthOutput, sharedWidth, sharedHeight, input, width, height,
                                                   srcOffsetX, srcOffsetY);
  } else {
    groundTruthZero<T, left, right, top, bottom>(groundTruthOutput, sharedWidth, sharedHeight, input, width, height,
                                                 srcOffsetX, srcOffsetY);
  }

  /*for (int y = 0; y < sharedHeight + top + bottom; ++y) {
    for (int x = 0; x < sharedWidth + left + right; ++x) {
      std::cout << groundTruthOutput[(sharedWidth + left + right) * y + x]<< " ";
    }
    std::cout << std::endl;
  }

  for (int y = 0; y < sharedHeight + top + bottom; ++y) {
    for (int x = 0; x < sharedWidth + left + right; ++x) {
      std::cout << output[(sharedWidth + left + right) * y + x]<< " ";
    }
    std::cout << std::endl;
  }*/
  ENSURE_2D_ARRAY_EQ(groundTruthOutput.data(), output.data(), sharedWidth + left + right, sharedHeight + top + bottom);
}

template <typename T, unsigned left, unsigned right, unsigned top, unsigned bottom>
void testSharedUtils(const int width, const int height, bool extend) {
  std::vector<T> input;
  for (int i = 0; i < width * height; ++i) {
    input.push_back((T)i);
  }

  for (int sharedWidth = 1; sharedWidth < width; ++sharedWidth) {
    for (int sharedHeight = 1; sharedHeight < height; ++sharedHeight) {
      for (int srcOffsetX = 0; srcOffsetX < width; ++srcOffsetX) {
        for (int srcOffsetY = 0; srcOffsetY < height; ++srcOffsetY) {
          runTest<T, left, right, top, bottom, 4, 4>(width, height, input, sharedWidth, sharedHeight, srcOffsetX,
                                                     srcOffsetY, extend);
        }
      }
    }
  }
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  cudaSetDevice(0);
  VideoStitch::Testing::testSharedUtils<int, 1u, 1u, 1u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<int, 2u, 1u, 1u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<int, 1u, 2u, 1u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<int, 1u, 1u, 2u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<int, 1u, 1u, 1u, 2u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<char, 1u, 1u, 1u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<char, 2u, 1u, 1u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<char, 1u, 2u, 1u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<char, 1u, 1u, 2u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<char, 1u, 1u, 1u, 2u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<uint16_t, 1u, 1u, 1u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<uint16_t, 2u, 1u, 1u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<uint16_t, 1u, 2u, 1u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<uint16_t, 1u, 1u, 2u, 1u>(6, 5, true);
  VideoStitch::Testing::testSharedUtils<uint16_t, 1u, 1u, 1u, 2u>(6, 5, true);

  VideoStitch::Testing::testSharedUtils<int, 1u, 1u, 1u, 1u>(6, 5, false);

  {
    const int width = 512;
    const int height = 512;
    std::vector<float> input;
    for (int i = 0; i < width * height; ++i) {
      input.push_back((float)i);
    }
    VideoStitch::Testing::runTest<float, 1u, 1u, 1u, 1u, 16, 16>(width, height, input, 16, 16, 110, 11, true);
  }
  cudaDeviceReset();
  return 0;
}
