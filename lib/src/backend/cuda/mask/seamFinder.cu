// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mask/seamFinder.hpp"

#include "backend/common/imageOps.hpp"
#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "cuda/util.hpp"
#include "core/rect.hpp"
#include "gpu/vectorTypes.hpp"
#include "gpu/memcpy.hpp"
#include "mask/mergerMaskConstant.hpp"

namespace VideoStitch {
namespace MergerMask {

#define SEAM_FINDER_KERNEL_SIZE_X 16
#define SEAM_FINDER_KERNEL_SIZE_Y 16

const __constant__ int border_dir_rows[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
const __constant__ int border_dir_cols[8] = {0, -1, 1, 0, -1, 1, -1, 1};

__device__ bool isInRange(const int2 coord, const int2 size) {
  if (coord.x < 0 || coord.y < 0 || coord.x >= size.x || coord.y >= size.y) {
    return false;
  }
  return true;
}

__device__ bool isValidPixel(const int2 coord, const int2 size, const uint32_t* const buffer) {
  if (!isInRange(coord, size)) {
    return false;
  }
  if (buffer[coord.y * size.x + coord.x] == INVALID_VALUE) {
    return false;
  }
  return true;
}

__device__ bool isBorder(const int wrapWidth, const int directionCount, const int2 coord, const int2 size,
                         const unsigned char id, const unsigned char* const __restrict__ inputsMap) {
  if (!isInRange(coord, size)) {
    return false;
  }
  if ((inputsMap[coord.y * size.x + coord.x] & id) == 0) {
    return false;
  }
  if (coord.y == 0 || coord.y == size.y - 1) {
    return true;
  }
  // For detecting borders, need to use 8-connected neighbors,
  // or else, it will cause a lot of errors detecting joint points of the 2 borders
  for (int i = 0; i < directionCount; i++) {
    const int2 newCoord =
        make_int2((coord.x + border_dir_rows[i] + wrapWidth) % wrapWidth, coord.y + border_dir_cols[i]);
    if (newCoord.x >= 0 && newCoord.x < size.x && newCoord.y >= 0 && newCoord.y < size.y) {
      // If one of the surrounding pixel is invalid, the current pixel can be consider as border
      if ((inputsMap[newCoord.y * size.x + newCoord.x] & id) == 0) {
        return true;
      }
    }
  }
  return false;
}

__global__ void bordersBufferKernel(const int directionCount, const int wrapWidth, const int2 offset, const int2 size,
                                    const unsigned char* const __restrict__ inputsMap,
                                    unsigned char* const __restrict__ bordersBuffer) {
  unsigned x0 = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y0 = blockIdx.y * blockDim.y + threadIdx.y;
  if (x0 < size.x && y0 < size.y) {
    uint32_t index = y0 * size.x + x0;
    bordersBuffer[index] = 0;

    const int x = x0 + offset.x;
    const int y = y0 + offset.y;

    if (isBorder(wrapWidth, directionCount, make_int2(x, y), size, 1 << 0, inputsMap)) {
      bordersBuffer[index] |= (1 << 0);
    }
    if (isBorder(wrapWidth, directionCount, make_int2(x, y), size, 1 << 1, inputsMap)) {
      bordersBuffer[index] |= (1 << 1);
    }
  }
}

Status SeamFinder::findBordersBuffer(const Core::Rect rect, const GPU::Buffer<const unsigned char> mapBuffer,
                                     GPU::Buffer<unsigned char> bordersBuffer, const int directionCount) {
  dim3 dimBlock(SEAM_FINDER_KERNEL_SIZE_X, SEAM_FINDER_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv((int)rect.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv((int)rect.getHeight(), dimBlock.y), 1);
  bordersBufferKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      directionCount, wrapWidth, make_int2((int)rect.left(), (int)rect.top()),
      make_int2((int)rect.getWidth(), (int)rect.getHeight()), mapBuffer.get(), bordersBuffer.get());
  return CUDA_STATUS;
}

__global__ void validMaskKernel(const int2 size, const uint32_t* const __restrict__ inputBuffer,
                                unsigned char* const __restrict__ maskBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < size.x && y < size.y) {
    uint32_t index = y * size.x + x;
    if (isValidPixel(make_int2(x, y), size, inputBuffer)) {
      maskBuffer[index] = 1;
    } else {
      maskBuffer[index] = 0;
    }
  }
}

Status SeamFinder::findValidMask(const Core::Rect rect, const GPU::Buffer<const uint32_t> inputBuffer,
                                 std::vector<unsigned char>& mask, GPU::Stream stream) {
  dim3 dimBlock(SEAM_FINDER_KERNEL_SIZE_X, SEAM_FINDER_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(rect.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv(rect.getHeight(), dimBlock.y), SEAM_DIRECTION);
  validMaskKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(make_int2((int)rect.getWidth(), (int)rect.getHeight()),
                                                          inputBuffer.get(), workMask.borrow().get());
  mask.resize(inputBuffer.numElements());
  if (mask.size() > 0) {
    return GPU::memcpyBlocking(mask.data(), workMask.borrow_const(), workMask.borrow_const().byteSize());
  } else {
    return CUDA_STATUS;
  }
}

// The cost function is computed as describe in
// Summa et al., Panorama Weaving: Fast and Flexible Seam Processing, Siggraph 2012
// Fig. 4b
__global__ void costsBufferKernel(const int kernelSize, const int warpWidth, const int2 offset0, const int2 size0,
                                  const uint32_t* const __restrict__ input0Buffer, const int2 offset1, const int2 size1,
                                  const uint32_t* const __restrict__ input1Buffer, const int2 offset, const int2 size,
                                  float* const __restrict__ costsBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < size.x && y < size.y && z < SEAM_DIRECTION) {
    uint32_t index = SEAM_DIRECTION * (y * size.x + x) + z;
    costsBuffer[index] = MAX_COST;
    int count = 0;
    const int perpendicular_dir = perpendicular_dirs[z];
    for (int i = -(kernelSize - 1); i <= kernelSize; i++) {
      const int xi = x + seam_dir_advance[z] * seam_dir_rows[z] + i * seam_dir_rows[perpendicular_dir];
      const int yi = y + seam_dir_advance[z] * seam_dir_cols[z] + i * seam_dir_cols[perpendicular_dir];
      if (xi >= 0 && xi < size.x && yi >= 0 && yi < size.y) {
        const int x0 = (offset.x + xi - offset0.x + warpWidth) % warpWidth;
        const int y0 = offset.y + yi - offset0.y;
        const int x1 = (offset.x + xi - offset1.x + warpWidth) % warpWidth;
        const int y1 = offset.y + yi - offset1.y;
        if (isValidPixel(make_int2(x0, y0), size0, input0Buffer) &&
            isValidPixel(make_int2(x1, y1), size1, input1Buffer)) {
          const uint32_t color0 = input0Buffer[y0 * size0.x + x0];
          const uint32_t color1 = input1Buffer[y1 * size1.x + x1];
          const float lWeight = 1.0f;
          const float aWeight = 1.0f;
          const float bWeight = 1.0f;
          const float labDifference =
              (lWeight * abs((float(Image::RGBA::r(color0)) - float(Image::RGBA::r(color1))) / 255.0f) +
               aWeight * abs((float(Image::RGBA::g(color0)) - float(Image::RGBA::g(color1))) / 255.0f) +
               bWeight * abs((float(Image::RGBA::b(color0)) - float(Image::RGBA::b(color1))) / 255.0f)) /
              (lWeight + aWeight + bWeight);
          count++;
          const float gradientDifference = abs((float(Image::RGBA::a(color0)) - Image::RGBA::a(color1)) / 255.0f);
          float gradientWeight = 0.6f;
          const float sad = (labDifference + gradientWeight * gradientDifference) / (1.0f + gradientWeight);
          if (costsBuffer[index] == MAX_COST) {
            costsBuffer[index] = 0.0f;
          }
          costsBuffer[index] += sad;
        }
      }
    }
    if (count) {
      if (count < kernelSize * 2) {
        costsBuffer[index] += (kernelSize * 2 - count) * PENALTY_COST;
        count += kernelSize * 2 - count;
      }
      costsBuffer[index] /= count;
    }
    costsBuffer[index] = costsBuffer[index] + MIN_PENALTY_COST;

    if (x <= 1 || y <= 1 || x >= size.x - 2 || y >= size.y - 2) {
      costsBuffer[index] += MAX_COST;
    }
  }
}

Status SeamFinder::prepareSeamCostBuffer(const Core::Rect rect, GPU::Buffer<float> costsBuffer) {
  dim3 dimBlock(SEAM_FINDER_KERNEL_SIZE_X, SEAM_FINDER_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv((int)rect.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv((int)rect.getHeight(), dimBlock.y), SEAM_DIRECTION);
  costsBufferKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      kernelSize, wrapWidth, make_int2((int)rect0.left(), (int)rect0.top()),
      make_int2((int)rect0.getWidth(), (int)rect0.getHeight()), input0Buffer.get(),
      make_int2((int)rect1.left(), (int)rect1.top()), make_int2((int)rect1.getWidth(), (int)rect1.getHeight()),
      input1Buffer.get(), make_int2((int)rect.left(), (int)rect.top()),
      make_int2((int)rect.getWidth(), (int)rect.getHeight()), costsBuffer.get());
  return CUDA_STATUS;
}

__global__ void inputsMapKernel(const int warpWidth, const videoreaderid_t id0, const int2 offset0, const int2 size0,
                                const uint32_t* const __restrict__ input0Buffer, const videoreaderid_t id1,
                                const int2 offset1, const int2 size1, const uint32_t* const __restrict__ input1Buffer,
                                const int2 offset, const int2 size, unsigned char* const __restrict__ inputsMap) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    const int index = y * size.x + x;
    inputsMap[index] = 0;
    const int x0 = (offset.x + x - offset0.x + warpWidth) % warpWidth;
    const int y0 = offset.y + y - offset0.y;
    if (isValidPixel(make_int2(x0, y0), size0, input0Buffer)) {
      inputsMap[index] += 1 << id0;
    }
    const int x1 = (offset.x + x - offset1.x + warpWidth) % warpWidth;
    const int y1 = offset.y + y - offset1.y;
    if (isValidPixel(make_int2(x1, y1), size1, input1Buffer)) {
      inputsMap[index] += 1 << id1;
    }
  }
}

Status SeamFinder::findInputsMap() {
  dim3 dimBlock(SEAM_FINDER_KERNEL_SIZE_X, SEAM_FINDER_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv((int)rect.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv((int)rect.getHeight(), dimBlock.y), SEAM_DIRECTION);
  inputsMapKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      wrapWidth, id0, make_int2((int)rect0.left(), (int)rect0.top()),
      make_int2((int)rect0.getWidth(), (int)rect0.getHeight()), input0Buffer.get(), id1,
      make_int2((int)rect1.left(), (int)rect1.top()), make_int2((int)rect1.getWidth(), (int)rect1.getHeight()),
      input1Buffer.get(), make_int2((int)rect.left(), (int)rect.top()),
      make_int2((int)rect.getWidth(), (int)rect.getHeight()), inputsMapBuffer.borrow().get());
  return CUDA_STATUS;
}

__global__ void blendImagesKernel(const int warpWidth, const videoreaderid_t id0, const int2 offset0, const int2 size0,
                                  const uint32_t* const __restrict__ input0Buffer, const videoreaderid_t id1,
                                  const int2 offset1, const int2 size1, const uint32_t* const __restrict__ input1Buffer,
                                  const int2 offset, const int2 size, const unsigned char* const __restrict__ inputsMap,
                                  uint32_t* __restrict__ outputBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    const int index = y * size.x + x;
    outputBuffer[index] = 0;
    const uint32_t pixelMap = inputsMap[index];

    const int x1 = (offset.x + x - offset1.x + warpWidth) % warpWidth;
    const int y1 = offset.y + y - offset1.y;
    if (x1 >= 0 && x1 < size1.x && y1 >= 0 && y1 < size1.y && ((pixelMap & (1 << id1)) == (1 << id1))) {
      outputBuffer[index] = input1Buffer[y1 * size1.x + x1];
    }

    const int x0 = (offset.x + x - offset0.x + warpWidth) % warpWidth;
    const int y0 = offset.y + y - offset0.y;
    if (x0 >= 0 && x0 < size0.x && y0 >= 0 && y0 < size0.y && ((pixelMap & (1 << id0)) == (1 << id0))) {
      outputBuffer[index] = input0Buffer[y0 * size0.x + x0];
    }
  }
}

Status SeamFinder::blendImages(GPU::Buffer<uint32_t> outputBuffer) {
  dim3 dimBlock(SEAM_FINDER_KERNEL_SIZE_X, SEAM_FINDER_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv((int)rect.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv((int)rect.getHeight(), dimBlock.y), SEAM_DIRECTION);
  blendImagesKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      wrapWidth, id0, make_int2((int)rect0.left(), (int)rect0.top()),
      make_int2((int)rect0.getWidth(), (int)rect0.getHeight()), input0Buffer.get(), id1,
      make_int2((int)rect1.left(), (int)rect1.top()), make_int2((int)rect1.getWidth(), (int)rect1.getHeight()),
      input1Buffer.get(), make_int2((int)rect.left(), (int)rect.top()),
      make_int2((int)rect.getWidth(), (int)rect.getHeight()), outputsMapBuffer.borrow_const().get(),
      outputBuffer.get());
  return CUDA_STATUS;
}

__global__ void findFeatheringMaskKernel(const int2 size, const uint32_t id,
                                         const unsigned char* const __restrict__ inputBuffer,
                                         uint32_t* __restrict__ outputBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    const int index = y * size.x + x;
    if (inputBuffer[index] == id) {
      outputBuffer[index] = 0;
    } else {
      outputBuffer[index] = 1;
    }
  }
}

Status SeamFinder::findFeatheringMask(const int2 size, const GPU::Buffer<const unsigned char> inputBuffer,
                                      GPU::Buffer<uint32_t> outputBuffer, GPU::Stream stream) {
  dim3 dimBlock(SEAM_FINDER_KERNEL_SIZE_X, SEAM_FINDER_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv((int)size.y, dimBlock.y), 1);
  const uint32_t id = (1 << id0) + (1 << id1);
  findFeatheringMaskKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(size, id, inputBuffer.get(), outputBuffer.get());
  return CUDA_STATUS;
}

}  // namespace MergerMask
}  // namespace VideoStitch
