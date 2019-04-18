// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "util.hpp"

namespace VideoStitch {
namespace Cuda {

#define MAXGRIDDIM 65535
dim3 compute2DGridForFlatBuffer(int64_t size, unsigned blockSize) {
  dim3 grid((unsigned)ceilDiv(size, blockSize), 1);
  while (grid.x > MAXGRIDDIM) {
    grid.x = (grid.x + 1) / 2;
    grid.y *= 2;
  }
  return grid;
}

}  // namespace Cuda
}  // namespace VideoStitch
