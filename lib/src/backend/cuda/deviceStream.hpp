// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/stream.hpp"

#ifndef VS_OPENCL
#include "cuda/error.hpp"
#include <cuda_runtime.h>
#endif

namespace VideoStitch {
namespace GPU {

class Stream::DeviceStream {
 public:
  DeviceStream() : cudaStream(NULL) {}

  explicit DeviceStream(cudaStream_t cs) : cudaStream(cs) {}

  operator cudaStream_t() const { return cudaStream; }

  bool operator==(const DeviceStream& other) const { return cudaStream == other.cudaStream; }

  bool operator!=(const DeviceStream& other) const { return !(*this == other); }

 private:
  cudaStream_t cudaStream;
};

}  // namespace GPU
}  // namespace VideoStitch
