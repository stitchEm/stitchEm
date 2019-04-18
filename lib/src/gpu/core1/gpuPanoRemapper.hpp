// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core1/panoRemapper.hpp"

namespace VideoStitch {
namespace Core {

class GPUPanoRemapper : public PanoRemapper {
  /**
   * Factory
   */
  static PanoRemapper* create(const PanoDefinition&);

  /**
   * Run asynchronously in the given stream.
   */
  void runAsync(const PanoDefinition&, int time, uint32_t* devBuffer, const Matrix33<double>& interactivePersp,
                GPU::Stream stream);

 private:
  GPUImpl* pimpl;
};

}  // namespace Core
}  // namespace VideoStitch
