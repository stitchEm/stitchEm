// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "score/scoringProcessor.hpp"

#include "backend/common/imageOps.hpp"
#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "cuda/util.hpp"

namespace {

#define Image_RGB210_a VideoStitch::Image::RGB210::a
#define Image_RGB210_r VideoStitch::Image::RGB210::r
#define Image_RGB210_g VideoStitch::Image::RGB210::g
#define Image_RGB210_b VideoStitch::Image::RGB210::b

#define Image_RGBA_a VideoStitch::Image::RGBA::a
#define Image_RGBA_r VideoStitch::Image::RGBA::r
#define Image_RGBA_g VideoStitch::Image::RGBA::g
#define Image_RGBA_b VideoStitch::Image::RGBA::b

#include <backend/cuda/gpuKernelDef.h>
#include <backend/common/score/scoringKernel.gpu>

#undef Image_RGB210_a
#undef Image_RGB210_r
#undef Image_RGB210_g
#undef Image_RGB210_b

#undef Image_RGBA_a
#undef Image_RGBA_r
#undef Image_RGBA_g
#undef Image_RGBA_b

}  // namespace

namespace VideoStitch {
namespace Image {

Status splitNoBlendImageMergerChannel(GPU::Buffer<float> dest_r, GPU::Buffer<float> dest_g,
                                      GPU::Buffer<unsigned char> dest_b, GPU::Buffer<const uint32_t> source,
                                      const unsigned width, const unsigned height, GPU::Stream stream) {
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, dimBlock.x), (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);

  splitNoBlendImageMergerChannelsKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      dest_r.get(), dest_g.get(), dest_b.get(), source.get(), width, height);
  return CUDA_STATUS;
}

}  // namespace Image
}  // namespace VideoStitch
