// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/image/downsampler.hpp"

#include "gpu/2dBuffer.hpp"

namespace VideoStitch {
namespace Image {

Status downsample(PixelFormat fmt, GPU::Buffer2D* planesIn, GPU::Buffer2D* planesOut, GPU::Stream stream) {
  int factor;

  assert(planesIn[0].getWidth() / planesOut[0].getWidth() == planesIn[0].getHeight() / planesOut[0].getHeight());
  factor = (int)(planesIn[0].getWidth() / planesOut[0].getWidth());

  if (factor <= 1) {
    return {Origin::Stitcher, ErrType::ImplementationError, "Unsupported downsampling factor"};
  }
  if (!(planesIn[0].getWidth() % factor == 0 && planesIn[0].getHeight() % factor == 0)) {
    return {Origin::Stitcher, ErrType::ImplementationError, "Invalid downsampling factor"};
  }
  switch (fmt) {
    case RGBA:
      return downsampleRGBA(factor, planesIn[0], planesOut[0], stream);
    case RGB:
      return downsampleRGB(factor, planesIn[0], planesOut[0], stream);
    case YV12:
      return downsampleYV12(factor, planesIn[0], planesIn[1], planesIn[2], planesOut[0], planesOut[1], planesOut[2],
                            stream);
    case NV12:
      return downsampleNV12(factor, planesIn[0], planesIn[1], planesOut[0], planesOut[1], stream);
    case UYVY:
    case YUY2:
      return downsampleYUV422(factor, planesIn[0], planesOut[0], stream);
    case YUV422P10:
      return downsampleYUV422P10(factor, planesIn[0], planesIn[1], planesIn[2], planesOut[0], planesOut[1],
                                 planesOut[2], stream);
    default:
      assert(false);
      return {Origin::Stitcher, ErrType::ImplementationError, "Unsupported colorspace for downsampling"};
  }
}

}  // namespace Image
}  // namespace VideoStitch
