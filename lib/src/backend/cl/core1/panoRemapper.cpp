// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "core1/panoRemapper.hpp"

#include "backend/cpp/core/transformTypes.hpp"

#include "gpu/core1/transform.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/allocator.hpp"

#include "../kernel.hpp"
#include "../surface.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/allocator.hpp"

namespace VideoStitch {
namespace Core {

Status reprojectAlphaToCubemap(int /*panoWidth*/, int /*panoHeight*/, int /*faceLength*/, GPU::Surface&,
                               Rect /*equirectBB*/, GPU::Buffer<unsigned char> /*xPosAlpha*/, Rect /*xPosBB*/,
                               GPU::Buffer<unsigned char> /*xNegAlpha*/, Rect /*xNegBB*/,
                               GPU::Buffer<unsigned char> /*yPosAlpha*/, Rect /*yPosBB*/,
                               GPU::Buffer<unsigned char> /*yNegAlpha*/, Rect /*yNegBB*/,
                               GPU::Buffer<unsigned char> /*zPosAlpha*/, Rect /*zPosBB*/,
                               GPU::Buffer<unsigned char> /*zNegAlpha*/, Rect /*zNegBB*/, bool /*equiangular*/,
                               GPU::Stream) {
  assert(false);
  return Status::OK();
}

Status rotateCubemap(const PanoDefinition& /*pano*/, GPU::CubemapSurface& /*cubemapSurface*/,
                     GPU::Buffer<uint32_t> /*xPosPbo*/, GPU::Buffer<uint32_t> /*xNegPbo*/,
                     GPU::Buffer<uint32_t> /*yPosPbo*/, GPU::Buffer<uint32_t> /*yNegPbo*/,
                     GPU::Buffer<uint32_t> /*zPosPbo*/, GPU::Buffer<uint32_t> /*zNegPbo*/,
                     const Matrix33<double>& /*perspective*/, bool /*equiangular*/, GPU::Stream /*stream*/) {
  assert(false);
  return Status::OK();
}

Status reproject(std::string kernelName, GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                 unsigned width, unsigned height, const Matrix33<double>& perspective, GPU::Stream stream) {
  vsfloat3x3 rotation;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rotation.values[i][j] = (float)perspective(i, j);
    }
  }

  auto kernel2D = GPU::Kernel::get(PROGRAM(zoneKernel), kernelName).setup2D(stream, (unsigned)width, (unsigned)height);
  return kernel2D.enqueueWithKernelArgs(pbo, tex.get(), (unsigned)width, (unsigned)height, inScale, outScale, rotation);
}

Status reprojectRectilinear(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                            unsigned width, unsigned height, const Matrix33<double>& perspective, GPU::Stream stream) {
  return reproject("remap_rectilinear", pbo, outScale, tex, inScale, width, height, perspective, stream);
}
Status reprojectEquirectangular(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                                unsigned width, unsigned height, const Matrix33<double>& perspective,
                                GPU::Stream stream) {
  return reproject("remap_equirectangular", pbo, outScale, tex, inScale, width, height, perspective, stream);
}
Status reprojectFullFrameFisheye(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                                 unsigned width, unsigned height, const Matrix33<double>& perspective,
                                 GPU::Stream stream) {
  return reproject("remap_fullframe_fisheye", pbo, outScale, tex, inScale, width, height, perspective, stream);
}
Status reprojectCircularFisheye(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                                unsigned width, unsigned height, const Matrix33<double>& perspective,
                                GPU::Stream stream) {
  return reproject("remap_circular_fisheye", pbo, outScale, tex, inScale, width, height, perspective, stream);
}
Status reprojectStereographic(GPU::Buffer<uint32_t> pbo, float2 outScale, GPU::Surface& tex, float2 inScale,
                              unsigned width, unsigned height, const Matrix33<double>& perspective,
                              GPU::Stream stream) {
  return reproject("remap_stereographic", pbo, outScale, tex, inScale, width, height, perspective, stream);
}

}  // namespace Core
}  // namespace VideoStitch
