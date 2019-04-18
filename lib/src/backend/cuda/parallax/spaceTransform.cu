// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "cuda/util.hpp"

#include "backend/cuda/parallax/kernels/mapInverseFunction.cu"
#include "backend/cuda/parallax/kernels/warpSpaceKernel.cu"

namespace VideoStitch {
namespace Core {
namespace {
#define MAP_KERNEL_BLOCK_SIZE_X 16
#define MAP_KERNEL_BLOCK_SIZE_Y 8

#define MAPCOORDINPUTTOOUTPUT_DEF3(INPUTTOSPHERE, INVERSEDISTORTIONMETERS, INVERSEDISTORTIONPIXELS, ISWITHIN)            \
  Status mapCoordInputToOutput_##INPUTTOSPHERE##_##INVERSEDISTORTIONMETERS##_##INVERSEDISTORTIONPIXELS##_##ISWITHIN(     \
      const int time, GPU::Buffer<float2> outputBuffer, const int inputWidth, const int inputHeight,                     \
      const GPU::Buffer<const float2> inputBuffer, const GPU::Buffer<const uint32_t> inputMask,                          \
      const PanoDefinition& pano, const int id, GPU::Stream gpuStream) const {                                           \
    const InputDefinition& im = pano.getInput(id);                                                                       \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                           \
    TransformGeoParams params(im, geometry, pano);                                                                       \
    float2 center, iscale, pscale;                                                                                       \
    center.x = (float)im.getCenterX(geometry);                                                                           \
    center.y = (float)im.getCenterY(geometry);                                                                           \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                     \
    iscale.y = (float)geometry.getVerticalFocal();                                                                       \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);            \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);       \
                                                                                                                         \
    /*NOTE: here we assume that height and width are multiples of dimBlock.x and dimblock.y*/                            \
    dim3 dimBlock(MAP_KERNEL_BLOCK_SIZE_X, MAP_KERNEL_BLOCK_SIZE_Y, 1);                                                  \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(inputWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(inputHeight, dimBlock.y),      \
                 1);                                                                                                     \
    warpCoordInputToOutputKernel_##INPUTTOSPHERE##_##ISWITHIN##_##INVERSEDISTORTIONMETERS##_##INVERSEDISTORTIONPIXELS<<< \
        dimGrid, dimBlock, 0, gpuStream.get()>>>(                                                                        \
        outputBuffer.get(), (int)pano.getWidth(), (int)pano.getHeight(), (unsigned)im.getWidth(),                        \
        (unsigned)im.getHeight(), (int)im.getCropLeft(), (int)im.getCropRight(), (int)im.getCropTop(),                   \
        (int)im.getCropBottom(), (unsigned)inputWidth, (unsigned)inputHeight, id, inputBuffer.get(), inputMask.get(),    \
        pscale, getCombinedPose(params.getPoseInverse()), (float)pano.getSphereScale(), iscale,                          \
        params.getDistortion(), center);                                                                                 \
    return Status::OK();                                                                                                 \
  }

#define MAPCOORDOUTPUTTOINPUT_DEF3(SPHERETOINPUT, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN)                      \
  Status mapCoordOutputToInput_##SPHERETOINPUT##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN(               \
      const int time, const int offsetX, const int offsetY, const int croppedWidth, const int croppedHeight,         \
      GPU::Buffer<float2> outputBuffer, GPU::Buffer<uint32_t> maskBuffer, const PanoDefinition& pano, const int id,  \
      GPU::Stream gpuStream) const {                                                                                 \
    const InputDefinition& im = pano.getInput(id);                                                                   \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                       \
    TransformGeoParams params(im, geometry, pano);                                                                   \
    float2 center, iscale, pscale;                                                                                   \
    center.x = (float)im.getCenterX(geometry);                                                                       \
    center.y = (float)im.getCenterY(geometry);                                                                       \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                 \
    iscale.y = (float)geometry.getVerticalFocal();                                                                   \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);        \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);   \
                                                                                                                     \
    /*NOTE: here we assume that height and width are multiples of dimBlock.x and dimblock.y*/                        \
    dim3 dimBlock(MAP_KERNEL_BLOCK_SIZE_X, MAP_KERNEL_BLOCK_SIZE_Y, 1);                                              \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(croppedWidth, dimBlock.x),                                                  \
                 (unsigned)Cuda::ceilDiv(croppedHeight, dimBlock.y), 1);                                             \
                                                                                                                     \
    warpCoordOutputToInputKernel_##SPHERETOINPUT##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS<<<           \
        dimGrid, dimBlock, 0, gpuStream.get()>>>(                                                                    \
        outputBuffer.get(), maskBuffer.get(), id, (int)offsetX, (int)offsetY, (int)croppedWidth, (int)croppedHeight, \
        (int)pano.getWidth(), (int)pano.getHeight(), (unsigned)im.getWidth(), (unsigned)im.getHeight(),              \
        (int)im.getCropLeft(), (int)im.getCropRight(), (int)im.getCropTop(), (int)im.getCropBottom(), pscale,        \
        getCombinedInversePose(params.getPose()), iscale, params.getDistortion(), center);                           \
    return Status::OK();                                                                                             \
  }

}  // namespace
}  // namespace Core
}  // namespace VideoStitch

#include "backend/common/parallax/spaceTransform.impl"

namespace VideoStitch {
namespace Core {

SpaceTransform* SpaceTransform::create(const InputDefinition& im, const Vector3<double> oldOriCoord,
                                       const Vector3<double> newOriCoord) {
  return createSpaceTransform(im, oldOriCoord, newOriCoord);
}

}  // namespace Core
}  // namespace VideoStitch
