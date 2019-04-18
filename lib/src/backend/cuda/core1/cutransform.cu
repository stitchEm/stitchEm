// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "backend/common/imageOps.hpp"
#include "backend/common/vectorOps.hpp"

#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/surface.hpp"
#include "backend/cuda/deviceStream.hpp"

#include "cuda/util.hpp"

#include "backend/cuda/parallax/kernels/mapInverseFunction.cu"
#include "backend/cuda/core1/kernels/photoCorrectionFunction.cu"
#include "backend/cuda/core1/kernels/warpKernel.cu"
#include "backend/cuda/core1/kernels/distortionKernel.cu"
#include "backend/cuda/core1/kernels/undistortKernel.cu"
#include "backend/cuda/core1/kernels/zoneKernel.cu"
#include "backend/cuda/parallax/spaceTransform.cu"
#include "gpu/image/imgInsert.hpp"

namespace VideoStitch {
namespace Core {
namespace {

#define MAPBUFFER_PANO3(MERGER, INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN, PHOTORESPONSE)           \
  Status                                                                                                                \
      mapBuffer_##MERGER##_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##PHOTORESPONSE(  \
          int time, GPU::Buffer<uint32_t> devOut, GPU::Surface& panoSurf, const unsigned char* mask,                    \
          const Rect& outputBounds, const PanoDefinition& pano, const InputDefinition& im, GPU::Surface& surface,       \
          GPU::Stream gpuStream) const {                                                                                \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                          \
    TransformGeoParams params(im, geometry, pano);                                                                      \
    const float3 colorMult =                                                                                            \
        PhotoTransform::ColorCorrectionParams(im.getExposureValue().at(time), im.getRedCB().at(time),                   \
                                              im.getGreenCB().at(time), im.getBlueCB().at(time))                        \
            .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),                         \
                                    pano.getGreenCB().at(time), pano.getBlueCB().at(time));                             \
    const unsigned height =                                                                                             \
        (mergeformat == ImageMerger::Format::Gradient)                                                                  \
            ? min((unsigned)outputBounds.getHeight(), (unsigned)(pano.getHeight() - outputBounds.top()))                \
            : (unsigned)outputBounds.getHeight();                                                                       \
                                                                                                                        \
    /*NOTE: here we assume that height and width are multiples of dimBlock.x and dimblock.y*/                           \
    dim3 dimBlock(MAP_KERNEL_BLOCK_SIZE_X_##PHOTORESPONSE, MAP_KERNEL_BLOCK_SIZE_Y_##PHOTORESPONSE, 1);                 \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(outputBounds.getWidth(), dimBlock.x),                                          \
                 (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);                                                       \
                                                                                                                        \
    float2 center, iscale, pscale;                                                                                      \
    center.x = (float)im.getCenterX(geometry);                                                                          \
    center.y = (float)im.getCenterY(geometry);                                                                          \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                    \
    iscale.y = (float)geometry.getVerticalFocal();                                                                      \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);           \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);      \
                                                                                                                        \
    warpKernel_##MERGER##_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE<<< \
        dimGrid, dimBlock, 0, gpuStream.get()>>>(                                                                       \
        devOut.get(), panoSurf.get().surface(), mask, surface.get().texture(), (unsigned)im.getWidth(),                 \
        (unsigned)im.getHeight(), (unsigned)outputBounds.getWidth(), height, (unsigned)outputBounds.left(),             \
        (unsigned)outputBounds.top(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(),                            \
        (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(), (unsigned)im.getCropTop(),                             \
        (unsigned)im.getCropBottom(), pscale, params.getPose(), iscale, params.getDistortion(), center, colorMult,      \
        photo->getDevicePhotoParam().floatParam, (const float*)photo->getDevicePhotoParam().transformData,              \
        (float)im.getVignettingCenterX(), (float)im.getVignettingCenterY(),                                             \
        (float)TransformGeoParams::getInverseDemiDiagonalSquared(im), (float)im.getVignettingCoeff0(),                  \
        (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(), (float)im.getVignettingCoeff3());             \
                                                                                                                        \
    return Status::OK();                                                                                                \
  }

#define WARP_CUBEMAP3(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN, PHOTORESPONSE, EQUIANGULAR)                    \
  Status                                                                                                                            \
      warpCubemap_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##PHOTORESPONSE##_##EQUIANGULAR(       \
          int time, GPU::Buffer<uint32_t> xPos, const Rect& xPosBB, GPU::Buffer<uint32_t> xNeg, const Rect& xNegBB,                 \
          GPU::Buffer<uint32_t> yPos, const Rect& yPosBB, GPU::Buffer<uint32_t> yNeg, const Rect& yNegBB,                           \
          GPU::Buffer<uint32_t> zPos, const Rect& zPosBB, GPU::Buffer<uint32_t> zNeg, const Rect& zNegBB,                           \
          const PanoDefinition& pano, const InputDefinition& im, GPU::Surface& surface, GPU::Stream stream) const {                 \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                                      \
    TransformGeoParams params(im, geometry, pano);                                                                                  \
    const float3 colorMult =                                                                                                        \
        PhotoTransform::ColorCorrectionParams(im.getExposureValue().at(time), im.getRedCB().at(time),                               \
                                              im.getGreenCB().at(time), im.getBlueCB().at(time))                                    \
            .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),                                     \
                                    pano.getGreenCB().at(time), pano.getBlueCB().at(time));                                         \
                                                                                                                                    \
    float2 center, iscale;                                                                                                          \
    center.x = (float)im.getCenterX(geometry);                                                                                      \
    center.y = (float)im.getCenterY(geometry);                                                                                      \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                                \
    iscale.y = (float)geometry.getVerticalFocal();                                                                                  \
                                                                                                                                    \
    dim3 dimBlock(ZONE_KERNEL_BLOCK_SIZE_X, ZONE_KERNEL_BLOCK_SIZE_Y, 1);                                                           \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(pano.getLength(), dimBlock.x),                                                             \
                 (unsigned)Cuda::ceilDiv(pano.getLength(), dimBlock.y), 1);                                                         \
                                                                                                                                    \
    if (!xPosBB.empty()) {                                                                                                          \
      warpKernel_XPos_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR<<< \
          dimGrid, dimBlock, 0, stream.get()>>>(                                                                                    \
          xPos.get(), surface.get().texture(), (unsigned)im.getWidth(), (unsigned)im.getHeight(),                                   \
          (unsigned)xPosBB.getWidth(), (unsigned)xPosBB.getHeight(), (unsigned)xPosBB.left(), (unsigned)xPosBB.top(),               \
          (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                                      \
          (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),                \
          center, colorMult, photo->getDevicePhotoParam().floatParam,                                                               \
          (const float*)photo->getDevicePhotoParam().transformData, (float)im.getVignettingCenterX(),                               \
          (float)im.getVignettingCenterY(), (float)TransformGeoParams::getInverseDemiDiagonalSquared(im),                           \
          (float)im.getVignettingCoeff0(), (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(),                        \
          (float)im.getVignettingCoeff3());                                                                                         \
    }                                                                                                                               \
    if (!xNegBB.empty()) {                                                                                                          \
      warpKernel_XNeg_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR<<< \
          dimGrid, dimBlock, 0, stream.get()>>>(                                                                                    \
          xNeg.get(), surface.get().texture(), (unsigned)im.getWidth(), (unsigned)im.getHeight(),                                   \
          (unsigned)xNegBB.getWidth(), (unsigned)xNegBB.getHeight(), (unsigned)xNegBB.left(), (unsigned)xNegBB.top(),               \
          (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                                      \
          (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),                \
          center, colorMult, photo->getDevicePhotoParam().floatParam,                                                               \
          (const float*)photo->getDevicePhotoParam().transformData, (float)im.getVignettingCenterX(),                               \
          (float)im.getVignettingCenterY(), (float)TransformGeoParams::getInverseDemiDiagonalSquared(im),                           \
          (float)im.getVignettingCoeff0(), (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(),                        \
          (float)im.getVignettingCoeff3());                                                                                         \
    }                                                                                                                               \
    if (!yPosBB.empty()) {                                                                                                          \
      warpKernel_YPos_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR<<< \
          dimGrid, dimBlock, 0, stream.get()>>>(                                                                                    \
          yPos.get(), surface.get().texture(), (unsigned)im.getWidth(), (unsigned)im.getHeight(),                                   \
          (unsigned)yPosBB.getWidth(), (unsigned)yPosBB.getHeight(), (unsigned)yPosBB.left(), (unsigned)yPosBB.top(),               \
          (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                                      \
          (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),                \
          center, colorMult, photo->getDevicePhotoParam().floatParam,                                                               \
          (const float*)photo->getDevicePhotoParam().transformData, (float)im.getVignettingCenterX(),                               \
          (float)im.getVignettingCenterY(), (float)TransformGeoParams::getInverseDemiDiagonalSquared(im),                           \
          (float)im.getVignettingCoeff0(), (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(),                        \
          (float)im.getVignettingCoeff3());                                                                                         \
    }                                                                                                                               \
    if (!yNegBB.empty()) {                                                                                                          \
      warpKernel_YNeg_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR<<< \
          dimGrid, dimBlock, 0, stream.get()>>>(                                                                                    \
          yNeg.get(), surface.get().texture(), (unsigned)im.getWidth(), (unsigned)im.getHeight(),                                   \
          (unsigned)yNegBB.getWidth(), (unsigned)yNegBB.getHeight(), (unsigned)yNegBB.left(), (unsigned)yNegBB.top(),               \
          (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                                      \
          (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),                \
          center, colorMult, photo->getDevicePhotoParam().floatParam,                                                               \
          (const float*)photo->getDevicePhotoParam().transformData, (float)im.getVignettingCenterX(),                               \
          (float)im.getVignettingCenterY(), (float)TransformGeoParams::getInverseDemiDiagonalSquared(im),                           \
          (float)im.getVignettingCoeff0(), (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(),                        \
          (float)im.getVignettingCoeff3());                                                                                         \
    }                                                                                                                               \
    if (!zPosBB.empty()) {                                                                                                          \
      warpKernel_ZPos_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR<<< \
          dimGrid, dimBlock, 0, stream.get()>>>(                                                                                    \
          zPos.get(), surface.get().texture(), (unsigned)im.getWidth(), (unsigned)im.getHeight(),                                   \
          (unsigned)zPosBB.getWidth(), (unsigned)zPosBB.getHeight(), (unsigned)zPosBB.left(), (unsigned)zPosBB.top(),               \
          (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                                      \
          (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),                \
          center, colorMult, photo->getDevicePhotoParam().floatParam,                                                               \
          (const float*)photo->getDevicePhotoParam().transformData, (float)im.getVignettingCenterX(),                               \
          (float)im.getVignettingCenterY(), (float)TransformGeoParams::getInverseDemiDiagonalSquared(im),                           \
          (float)im.getVignettingCoeff0(), (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(),                        \
          (float)im.getVignettingCoeff3());                                                                                         \
    }                                                                                                                               \
    if (!zNegBB.empty()) {                                                                                                          \
      warpKernel_ZNeg_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR<<< \
          dimGrid, dimBlock, 0, stream.get()>>>(                                                                                    \
          zNeg.get(), surface.get().texture(), (unsigned)im.getWidth(), (unsigned)im.getHeight(),                                   \
          (unsigned)zNegBB.getWidth(), (unsigned)zNegBB.getHeight(), (unsigned)zNegBB.left(), (unsigned)zNegBB.top(),               \
          (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                                      \
          (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),                \
          center, colorMult, photo->getDevicePhotoParam().floatParam,                                                               \
          (const float*)photo->getDevicePhotoParam().transformData, (float)im.getVignettingCenterX(),                               \
          (float)im.getVignettingCenterY(), (float)TransformGeoParams::getInverseDemiDiagonalSquared(im),                           \
          (float)im.getVignettingCoeff0(), (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(),                        \
          (float)im.getVignettingCoeff3());                                                                                         \
    }                                                                                                                               \
                                                                                                                                    \
    return Status::OK();                                                                                                            \
  }

#define MAPBUFFERLOOKUP_PANO3(MERGER, ISWITHIN, PHOTORESPONSE)                                                         \
  Status mapBufferLookup_##MERGER##_##ISWITHIN##_##PHOTORESPONSE(                                                      \
      int time, GPU::Buffer<uint32_t> devOut, GPU::Surface& panoSurf, const unsigned char* mask,                       \
      const GPU::Surface& coordIn, const float coordShrinkFactor, const Rect& outputBounds,                            \
      const PanoDefinition& pano, const InputDefinition& im, GPU::Surface& surface, GPU::Stream gpuStream) const {     \
    const float3 colorMult =                                                                                           \
        PhotoTransform::ColorCorrectionParams(im.getExposureValue().at(time), im.getRedCB().at(time),                  \
                                              im.getGreenCB().at(time), im.getBlueCB().at(time))                       \
            .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),                        \
                                    pano.getGreenCB().at(time), pano.getBlueCB().at(time));                            \
    const unsigned height =                                                                                            \
        (mergeformat == ImageMerger::Format::Gradient)                                                                 \
            ? min((unsigned)outputBounds.getHeight(), (unsigned)(pano.getHeight() - outputBounds.top()))               \
            : (unsigned)outputBounds.getHeight();                                                                      \
                                                                                                                       \
    /*NOTE: here we assume that height and width are multiples of dimBlock.x and dimblock.y*/                          \
    dim3 dimBlock(MAP_KERNEL_BLOCK_SIZE_X_##PHOTORESPONSE, MAP_KERNEL_BLOCK_SIZE_Y_##PHOTORESPONSE, 1);                \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(outputBounds.getWidth(), dimBlock.x),                                         \
                 (unsigned)Cuda::ceilDiv(height, dimBlock.y), 1);                                                      \
                                                                                                                       \
    warpLookupKernel_##MERGER##_##ISWITHIN##_##PHOTORESPONSE<<<dimGrid, dimBlock, 0, gpuStream.get()>>>(               \
        devOut.get(), panoSurf.get().surface(), mask, coordIn.get().texture(), coordShrinkFactor,                      \
        surface.get().texture(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)outputBounds.getWidth(), \
        height, (unsigned)outputBounds.left(), (unsigned)outputBounds.top(), (unsigned)pano.getWidth(),                \
        (unsigned)pano.getHeight(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                           \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), colorMult, photo->getDevicePhotoParam().floatParam,   \
        (const float*)photo->getDevicePhotoParam().transformData, (float)im.getVignettingCenterX(),                    \
        (float)im.getVignettingCenterY(), (float)TransformGeoParams::getInverseDemiDiagonalSquared(im),                \
        (float)im.getVignettingCoeff0(), (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(),             \
        (float)im.getVignettingCoeff3());                                                                              \
                                                                                                                       \
    return Status::OK();                                                                                               \
  }

#define MAPBUFFERCOORD_PANO3(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN)                            \
  Status mapBufferCoord_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN(                      \
      int time, GPU::Surface& devOut, const Rect& outputBounds, const PanoDefinition& pano, const InputDefinition& im, \
      GPU::Stream gpuStream) const {                                                                                   \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                         \
    TransformGeoParams params(im, geometry, pano);                                                                     \
                                                                                                                       \
    /*NOTE: here we assume that height and width are multiples of dimBlock.x and dimblock.y*/                          \
    dim3 dimBlock(MAP_KERNEL_BLOCK_SIZE_X, MAP_KERNEL_BLOCK_SIZE_Y, 1);                                                \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(outputBounds.getWidth(), dimBlock.x),                                         \
                 (unsigned)Cuda::ceilDiv(outputBounds.getHeight(), dimBlock.y), 1);                                    \
                                                                                                                       \
    float2 center, iscale, pscale;                                                                                     \
    unsigned panoWidth = (unsigned)(pano.getWidth() / pano.getPrecomputedCoordinateShrinkFactor());                    \
    unsigned panoHeight = (unsigned)(pano.getHeight() / pano.getPrecomputedCoordinateShrinkFactor());                  \
    center.x = (float)im.getCenterX(geometry);                                                                         \
    center.y = (float)im.getCenterY(geometry);                                                                         \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                   \
    iscale.y = (float)geometry.getVerticalFocal();                                                                     \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, panoWidth, 360.f);                \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, panoHeight, 360.f);           \
                                                                                                                       \
    warpCoordKernel_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS<<<dimGrid, dimBlock, 0,   \
                                                                                               gpuStream.get()>>>(     \
        devOut.get().surface(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)im.getCropLeft(),         \
        (unsigned)im.getCropRight(), (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(),                          \
        (unsigned)outputBounds.getWidth(), (unsigned)outputBounds.getHeight(), (unsigned)outputBounds.left(),          \
        (unsigned)outputBounds.top(), panoWidth, panoHeight, (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),  \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), pscale, params.getPose(), iscale,                     \
        params.getDistortion(), center);                                                                               \
                                                                                                                       \
    return Status::OK();                                                                                               \
  }

#define MAPCOORDINPUT_PANO3(INPUTPROJECTION, INVERSEDISTORTIONMETERS, INVERSEDISTORTIONPIXELS, ISWITHIN)              \
  Status mapCoordInput_##INPUTPROJECTION##_##INVERSEDISTORTIONMETERS##_##INVERSEDISTORTIONPIXELS##_##ISWITHIN(        \
      int time, const int scaleFactor, GPU::Buffer<float2> inputCoord, const PanoDefinition& pano,                    \
      const InputDefinition& im, GPU::Stream gpuStream) const {                                                       \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                        \
    TransformGeoParams params(im, geometry, pano);                                                                    \
                                                                                                                      \
    /*NOTE: here we assume that height and width are multiples of dimBlock.x and dimblock.y*/                         \
    dim3 dimBlock(MAP_KERNEL_BLOCK_SIZE_X, MAP_KERNEL_BLOCK_SIZE_Y, 1);                                               \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(scaleFactor* im.getWidth(), dimBlock.x),                                     \
                 (unsigned)Cuda::ceilDiv(scaleFactor* im.getHeight(), dimBlock.y), 1);                                \
                                                                                                                      \
    float2 center, iscale, pscale;                                                                                    \
    center.x = (float)im.getCenterX(geometry);                                                                        \
    center.y = (float)im.getCenterY(geometry);                                                                        \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                  \
    iscale.y = (float)geometry.getVerticalFocal();                                                                    \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);         \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);    \
                                                                                                                      \
    warpCoordInputKernel_##INPUTPROJECTION##_##ISWITHIN##_##INVERSEDISTORTIONMETERS##_##INVERSEDISTORTIONPIXELS<<<    \
        dimGrid, dimBlock, 0, gpuStream.get()>>>(                                                                     \
        inputCoord.get(), scaleFactor, (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)im.getCropLeft(), \
        (unsigned)im.getCropRight(), (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(),                         \
        (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), pscale, params.getPoseInverse(),                       \
        (float)pano.getSphereScale(), iscale, params.getDistortion(), center);                                        \
                                                                                                                      \
    return Status::OK();                                                                                              \
  }

#define COMPUTEZONE_PANO3(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN)                             \
  Status computeZone_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN(                       \
      GPU::Buffer<uint32_t> devOut, const PanoDefinition& pano, const InputDefinition& im, videoreaderid_t imId,     \
      GPU::Buffer<const unsigned char> maskDevBuffer, GPU::Stream stream) const {                                    \
    GeometryDefinition geometry = im.getGeometries().at(0);                                                          \
    TransformGeoParams params(im, geometry, pano);                                                                   \
                                                                                                                     \
    float2 center, iscale, pscale;                                                                                   \
    center.x = (float)im.getCenterX(geometry);                                                                       \
    center.y = (float)im.getCenterY(geometry);                                                                       \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                 \
    iscale.y = (float)geometry.getVerticalFocal();                                                                   \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);        \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);   \
                                                                                                                     \
    dim3 dimBlock(ZONE_KERNEL_BLOCK_SIZE_X, ZONE_KERNEL_BLOCK_SIZE_Y, 1);                                            \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(pano.getWidth(), dimBlock.x),                                               \
                 (unsigned)Cuda::ceilDiv(pano.getHeight(), dimBlock.y), 1);                                          \
                                                                                                                     \
    zoneKernel_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS<<<dimGrid, dimBlock, 0,      \
                                                                                          stream.get()>>>(           \
        devOut.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(), \
        (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), (unsigned)im.getCropLeft(),                           \
        (unsigned)im.getCropRight(), (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), pscale,                \
        params.getPose(), iscale, params.getDistortion(), center);                                                   \
                                                                                                                     \
    return Status::OK();                                                                                             \
  }

#define CUBEMAPMAP_PANO4(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN, EQUIANGULAR)                  \
  Status cubemapMap_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##EQUIANGULAR(         \
      GPU::Buffer<uint32_t> xPos, GPU::Buffer<uint32_t> xNeg, GPU::Buffer<uint32_t> yPos, GPU::Buffer<uint32_t> yNeg, \
      GPU::Buffer<uint32_t> zPos, GPU::Buffer<uint32_t> zNeg, const PanoDefinition& pano, const InputDefinition& im,  \
      videoreaderid_t imId, GPU::Buffer<const unsigned char> maskDevBuffer, GPU::Stream stream) const {               \
    GeometryDefinition geometry = im.getGeometries().at(0);                                                           \
    TransformGeoParams params(im, geometry, pano);                                                                    \
                                                                                                                      \
    float2 center, iscale;                                                                                            \
    center.x = (float)im.getCenterX(geometry);                                                                        \
    center.y = (float)im.getCenterY(geometry);                                                                        \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                  \
    iscale.y = (float)geometry.getVerticalFocal();                                                                    \
                                                                                                                      \
    dim3 dimBlock(ZONE_KERNEL_BLOCK_SIZE_X, ZONE_KERNEL_BLOCK_SIZE_Y, 1);                                             \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(pano.getLength(), dimBlock.x),                                               \
                 (unsigned)Cuda::ceilDiv(pano.getLength(), dimBlock.y), 1);                                           \
                                                                                                                      \
    cubemapMapKernel_XPos_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR<<< \
        dimGrid, dimBlock, 0, stream.get()>>>(                                                                        \
        xPos.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),    \
        center);                                                                                                      \
    cubemapMapKernel_XNeg_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR<<< \
        dimGrid, dimBlock, 0, stream.get()>>>(                                                                        \
        xNeg.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),    \
        center);                                                                                                      \
    cubemapMapKernel_YPos_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR<<< \
        dimGrid, dimBlock, 0, stream.get()>>>(                                                                        \
        yPos.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),    \
        center);                                                                                                      \
    cubemapMapKernel_YNeg_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR<<< \
        dimGrid, dimBlock, 0, stream.get()>>>(                                                                        \
        yNeg.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),    \
        center);                                                                                                      \
    cubemapMapKernel_ZPos_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR<<< \
        dimGrid, dimBlock, 0, stream.get()>>>(                                                                        \
        zPos.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),    \
        center);                                                                                                      \
    cubemapMapKernel_ZNeg_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR<<< \
        dimGrid, dimBlock, 0, stream.get()>>>(                                                                        \
        zNeg.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, params.getDistortion(),    \
        center);                                                                                                      \
                                                                                                                      \
    return Status::OK();                                                                                              \
  }

#define MAPDISTORTION_PANO3(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN)                            \
  Status mapDistortion_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN(                      \
      int time, GPU::Buffer<unsigned char> devOut, const Rect& outputBounds, const PanoDefinition& pano,              \
      const InputDefinition& im, GPU::Stream gpuStream) const {                                                       \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                        \
    TransformGeoParams params(im, geometry, pano);                                                                    \
    /*NOTE: here we assume that height and width are multiples of dimBlock.x and dimblock.y*/                         \
    dim3 dimBlock(ZONE_KERNEL_BLOCK_SIZE_X, ZONE_KERNEL_BLOCK_SIZE_Y, 1);                                             \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(outputBounds.getWidth(), dimBlock.x),                                        \
                 (unsigned)Cuda::ceilDiv(outputBounds.getHeight(), dimBlock.y), 1);                                   \
                                                                                                                      \
    float2 center, iscale, pscale;                                                                                    \
    center.x = (float)im.getCenterX(geometry);                                                                        \
    center.y = (float)im.getCenterY(geometry);                                                                        \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                  \
    iscale.y = (float)geometry.getVerticalFocal();                                                                    \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);         \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);    \
                                                                                                                      \
    distortionKernel_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS<<<dimGrid, dimBlock, 0, \
                                                                                                gpuStream.get()>>>(   \
        devOut.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)outputBounds.getWidth(),           \
        (unsigned)outputBounds.getHeight(), (unsigned)outputBounds.left(), (unsigned)outputBounds.top(),              \
        (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), (unsigned)im.getCropLeft(),                            \
        (unsigned)im.getCropRight(), (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), pscale,                 \
        params.getPose(), iscale, params.getDistortion(), center);                                                    \
    return Status::OK();                                                                                              \
  }

#define UNDISTORT_INPUT5(INPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN, OUTPROJECTION, PHOTORESPONSE)               \
  Status                                                                                                                         \
      undistortInput_##INPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##OUTPROJECTION##_##PHOTORESPONSE(  \
          int time, GPU::Surface& dst, const GPU::Surface& src, const PanoDefinition& pano,                                      \
          const InputDefinition& recordedInput, const InputDefinition& undistortedOutput, GPU::Stream& stream) const {           \
    GeometryDefinition inGeometry = recordedInput.getGeometries().at(time);                                                      \
    /* let's have rig center == camera center for the mapping */                                                                 \
    inGeometry.setTranslationX(0);                                                                                               \
    inGeometry.setTranslationY(0);                                                                                               \
    inGeometry.setTranslationZ(0);                                                                                               \
                                                                                                                                 \
    TransformGeoParams inParams(recordedInput, inGeometry, pano);                                                                \
                                                                                                                                 \
    GeometryDefinition outGeometry = undistortedOutput.getGeometries().at(time);                                                 \
    /* let's have rig center == camera center for the mapping */                                                                 \
    outGeometry.setTranslationX(0);                                                                                              \
    outGeometry.setTranslationY(0);                                                                                              \
    outGeometry.setTranslationZ(0);                                                                                              \
                                                                                                                                 \
    TransformGeoParams outParams(undistortedOutput, outGeometry, pano);                                                          \
                                                                                                                                 \
    const float2 inScale{(float)inGeometry.getHorizontalFocal(), (float)inGeometry.getVerticalFocal()};                          \
    const float2 outScale{(float)outGeometry.getHorizontalFocal(), (float)outGeometry.getVerticalFocal()};                       \
                                                                                                                                 \
    const float2 center{(float)recordedInput.getCenterX(inGeometry), (float)recordedInput.getCenterY(inGeometry)};               \
                                                                                                                                 \
    const float3 colorMult = PhotoTransform::ColorCorrectionParams(                                                              \
                                 recordedInput.getExposureValue().at(time), recordedInput.getRedCB().at(time),                   \
                                 recordedInput.getGreenCB().at(time), recordedInput.getBlueCB().at(time))                        \
                                 .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),             \
                                                         pano.getGreenCB().at(time), pano.getBlueCB().at(time));                 \
                                                                                                                                 \
    dim3 dimBlock(ZONE_KERNEL_BLOCK_SIZE_X, ZONE_KERNEL_BLOCK_SIZE_Y, 1);                                                        \
    dim3 dimGrid((unsigned)Cuda::ceilDiv(undistortedOutput.getWidth(), dimBlock.x),                                              \
                 (unsigned)Cuda::ceilDiv(undistortedOutput.getHeight(), dimBlock.y), 1);                                         \
    warpInputKernel_##INPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##OUTPROJECTION##_##PHOTORESPONSE<<< \
        dimGrid, dimBlock, 0, stream.get()>>>(                                                                                   \
        dst.get().surface(), src.get().texture(), (unsigned)recordedInput.getWidth(),                                            \
        (unsigned)recordedInput.getHeight(), (unsigned)undistortedOutput.getWidth(),                                             \
        (unsigned)undistortedOutput.getHeight(), (unsigned)recordedInput.getCropLeft(),                                          \
        (unsigned)recordedInput.getCropRight(), (unsigned)recordedInput.getCropTop(),                                            \
        (unsigned)recordedInput.getCropBottom(), inParams.getPose(), outParams.getPoseInverse(),                                 \
        inParams.getDistortion(), center, inScale, outScale, colorMult, photo->getDevicePhotoParam().floatParam,                 \
        (const float*)photo->getDevicePhotoParam().transformData, (float)recordedInput.getVignettingCenterX(),                   \
        (float)recordedInput.getVignettingCenterY(),                                                                             \
        (float)TransformGeoParams::getInverseDemiDiagonalSquared(recordedInput),                                                 \
        (float)recordedInput.getVignettingCoeff0(), (float)recordedInput.getVignettingCoeff1(),                                  \
        (float)recordedInput.getVignettingCoeff2(), (float)recordedInput.getVignettingCoeff3());                                 \
    return Status::OK();                                                                                                         \
  }

}  // namespace
}  // namespace Core
}  // namespace VideoStitch

#include "backend/common/core1/transform.impl"

namespace VideoStitch {
namespace Core {
Transform* Transform::create(const InputDefinition& im) { return createTransform(im, ImageMerger::Format::None); }

Transform* Transform::create(const InputDefinition& im, const ImageMerger::Format type) {
  return createTransform(im, type);
}

}  // namespace Core
}  // namespace VideoStitch
