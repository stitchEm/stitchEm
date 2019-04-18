// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../surface.hpp"
#include "../context.hpp"
#include "../kernel.hpp"

namespace {

#include "cubemapMapKernel.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(cubemapMapKernel, false);

#include "undistortKernel.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(undistortKernel, false);

#include "warpCoordKernel.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(warpCoordKernel, false);

#include "warpCoordInputKernel.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(warpCoordInputKernel, false);

#include "warpFaceKernel_x.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(warpFaceKernel_x, false);

#include "warpFaceKernel_y.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(warpFaceKernel_y, false);

#include "warpFaceKernel_z.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(warpFaceKernel_z, false);

#include "warpLookupKernel.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(warpLookupKernel, false);

#include "warpKernel.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(warpKernel, true);

#include "zoneKernel.xxd"
INDIRECT_REGISTER_OPENCL_PROGRAM(zoneKernel, true);

}  // namespace

#define STRING(str) QUOTE(str)
#define QUOTE(str) #str

#ifdef CL_ARGS_WORKAROUND
#define GET_DISTORTION(params) (*(cl_float16*)&params.getDistortion())
#else
#define GET_DISTORTION(params) (params.getDistortion())
#endif

namespace VideoStitch {
namespace Core {
namespace {

#define MAPBUFFER_PANO3(MERGER, INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN, PHOTORESPONSE)          \
  Status                                                                                                               \
      mapBuffer_##MERGER##_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##PHOTORESPONSE( \
          int time, GPU::Buffer<uint32_t> devOut, GPU::Surface& panoSurf, const unsigned char* mask,                   \
          const Rect& outputBounds, const PanoDefinition& pano, const InputDefinition& im, GPU::Surface& surface,      \
          GPU::Stream gpuStream) const {                                                                               \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                         \
    TransformGeoParams params(im, geometry, pano);                                                                     \
    const float3 colorMult =                                                                                           \
        PhotoTransform::ColorCorrectionParams(im.getExposureValue().at(time), im.getRedCB().at(time),                  \
                                              im.getGreenCB().at(time), im.getBlueCB().at(time))                       \
            .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),                        \
                                    pano.getGreenCB().at(time), pano.getBlueCB().at(time));                            \
    const cl_float3 colorMultPadded{{colorMult.x, colorMult.y, colorMult.z}};                                          \
    const unsigned height = ((mergeformat == ImageMerger::Format::Gradient) &&                                         \
                             (outputBounds.getHeight() > (pano.getHeight() - outputBounds.top())))                     \
                                ? (unsigned)(pano.getHeight() - outputBounds.top())                                    \
                                : (unsigned)outputBounds.getHeight();                                                  \
                                                                                                                       \
    float2 center, iscale, pscale;                                                                                     \
    center.x = (float)im.getCenterX(geometry);                                                                         \
    center.y = (float)im.getCenterY(geometry);                                                                         \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                   \
    iscale.y = (float)geometry.getVerticalFocal();                                                                     \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);          \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);     \
    auto kernel2D =                                                                                                    \
        GPU::Kernel::get(                                                                                              \
            PROGRAM(warpKernel),                                                                                       \
            "warpKernel_" QUOTE(                                                                                       \
                MERGER##_##INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE))  \
            .setup2D(gpuStream, (unsigned)outputBounds.getWidth(), (unsigned)outputBounds.getHeight());                \
    return kernel2D.enqueueWithKernelArgs(                                                                             \
        devOut.get(), panoSurf.get(), mask, surface.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(),          \
        (unsigned)outputBounds.getWidth(), height, (unsigned)outputBounds.left(), (unsigned)outputBounds.top(),        \
        (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), (unsigned)im.getCropLeft(),                             \
        (unsigned)im.getCropRight(), (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), pscale,                  \
        params.getPose(), iscale, GET_DISTORTION(params), center, colorMultPadded,                                     \
        photo->getDevicePhotoParam().floatParam, photo->getDevicePhotoParam().transformData,                           \
        (float)im.getVignettingCenterX(), (float)im.getVignettingCenterY(),                                            \
        (float)TransformGeoParams::getInverseDemiDiagonalSquared(im), (float)im.getVignettingCoeff0(),                 \
        (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(), (float)im.getVignettingCoeff3());            \
  }

#define WARP_CUBEMAP3(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN, PHOTORESPONSE, EQUIANGULAR)              \
  Status                                                                                                                      \
      warpCubemap_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##PHOTORESPONSE##_##EQUIANGULAR( \
          int time, GPU::Buffer<uint32_t> xPos, const Rect& xPosBB, GPU::Buffer<uint32_t> xNeg, const Rect& xNegBB,           \
          GPU::Buffer<uint32_t> yPos, const Rect& yPosBB, GPU::Buffer<uint32_t> yNeg, const Rect& yNegBB,                     \
          GPU::Buffer<uint32_t> zPos, const Rect& zPosBB, GPU::Buffer<uint32_t> zNeg, const Rect& zNegBB,                     \
          const PanoDefinition& pano, const InputDefinition& im, GPU::Surface& surface, GPU::Stream gpuStream) const {        \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                                \
    TransformGeoParams params(im, geometry, pano);                                                                            \
    const float3 colorMult =                                                                                                  \
        PhotoTransform::ColorCorrectionParams(im.getExposureValue().at(time), im.getRedCB().at(time),                         \
                                              im.getGreenCB().at(time), im.getBlueCB().at(time))                              \
            .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),                               \
                                    pano.getGreenCB().at(time), pano.getBlueCB().at(time));                                   \
    const cl_float3 colorMultPadded{{colorMult.x, colorMult.y, colorMult.z}};                                                 \
                                                                                                                              \
    float2 center, iscale;                                                                                                    \
    center.x = (float)im.getCenterX(geometry);                                                                                \
    center.y = (float)im.getCenterY(geometry);                                                                                \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                          \
    iscale.y = (float)geometry.getVerticalFocal();                                                                            \
                                                                                                                              \
    if (!xPosBB.empty()) {                                                                                                    \
      auto kernel2D =                                                                                                         \
          GPU::Kernel::get(                                                                                                   \
              PROGRAM(warpFaceKernel_x),                                                                                      \
              "warpKernel_XPos_" QUOTE(                                                                                       \
                  INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR))  \
              .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                                    \
      return kernel2D.enqueueWithKernelArgs(                                                                                  \
          xPos.get(), surface.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)xPosBB.getWidth(),          \
          (unsigned)xPosBB.getHeight(), (unsigned)xPosBB.left(), (unsigned)xPosBB.top(), (unsigned)pano.getLength(),          \
          (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(), (unsigned)im.getCropTop(),                                 \
          (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params), center, colorMultPadded,            \
          photo->getDevicePhotoParam().floatParam, photo->getDevicePhotoParam().transformData,                                \
          (float)im.getVignettingCenterX(), (float)im.getVignettingCenterY(),                                                 \
          (float)TransformGeoParams::getInverseDemiDiagonalSquared(im), (float)im.getVignettingCoeff0(),                      \
          (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(), (float)im.getVignettingCoeff3());                 \
    }                                                                                                                         \
    if (!xNegBB.empty()) {                                                                                                    \
      auto kernel2D =                                                                                                         \
          GPU::Kernel::get(                                                                                                   \
              PROGRAM(warpFaceKernel_x),                                                                                      \
              "warpKernel_XNeg_" QUOTE(                                                                                       \
                  INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR))  \
              .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                                    \
      return kernel2D.enqueueWithKernelArgs(                                                                                  \
          xNeg.get(), surface.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)xNegBB.getWidth(),          \
          (unsigned)xNegBB.getHeight(), (unsigned)xNegBB.left(), (unsigned)xNegBB.top(), (unsigned)pano.getLength(),          \
          (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(), (unsigned)im.getCropTop(),                                 \
          (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params), center, colorMultPadded,            \
          photo->getDevicePhotoParam().floatParam, photo->getDevicePhotoParam().transformData,                                \
          (float)im.getVignettingCenterX(), (float)im.getVignettingCenterY(),                                                 \
          (float)TransformGeoParams::getInverseDemiDiagonalSquared(im), (float)im.getVignettingCoeff0(),                      \
          (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(), (float)im.getVignettingCoeff3());                 \
    }                                                                                                                         \
    if (!yPosBB.empty()) {                                                                                                    \
      auto kernel2D =                                                                                                         \
          GPU::Kernel::get(                                                                                                   \
              PROGRAM(warpFaceKernel_y),                                                                                      \
              "warpKernel_YPos_" QUOTE(                                                                                       \
                  INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR))  \
              .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                                    \
      return kernel2D.enqueueWithKernelArgs(                                                                                  \
          yPos.get(), surface.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)yPosBB.getWidth(),          \
          (unsigned)yPosBB.getHeight(), (unsigned)yPosBB.left(), (unsigned)yPosBB.top(), (unsigned)pano.getLength(),          \
          (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(), (unsigned)im.getCropTop(),                                 \
          (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params), center, colorMultPadded,            \
          photo->getDevicePhotoParam().floatParam, photo->getDevicePhotoParam().transformData,                                \
          (float)im.getVignettingCenterX(), (float)im.getVignettingCenterY(),                                                 \
          (float)TransformGeoParams::getInverseDemiDiagonalSquared(im), (float)im.getVignettingCoeff0(),                      \
          (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(), (float)im.getVignettingCoeff3());                 \
    }                                                                                                                         \
    if (!yNegBB.empty()) {                                                                                                    \
      auto kernel2D =                                                                                                         \
          GPU::Kernel::get(                                                                                                   \
              PROGRAM(warpFaceKernel_y),                                                                                      \
              "warpKernel_YNeg_" QUOTE(                                                                                       \
                  INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR))  \
              .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                                    \
      return kernel2D.enqueueWithKernelArgs(                                                                                  \
          yNeg.get(), surface.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)yNegBB.getWidth(),          \
          (unsigned)yNegBB.getHeight(), (unsigned)yNegBB.left(), (unsigned)yNegBB.top(), (unsigned)pano.getLength(),          \
          (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(), (unsigned)im.getCropTop(),                                 \
          (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params), center, colorMultPadded,            \
          photo->getDevicePhotoParam().floatParam, photo->getDevicePhotoParam().transformData,                                \
          (float)im.getVignettingCenterX(), (float)im.getVignettingCenterY(),                                                 \
          (float)TransformGeoParams::getInverseDemiDiagonalSquared(im), (float)im.getVignettingCoeff0(),                      \
          (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(), (float)im.getVignettingCoeff3());                 \
    }                                                                                                                         \
    if (!zPosBB.empty()) {                                                                                                    \
      auto kernel2D =                                                                                                         \
          GPU::Kernel::get(                                                                                                   \
              PROGRAM(warpFaceKernel_z),                                                                                      \
              "warpKernel_ZPos_" QUOTE(                                                                                       \
                  INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR))  \
              .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                                    \
      return kernel2D.enqueueWithKernelArgs(                                                                                  \
          zPos.get(), surface.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)zPosBB.getWidth(),          \
          (unsigned)zPosBB.getHeight(), (unsigned)zPosBB.left(), (unsigned)zPosBB.top(), (unsigned)pano.getLength(),          \
          (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(), (unsigned)im.getCropTop(),                                 \
          (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params), center, colorMultPadded,            \
          photo->getDevicePhotoParam().floatParam, photo->getDevicePhotoParam().transformData,                                \
          (float)im.getVignettingCenterX(), (float)im.getVignettingCenterY(),                                                 \
          (float)TransformGeoParams::getInverseDemiDiagonalSquared(im), (float)im.getVignettingCoeff0(),                      \
          (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(), (float)im.getVignettingCoeff3());                 \
    }                                                                                                                         \
    if (!zNegBB.empty()) {                                                                                                    \
      auto kernel2D =                                                                                                         \
          GPU::Kernel::get(                                                                                                   \
              PROGRAM(warpFaceKernel_z),                                                                                      \
              "warpKernel_ZNeg_" QUOTE(                                                                                       \
                  INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##PHOTORESPONSE##_##EQUIANGULAR))  \
              .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                                    \
      return kernel2D.enqueueWithKernelArgs(                                                                                  \
          zNeg.get(), surface.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)zNegBB.getWidth(),          \
          (unsigned)zNegBB.getHeight(), (unsigned)zNegBB.left(), (unsigned)zNegBB.top(), (unsigned)pano.getLength(),          \
          (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(), (unsigned)im.getCropTop(),                                 \
          (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params), center, colorMultPadded,            \
          photo->getDevicePhotoParam().floatParam, photo->getDevicePhotoParam().transformData,                                \
          (float)im.getVignettingCenterX(), (float)im.getVignettingCenterY(),                                                 \
          (float)TransformGeoParams::getInverseDemiDiagonalSquared(im), (float)im.getVignettingCoeff0(),                      \
          (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(), (float)im.getVignettingCoeff3());                 \
    }                                                                                                                         \
                                                                                                                              \
    return Status::OK();                                                                                                      \
  }

#define MAPBUFFERLOOKUP_PANO3(MERGER, ISWITHIN, PHOTORESPONSE)                                                        \
  Status mapBufferLookup_##MERGER##_##ISWITHIN##_##PHOTORESPONSE(                                                     \
      int time, GPU::Buffer<uint32_t> devOut, GPU::Surface& panoSurf, const unsigned char* mask,                      \
      const GPU::Surface& coordIn, const float coordShrinkFactor, const Rect& outputBounds,                           \
      const PanoDefinition& pano, const InputDefinition& im, GPU::Surface& surface, GPU::Stream gpuStream) const {    \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                        \
    TransformGeoParams params(im, geometry, pano);                                                                    \
    const float3 colorMult =                                                                                          \
        PhotoTransform::ColorCorrectionParams(im.getExposureValue().at(time), im.getRedCB().at(time),                 \
                                              im.getGreenCB().at(time), im.getBlueCB().at(time))                      \
            .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),                       \
                                    pano.getGreenCB().at(time), pano.getBlueCB().at(time));                           \
    const cl_float3 colorMultPadded{{colorMult.x, colorMult.y, colorMult.z}};                                         \
    const unsigned height = ((mergeformat == ImageMerger::Format::Gradient) &&                                        \
                             (outputBounds.getHeight() > (pano.getHeight() - outputBounds.top())))                    \
                                ? (unsigned)(pano.getHeight() - outputBounds.top())                                   \
                                : (unsigned)outputBounds.getHeight();                                                 \
                                                                                                                      \
    auto kernel2D =                                                                                                   \
        GPU::Kernel::get(PROGRAM(warpLookupKernel), "warpLookupKernel_" QUOTE(MERGER##_##ISWITHIN##_##PHOTORESPONSE)) \
            .setup2D(gpuStream, (unsigned)outputBounds.getWidth(), (unsigned)outputBounds.getHeight());               \
    return kernel2D.enqueueWithKernelArgs(                                                                            \
        devOut.get(), panoSurf.get(), mask, coordIn.get(), coordShrinkFactor, surface.get(), (unsigned)im.getWidth(), \
        (unsigned)im.getHeight(), (unsigned)outputBounds.getWidth(), height, (unsigned)outputBounds.left(),           \
        (unsigned)outputBounds.top(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(),                          \
        (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(), (unsigned)im.getCropTop(),                           \
        (unsigned)im.getCropBottom(), colorMultPadded, photo->getDevicePhotoParam().floatParam,                       \
        photo->getDevicePhotoParam().transformData, (float)im.getVignettingCenterX(),                                 \
        (float)im.getVignettingCenterY(), (float)TransformGeoParams::getInverseDemiDiagonalSquared(im),               \
        (float)im.getVignettingCoeff0(), (float)im.getVignettingCoeff1(), (float)im.getVignettingCoeff2(),            \
        (float)im.getVignettingCoeff3());                                                                             \
  }

#define MAPBUFFERCOORD_PANO3(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN)                            \
  Status mapBufferCoord_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN(                      \
      int time, GPU::Surface& devOut, const Rect& outputBounds, const PanoDefinition& pano, const InputDefinition& im, \
      GPU::Stream gpuStream) const {                                                                                   \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                         \
    TransformGeoParams params(im, geometry, pano);                                                                     \
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
    auto kernel2D =                                                                                                    \
        GPU::Kernel::get(                                                                                              \
            PROGRAM(warpCoordKernel),                                                                                  \
            "warpCoordKernel_" QUOTE(INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS))          \
            .setup2D(gpuStream, (unsigned)outputBounds.getWidth(), (unsigned)outputBounds.getHeight());                \
    return kernel2D.enqueueWithKernelArgs(                                                                             \
        devOut.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)im.getCropLeft(),                   \
        (unsigned)im.getCropRight(), (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(),                          \
        (unsigned)outputBounds.getWidth(), (unsigned)outputBounds.getHeight(), (unsigned)outputBounds.left(),          \
        (unsigned)outputBounds.top(), panoWidth, panoHeight, (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),  \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), pscale, params.getPose(), iscale,                     \
        GET_DISTORTION(params), center);                                                                               \
  }

#define MAPCOORDINPUT_PANO3(INPUTPROJECTION, INVERSEDISTORTIONMETERS, INVERSEDISTORTIONPIXELS, ISWITHIN)              \
  Status mapCoordInput_##INPUTPROJECTION##_##INVERSEDISTORTIONMETERS##_##INVERSEDISTORTIONPIXELS##_##ISWITHIN(        \
      int time, const int scaleFactor, GPU::Buffer<float2> inputCoord, const PanoDefinition& pano,                    \
      const InputDefinition& im, GPU::Stream gpuStream) const {                                                       \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                        \
    TransformGeoParams params(im, geometry, pano);                                                                    \
                                                                                                                      \
    float2 center, iscale, pscale;                                                                                    \
    center.x = (float)im.getCenterX(geometry);                                                                        \
    center.y = (float)im.getCenterY(geometry);                                                                        \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                  \
    iscale.y = (float)geometry.getVerticalFocal();                                                                    \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);         \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);    \
                                                                                                                      \
    auto kernel2D =                                                                                                   \
        GPU::Kernel::get(PROGRAM(warpCoordInputKernel),                                                               \
                         "warpCoordInputKernel_" QUOTE(                                                               \
                             INPUTPROJECTION##_##ISWITHIN##_##INVERSEDISTORTIONMETERS##_##INVERSEDISTORTIONPIXELS))   \
            .setup2D(gpuStream, (unsigned)im.getWidth(), (unsigned)im.getHeight());                                   \
    return kernel2D.enqueueWithKernelArgs(                                                                            \
        inputCoord.get(), scaleFactor, (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)im.getCropLeft(), \
        (unsigned)im.getCropRight(), (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(),                         \
        (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), pscale, params.getPoseInverse(),                       \
        (float)pano.getSphereScale(), iscale, GET_DISTORTION(params), center);                                        \
  }

#define COMPUTEZONE_PANO3(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN)                             \
  Status computeZone_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN(                       \
      GPU::Buffer<uint32_t> devOut, const PanoDefinition& pano, const InputDefinition& im, unsigned imId,            \
      GPU::Buffer<const unsigned char> maskDevBuffer, GPU::Stream stream) const {                                    \
    GeometryDefinition geometry = im.getGeometries().at(0);                                                          \
    TransformGeoParams params(im, geometry, pano);                                                                   \
    float2 center, iscale, pscale;                                                                                   \
    center.x = (float)im.getCenterX(geometry);                                                                       \
    center.y = (float)im.getCenterY(geometry);                                                                       \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                 \
    iscale.y = (float)geometry.getVerticalFocal();                                                                   \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);        \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f);   \
                                                                                                                     \
    std::string kernelName =                                                                                         \
        "zoneKernel_" QUOTE(INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS);                 \
    auto kernel2D = GPU::Kernel::get(PROGRAM(zoneKernel), kernelName)                                                \
                        .setup2D(stream, (unsigned)pano.getWidth(), (unsigned)pano.getHeight());                     \
                                                                                                                     \
    return kernel2D.enqueueWithKernelArgs(                                                                           \
        devOut.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(), \
        (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), (unsigned)im.getCropLeft(),                           \
        (unsigned)im.getCropRight(), (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), pscale,                \
        params.getPose(), iscale, GET_DISTORTION(params), center);                                                   \
  }

#define CUBEMAPMAP_PANO4(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN, EQUIANGULAR)                  \
  Status cubemapMap_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##EQUIANGULAR(         \
      GPU::Buffer<uint32_t> xPos, GPU::Buffer<uint32_t> xNeg, GPU::Buffer<uint32_t> yPos, GPU::Buffer<uint32_t> yNeg, \
      GPU::Buffer<uint32_t> zPos, GPU::Buffer<uint32_t> zNeg, const PanoDefinition& pano, const InputDefinition& im,  \
      videoreaderid_t imId, GPU::Buffer<const unsigned char> maskDevBuffer, GPU::Stream gpuStream) const {            \
    GeometryDefinition geometry = im.getGeometries().at(0);                                                           \
    TransformGeoParams params(im, geometry, pano);                                                                    \
                                                                                                                      \
    float2 center, iscale;                                                                                            \
    center.x = (float)im.getCenterX(geometry);                                                                        \
    center.y = (float)im.getCenterY(geometry);                                                                        \
    iscale.x = (float)geometry.getHorizontalFocal();                                                                  \
    iscale.y = (float)geometry.getVerticalFocal();                                                                    \
                                                                                                                      \
    auto kernXPos =                                                                                                   \
        GPU::Kernel::get(PROGRAM(cubemapMapKernel),                                                                   \
                         "cubemapMapKernel_XPos_" QUOTE(                                                              \
                             INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR)) \
            .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                              \
    FAIL_RETURN(kernXPos.enqueueWithKernelArgs(                                                                       \
        xPos.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params),    \
        center));                                                                                                     \
    auto kernXNeg =                                                                                                   \
        GPU::Kernel::get(PROGRAM(cubemapMapKernel),                                                                   \
                         "cubemapMapKernel_XNeg_" QUOTE(                                                              \
                             INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR)) \
            .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                              \
    FAIL_RETURN(kernXNeg.enqueueWithKernelArgs(                                                                       \
        xNeg.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params),    \
        center));                                                                                                     \
    auto kernYPos =                                                                                                   \
        GPU::Kernel::get(PROGRAM(cubemapMapKernel),                                                                   \
                         "cubemapMapKernel_YPos_" QUOTE(                                                              \
                             INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR)) \
            .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                              \
    FAIL_RETURN(kernYPos.enqueueWithKernelArgs(                                                                       \
        yPos.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params),    \
        center));                                                                                                     \
    auto kernYNeg =                                                                                                   \
        GPU::Kernel::get(PROGRAM(cubemapMapKernel),                                                                   \
                         "cubemapMapKernel_YNeg_" QUOTE(                                                              \
                             INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR)) \
            .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                              \
    FAIL_RETURN(kernYNeg.enqueueWithKernelArgs(                                                                       \
        yNeg.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params),    \
        center));                                                                                                     \
    auto kernZPos =                                                                                                   \
        GPU::Kernel::get(PROGRAM(cubemapMapKernel),                                                                   \
                         "cubemapMapKernel_ZPos_" QUOTE(                                                              \
                             INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR)) \
            .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                              \
    FAIL_RETURN(kernZPos.enqueueWithKernelArgs(                                                                       \
        zPos.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params),    \
        center));                                                                                                     \
    auto kernZNeg =                                                                                                   \
        GPU::Kernel::get(PROGRAM(cubemapMapKernel),                                                                   \
                         "cubemapMapKernel_ZNeg_" QUOTE(                                                              \
                             INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##EQUIANGULAR)) \
            .setup2D(gpuStream, (unsigned)pano.getLength(), (unsigned)pano.getLength());                              \
    FAIL_RETURN(kernZNeg.enqueueWithKernelArgs(                                                                       \
        zNeg.get(), maskDevBuffer.get(), ((uint32_t)1 << imId), (unsigned)im.getWidth(), (unsigned)im.getHeight(),    \
        (unsigned)pano.getLength(), (unsigned)im.getCropLeft(), (unsigned)im.getCropRight(),                          \
        (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), params.getPose(), iscale, GET_DISTORTION(params),    \
        center));                                                                                                     \
                                                                                                                      \
    return Status::OK();                                                                                              \
  }

#define MAPDISTORTION_PANO3(INPUTPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN)                         \
  Status mapDistortion_##INPUTPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN(                   \
      int time, GPU::Buffer<unsigned char> devOut, const Rect& outputBounds, const PanoDefinition& pano,           \
      const InputDefinition& im, GPU::Stream stream) const {                                                       \
    GeometryDefinition geometry = im.getGeometries().at(time);                                                     \
    TransformGeoParams params(im, geometry, pano);                                                                 \
    /*NOTE: here we assume that height and width are multiples of dimBlock.x and dimblock.y*/                      \
    float2 center, iscale, pscale;                                                                                 \
    center.x = (float)im.getCenterX(geometry);                                                                     \
    center.y = (float)im.getCenterY(geometry);                                                                     \
    iscale.x = (float)geometry.getHorizontalFocal();                                                               \
    iscale.y = (float)geometry.getVerticalFocal();                                                                 \
    pscale.x = TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getWidth(), 360.f);      \
    pscale.y = 2 * TransformGeoParams::computePanoScale(PanoProjection::Equirectangular, pano.getHeight(), 360.f); \
                                                                                                                   \
    std::string kernelName =                                                                                       \
        "distortionKernel_" QUOTE(INPUTPROJECTION##_##ISWITHIN##_##DISTORTIONMETERS##_##DISTORTIONPIXELS);         \
    auto kernel2D = GPU::Kernel::get(PROGRAM(distortionKernel), kernelName)                                        \
                        .setup2D(stream, (unsigned)pano.getWidth(), (unsigned)pano.getHeight());                   \
    return kernel2D.enqueueWithKernelArgs(                                                                         \
        devOut.get(), (unsigned)im.getWidth(), (unsigned)im.getHeight(), (unsigned)outputBounds.getWidth(),        \
        (unsigned)outputBounds.getHeight(), (unsigned)outputBounds.left(), (unsigned)outputBounds.top(),           \
        (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), (unsigned)im.getCropLeft(),                         \
        (unsigned)im.getCropRight(), (unsigned)im.getCropTop(), (unsigned)im.getCropBottom(), pscale,              \
        params.getPose(), iscale, GET_DISTORTION(params), center);                                                 \
  }

#define UNDISTORT_INPUT5(INPROJECTION, DISTORTIONMETERS, DISTORTIONPIXELS, ISWITHIN, OUTPROJECTION, PHOTORESPONSE)              \
  Status                                                                                                                        \
      undistortInput_##INPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##OUTPROJECTION##_##PHOTORESPONSE( \
          int time, GPU::Surface& dst, const GPU::Surface& src, const PanoDefinition& pano,                                     \
          const InputDefinition& recordedInput, const InputDefinition& undistortedOutput, GPU::Stream& stream) const {          \
    GeometryDefinition inGeometry = recordedInput.getGeometries().at(time);                                                     \
    /* let's have rig center == camera center for the mapping */                                                                \
    inGeometry.setTranslationX(0);                                                                                              \
    inGeometry.setTranslationY(0);                                                                                              \
    inGeometry.setTranslationZ(0);                                                                                              \
                                                                                                                                \
    TransformGeoParams inParams(recordedInput, inGeometry, pano);                                                               \
                                                                                                                                \
    GeometryDefinition outGeometry = undistortedOutput.getGeometries().at(time);                                                \
    /* let's have rig center == camera center for the mapping */                                                                \
    outGeometry.setTranslationX(0);                                                                                             \
    outGeometry.setTranslationY(0);                                                                                             \
    outGeometry.setTranslationZ(0);                                                                                             \
                                                                                                                                \
    TransformGeoParams outParams(undistortedOutput, outGeometry, pano);                                                         \
                                                                                                                                \
    const float2 inScale{(float)inGeometry.getHorizontalFocal(), (float)inGeometry.getVerticalFocal()};                         \
    const float2 outScale{(float)outGeometry.getHorizontalFocal(), (float)outGeometry.getVerticalFocal()};                      \
                                                                                                                                \
    auto distortion = GET_DISTORTION(inParams);                                                                                 \
    const float2 center{(float)recordedInput.getCenterX(inGeometry), (float)recordedInput.getCenterY(inGeometry)};              \
                                                                                                                                \
    const float3 colorMult = PhotoTransform::ColorCorrectionParams(                                                             \
                                 recordedInput.getExposureValue().at(time), recordedInput.getRedCB().at(time),                  \
                                 recordedInput.getGreenCB().at(time), recordedInput.getBlueCB().at(time))                       \
                                 .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),            \
                                                         pano.getGreenCB().at(time), pano.getBlueCB().at(time));                \
    const cl_float3 colorMultPadded{{colorMult.x, colorMult.y, colorMult.z}};                                                   \
                                                                                                                                \
    std::string kernelName = "warpInputKernel_" QUOTE(                                                                          \
        INPROJECTION##_##DISTORTIONMETERS##_##DISTORTIONPIXELS##_##ISWITHIN##_##OUTPROJECTION##_##PHOTORESPONSE);               \
    auto kernel2D =                                                                                                             \
        GPU::Kernel::get(PROGRAM(undistortKernel), kernelName)                                                                  \
            .setup2D(stream, (unsigned)undistortedOutput.getWidth(), (unsigned)undistortedOutput.getHeight());                  \
    return kernel2D.enqueueWithKernelArgs(                                                                                      \
        dst.get(), src.get(), (unsigned)recordedInput.getWidth(), (unsigned)recordedInput.getHeight(),                          \
        (unsigned)undistortedOutput.getWidth(), (unsigned)undistortedOutput.getHeight(),                                        \
        (unsigned)recordedInput.getCropLeft(), (unsigned)recordedInput.getCropRight(),                                          \
        (unsigned)recordedInput.getCropTop(), (unsigned)recordedInput.getCropBottom(), inParams.getPose(),                      \
        outParams.getPoseInverse(), distortion, center, inScale, outScale, colorMultPadded,                                     \
        photo->getDevicePhotoParam().floatParam, photo->getDevicePhotoParam().transformData,                                    \
        (float)recordedInput.getVignettingCenterX(), (float)recordedInput.getVignettingCenterY(),                               \
        (float)TransformGeoParams::getInverseDemiDiagonalSquared(recordedInput),                                                \
        (float)recordedInput.getVignettingCoeff0(), (float)recordedInput.getVignettingCoeff1(),                                 \
        (float)recordedInput.getVignettingCoeff2(), (float)recordedInput.getVignettingCoeff3());                                \
  }

/*Namespaces end*/
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
