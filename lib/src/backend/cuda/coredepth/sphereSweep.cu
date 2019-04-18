// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/coredepth/sweep.hpp"

#include "backend/common/coredepth/sphereSweepParams.h"

#include "../surface.hpp"

#include "gpu/memcpy.hpp"

#include "core/transformGeoParams.hpp"

#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/panoDef.hpp"

#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceBuffer2D.hpp"
#include "backend/cuda/surface.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "cuda/util.hpp"
#include "gpu/buffer.hpp"

#include "kernels/sphereSweepKernel.cu"

#include <math.h>

static const int CudaBlockSize = 16;

namespace VideoStitch {
namespace GPU {

static int numCall = 0;

PotentialValue<struct InputParams6> prepareInputParams(const Core::PanoDefinition& panoDef, int time,
                                                       float scale = 1.f) {
  struct InputParams6 inputParamsArray;

  for (videoreaderid_t videoInputID = 0; videoInputID < panoDef.numVideoInputs(); videoInputID++) {
    const Core::InputDefinition& input = panoDef.getVideoInput(videoInputID);
    const Core::GeometryDefinition geometry = input.getGeometries().at(time);
    Core::TransformGeoParams params(input, geometry, panoDef);

    if (geometry.hasDistortion()) {
      return PotentialValue<struct InputParams6>({Origin::Stitcher, ErrType::ImplementationError,
                                                  "Sphere sweep does not handle distortion parameters in inputs"});
    }
    float2 center, iscale;
    center.x = (float)input.getCenterX(geometry) / scale;
    center.y = (float)input.getCenterY(geometry) / scale;
    iscale.x = (float)geometry.getHorizontalFocal() / scale;
    iscale.y = (float)geometry.getVerticalFocal() / scale;

    InputParams& inputParams = inputParamsArray.params[videoInputID];
    inputParams.distortion = params.getDistortion();
    inputParams.transform = params.getPose();
    inputParams.inverseTransform = params.getPoseInverse();
    inputParams.scale = iscale;
    inputParams.centerShift = center;
    inputParams.texWidth = (int)(input.getWidth() / scale);
    inputParams.texHeight = (int)(input.getHeight() / scale);
    inputParams.cropLeft = (int)(input.getCropLeft() / scale);
    inputParams.cropRight = (int)(input.getCropRight() / scale);
    inputParams.cropTop = (int)(input.getCropTop() / scale);
    inputParams.cropBottom = (int)(input.getCropBottom() / scale);
  }

  return PotentialValue<struct InputParams6>(inputParamsArray);
}

static read_only image2d_t getSurfaceFromMap(const videoreaderid_t index,
                                             const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces) {
  return (surfaces.find(index) != surfaces.end()) ? surfaces.find(index)->second->pimpl->surface->get().texture() : 0;
}

Status splatInputWithDepthIntoPano(const Core::PanoDefinition& panoDef, Core::PanoSurface& pano,
                                   const GPU::Surface& depthSurface,
                                   const std::map<videoreaderid_t, Core::SourceSurface*>& inputSurfaces,
                                   GPU::Stream stream) {
  // TODO
  int time = 0;

  const videoreaderid_t inputID = 0;

  Buffer<uint32_t> panoBuffer = pano.pimpl->buffer;

  float2 pscale;
  pscale.x = Core::TransformGeoParams::computePanoScale(Core::PanoProjection::Equirectangular, pano.getWidth(), 360.f);
  pscale.y =
      2 * Core::TransformGeoParams::computePanoScale(Core::PanoProjection::Equirectangular, pano.getHeight(), 360.f);

  auto potInputParamsArray = prepareInputParams(panoDef, time);
  FAIL_RETURN(potInputParamsArray.status());
  const InputParams& referenceInput = potInputParamsArray.value().params[inputID];

  const float offset = cosf(numCall++ / 20.f * (float)M_PI / 2.f) * 0.2f;

  const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(referenceInput.texWidth, dimBlock.x),
                     (unsigned)Cuda::ceilDiv(referenceInput.texHeight, dimBlock.y), 1);

  Core::splatInputWithDepthIntoPano<<<dimGrid, dimBlock, 0, stream.get()>>>(
      panoBuffer.get(), (unsigned)pano.getWidth(), (unsigned)pano.getHeight(), pscale,
      getSurfaceFromMap(inputID, inputSurfaces), depthSurface.get().surface(), referenceInput, panoDef.numVideoInputs(),
      offset);

  Logger::get(Logger::Info) << "SphereSweep frame " << numCall << std::endl;
  return Status::OK();
}

Status sphereSweepInput(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                        const std::map<videoreaderid_t, Core::SourceSurface*>& inputSurfaces,
                        const Core::PanoDefinition& panoDef, GPU::Stream& stream, const float scale) {
  // debug command line pipeline: just copy input surface to output surface
  // via temporary buffer as we don't have a surface->surface copy function
  //  auto tmpBuf = GPU::uniqueBuffer<uint32_t>(inputDef.getWidth() * inputDef.getHeight(), "tmp bfu");
  //  GPU::memcpyAsync(tmpBuf.borrow(), *gpuSurf, stream);
  //  GPU::memcpyAsync(dst, tmpBuf.borrow_const(), stream);
  // stream.synchronize()

  if (panoDef.numVideoInputs() > maxDepthInputs()) {
    return Status{Origin::Stitcher, ErrType::ImplementationError,
                  "Sphere sweep only implemented up to 6 inputs (hardcoded)"};
  }

  auto potInputParamsArray = prepareInputParams(panoDef, frame, scale);
  FAIL_RETURN(potInputParamsArray.status());
  const struct InputParams6 inputParamsArray = potInputParamsArray.releaseValue();

  const InputParams& referenceInput = inputParamsArray.params[sourceID];

  // Running a kernel that takes > 1s destabilizes the system
  // (Display manager resets or kernel panic)
  // As the current version is not optimised and works at full resolution it can take several seconds to complete
  // --> Tile the work. Each tile should complete in less than 1 second.
  const int numBlocks = 16;
  // Make sure texture width is a multiple of numBlocks
  const int paddedTexWidth = (int)Cuda::ceilDiv(referenceInput.texWidth, numBlocks) * numBlocks;
  const int paddedTexHeight = (int)Cuda::ceilDiv(referenceInput.texHeight, numBlocks) * numBlocks;
  for (int cx = 0; cx < numBlocks; cx++) {
    for (int cy = 0; cy < numBlocks; cy++) {
      const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
      const dim3 dimGrid((unsigned)Cuda::ceilDiv(paddedTexWidth / numBlocks, dimBlock.x),
                         (unsigned)Cuda::ceilDiv(paddedTexHeight / numBlocks, dimBlock.y), 1);
      Core::sphereSweepInputKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
          dst.get().surface(), (unsigned)dst.width(), (unsigned)dst.height(), nullptr,
          getSurfaceFromMap(0, inputSurfaces), getSurfaceFromMap(1, inputSurfaces), getSurfaceFromMap(2, inputSurfaces),
          getSurfaceFromMap(3, inputSurfaces), getSurfaceFromMap(4, inputSurfaces), getSurfaceFromMap(5, inputSurfaces),
          inputParamsArray, sourceID, panoDef.numVideoInputs(), cx, cy, paddedTexWidth / numBlocks,
          paddedTexHeight / numBlocks);
      // Force synchronization after tile computation for system stability
      stream.synchronize();
    }
  }
  Logger::get(Logger::Info) << "SphereSweep frame " << frame << " input " << sourceID << std::endl;
  return Status::OK();
}

Status sphereSweepInputSGM(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                           GPU::HostBuffer<unsigned short>& hostCostVolume,
                           const std::map<videoreaderid_t, Core::SourceSurface*>& inputSurfaces,
                           const Core::PanoDefinition& panoDef, GPU::Stream& stream, const float scale) {
  // debug command line pipeline: just copy input surface to output surface
  // via temporary buffer as we don't have a surface->surface copy function
  //  auto tmpBuf = GPU::uniqueBuffer<uint32_t>(inputDef.getWidth() * inputDef.getHeight(), "tmp bfu");
  //  GPU::memcpyAsync(tmpBuf.borrow(), *gpuSurf, stream);
  //  GPU::memcpyAsync(dst, tmpBuf.borrow_const(), stream);
  // stream.synchronize()

  if (panoDef.numVideoInputs() > maxDepthInputs()) {
    return Status{Origin::Stitcher, ErrType::ImplementationError,
                  "Sphere sweep only implemented up to 6 inputs (hardcoded)"};
  }

  auto potInputParamsArray = prepareInputParams(panoDef, frame, scale);
  FAIL_RETURN(potInputParamsArray.status());
  const struct InputParams6 inputParamsArray = potInputParamsArray.releaseValue();

  const InputParams& referenceInput = inputParamsArray.params[sourceID];

  GPU::UniqueBuffer<unsigned short> devCostVolume;
  PROPAGATE_FAILURE_STATUS(
      devCostVolume.alloc(referenceInput.texWidth * referenceInput.texHeight * numSphereSweeps(), "SGM score volume"));

  // Running a kernel that takes > 1s destabilizes the system
  // (Display manager resets or kernel panic)
  // As the current version is not optimised and works at full resolution it can take several seconds to complete
  // --> Tile the work. Each tile should complete in less than 1 second.
  const int numBlocks = 16;
  // Make sure texture width is a multiple of numBlocks
  const int paddedTexWidth = (int)Cuda::ceilDiv(referenceInput.texWidth, numBlocks) * numBlocks;
  const int paddedTexHeight = (int)Cuda::ceilDiv(referenceInput.texHeight, numBlocks) * numBlocks;
  for (int cx = 0; cx < numBlocks; cx++) {
    for (int cy = 0; cy < numBlocks; cy++) {
      const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
      const dim3 dimGrid((unsigned)Cuda::ceilDiv(paddedTexWidth / numBlocks, dimBlock.x),
                         (unsigned)Cuda::ceilDiv(paddedTexHeight / numBlocks, dimBlock.y), 1);
      Core::sphereSweepInputKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
          dst.get().surface(), (unsigned)dst.width(), (unsigned)dst.height(), devCostVolume.borrow().devicePtr(),
          getSurfaceFromMap(0, inputSurfaces), getSurfaceFromMap(1, inputSurfaces), getSurfaceFromMap(2, inputSurfaces),
          getSurfaceFromMap(3, inputSurfaces), getSurfaceFromMap(4, inputSurfaces), getSurfaceFromMap(5, inputSurfaces),
          inputParamsArray, sourceID, panoDef.numVideoInputs(), cx, cy, paddedTexWidth / numBlocks,
          paddedTexHeight / numBlocks);
      // Force synchronization after tile computation for system stability
      FAIL_RETURN(stream.synchronize());
    }
  }
  Logger::get(Logger::Info) << "SphereSweep frame " << frame << " input " << sourceID << std::endl;

  // copy scoreVolume to host
  FAIL_RETURN(GPU::memcpyAsync(
      hostCostVolume.hostPtr(), devCostVolume.borrow_const(),
      referenceInput.texWidth * referenceInput.texHeight * numSphereSweeps() * sizeof(unsigned short), stream));
  stream.synchronize();
  Logger::get(Logger::Info) << "SphereSweep score volume copied back to host" << std::endl;

  return Status::OK();
}

Status sphereSweepInputDisparityToDepth(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                                        GPU::HostBuffer<short>& hostDisparity, bool useHostDisparity,
                                        const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces,
                                        const Core::PanoDefinition& panoDef, GPU::Stream& stream, const float scale) {
  if (panoDef.numVideoInputs() > maxDepthInputs()) {
    return Status{Origin::Stitcher, ErrType::ImplementationError,
                  "Sphere sweep only implemented up to 6 inputs (hardcoded)"};
  }

  auto potInputParamsArray = prepareInputParams(panoDef, frame, scale);
  FAIL_RETURN(potInputParamsArray.status());
  const struct InputParams6 inputParamsArray = potInputParamsArray.releaseValue();

  const InputParams& referenceInput = inputParamsArray.params[sourceID];

  // copy host disparity to GPU buffer
  PotentialValue<GPU::Buffer<short>> potDevBuf =
      GPU::Buffer<short>::allocate(referenceInput.texWidth * referenceInput.texHeight, "SGM output disparity");
  FAIL_RETURN(potDevBuf.status());
  GPU::Buffer<short> devDisparity(potDevBuf.releaseValue());

  FAIL_RETURN(GPU::memcpyAsync(devDisparity, hostDisparity.hostPtr(),
                               referenceInput.texWidth * referenceInput.texHeight * sizeof(short), stream));
  stream.synchronize();

  // Running a kernel that takes > 1s destabilizes the system
  // (Display manager resets or kernel panic)
  // As the current version is not optimised and works at full resolution it can take several seconds to complete
  // --> Tile the work. Each tile should complete in less than 1 second.
  const int numBlocks = 4;
  // Make sure texture width is a multiple of numBlocks
  const int paddedTexWidth = (int)Cuda::ceilDiv(referenceInput.texWidth, numBlocks) * numBlocks;
  const int paddedTexHeight = (int)Cuda::ceilDiv(referenceInput.texHeight, numBlocks) * numBlocks;
  for (int cx = 0; cx < numBlocks; cx++) {
    for (int cy = 0; cy < numBlocks; cy++) {
      const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
      const dim3 dimGrid((unsigned)Cuda::ceilDiv(paddedTexWidth / numBlocks, dimBlock.x),
                         (unsigned)Cuda::ceilDiv(paddedTexHeight / numBlocks, dimBlock.y), 1);
      Core::sphereSweepInputDisparityToDepthKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
          dst.get().surface(), (unsigned)dst.width(), (unsigned)dst.height(),
          (useHostDisparity) ? devDisparity.devicePtr() : nullptr, getSurfaceFromMap(sourceID, surfaces), cx, cy,
          paddedTexWidth / numBlocks, paddedTexHeight / numBlocks);
      // Force synchronization after tile computation for system stability
      stream.synchronize();
    }
  }

  Logger::get(Logger::Info) << "SphereSweep disparity to depth on input " << sourceID << std::endl;

  FAIL_RETURN(devDisparity.release());

  return Status();
}

Status sphereSweepInputStep(videoreaderid_t sourceID, int frame, GPU::Surface& dst, GPU::Surface& depthSrcNextLevel,
                            const std::map<videoreaderid_t, Core::SourceSurface*>& inputSurfaces,
                            const Core::PanoDefinition& panoDef, GPU::Stream& stream, const float scale) {
  // debug command line pipeline: just copy input surface to output surface
  // via temporary buffer as we don't have a surface->surface copy function
  //  auto tmpBuf = GPU::uniqueBuffer<uint32_t>(inputDef.getWidth() * inputDef.getHeight(), "tmp bfu");
  //  GPU::memcpyAsync(tmpBuf.borrow(), *gpuSurf, stream);
  //  GPU::memcpyAsync(dst, tmpBuf.borrow_const(), stream);
  // stream.synchronize()

  if (panoDef.numVideoInputs() > 6) {
    return Status{Origin::Stitcher, ErrType::ImplementationError,
                  "Sphere sweep only implemented for 6 inputs maximum (hardcoded)"};
  }

  auto potInputParamsArray = prepareInputParams(panoDef, frame, scale);
  FAIL_RETURN(potInputParamsArray.status());
  const struct InputParams6 inputParamsArray = potInputParamsArray.releaseValue();

  const InputParams& referenceInput = inputParamsArray.params[sourceID];

  // Running a kernel that takes > 1s destabilizes the system
  // (Display manager resets or kernel panic)
  // As the current version is not optimised and works at full resolution it can take several seconds to complete
  // --> Tile the work. Each tile should complete in less than 1 second.
  const int numBlocks = 4;
  const int paddedTexWidth = (int)Cuda::ceilDiv(referenceInput.texWidth, numBlocks) * numBlocks;
  const int paddedTexHeight = (int)Cuda::ceilDiv(referenceInput.texHeight, numBlocks) * numBlocks;

  // search around best depth from lower level pyramid
  // search span is in log2, covers [log2(bestDepth) - searchSpan, log2(bestDepth) + searchSpan]
  const float searchSpan = scale / 8.f;  // decrease searched depths on upper levels

  for (int cx = 0; cx < numBlocks; cx++) {
    for (int cy = 0; cy < numBlocks; cy++) {
      const dim3 dimBlock(CudaBlockSize, CudaBlockSize, 1);
      const dim3 dimGrid((unsigned)Cuda::ceilDiv(paddedTexWidth / numBlocks, dimBlock.x),
                         (unsigned)Cuda::ceilDiv(paddedTexHeight / numBlocks, dimBlock.y), 1);
      Core::sphereSweepInputKernelStep<<<dimGrid, dimBlock, 0, stream.get()>>>(
          dst.get().surface(), (unsigned)dst.width(), (unsigned)dst.height(), depthSrcNextLevel.get().surface(),
          (unsigned)depthSrcNextLevel.width(), (unsigned)depthSrcNextLevel.height(),
          getSurfaceFromMap(0, inputSurfaces), getSurfaceFromMap(1, inputSurfaces), getSurfaceFromMap(2, inputSurfaces),
          getSurfaceFromMap(3, inputSurfaces), getSurfaceFromMap(4, inputSurfaces), getSurfaceFromMap(5, inputSurfaces),
          inputParamsArray, sourceID, panoDef.numVideoInputs(), cx, cy, paddedTexWidth / numBlocks,
          paddedTexHeight / numBlocks, searchSpan);
      // Force synchronization after tile computation for system stability
      stream.synchronize();
    }
  }
  Logger::get(Logger::Info) << "SphereSweep step frame " << frame << " input " << sourceID
                            << " search span: " << searchSpan << std::endl;
  return Status::OK();
}

int numSphereSweeps() { return numSphereScales; }

int maxDepthInputs() { return NUM_INPUTS; }

}  // namespace GPU
}  // namespace VideoStitch
