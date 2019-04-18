// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "coredepth/sphereSweep.hpp"

#include "backend/common/coredepth/sphereSweepParams.h"
#include "bilateral/bilateral.hpp"
#include "core/transformGeoParams.hpp"
#include "core/surfacePyramid.hpp"
#include "gpu/allocator.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/surface.hpp"
#include "gpu/coredepth/sweep.hpp"
#include "gpu/image/downsampler.hpp"
#include "gpu/image/sampling.hpp"
#include "parallax/sgm.hpp"

#include "libvideostitch/depthDef.hpp"
#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/panoDef.hpp"

#include "opencv2/imgproc.hpp"

//#define DUMP_IMAGE_PYRAMID
//#define DUMP_DEPTH_IMAGE

#if defined(DUMP_IMAGE_PYRAMID) || defined(DUMP_DEPTH_IMAGE)

#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif  // NDEBUG

#include "util/debugUtils.hpp"
#endif

#include <memory.h>

namespace VideoStitch {
namespace Core {

Status sphereSweepIntoPano(const Core::PanoDefinition& panoDef, const Core::DepthDefinition& depthDef,
                           Core::PanoSurface& pano,
                           const std::map<videoreaderid_t, Core::SourceSurface*>& inputSurfaces,
                           std::vector<InputPyramid>& pyramids, const DepthPyramid& depthPyramid, GPU::Stream stream) {
  if (panoDef.numVideoInputs() > 6) {
    return Status{Origin::Stitcher, ErrType::ImplementationError,
                  "Sphere sweep only implemented for 6 inputs maximum (hardcoded)"};
  }

  if (!depthDef.isMultiScale()) {
    // to implement single scale depth merging: need to allocate a buffer for the depth
    return Status{Origin::Stitcher, ErrType::ImplementationError,
                  "Depth sweep merger only supports multi-scale sweeping"};
  }

  const videoreaderid_t inputID = 0;
  const int frame = 0;

  auto potDepthSurf = Core::OffscreenAllocator::createDepthSurface(
      depthPyramid.getLevel(0).width(), depthPyramid.getLevel(0).height(), "Downsampled depth surface");
  FAIL_RETURN(potDepthSurf.status())
  GPU::Surface& depthSurface = *potDepthSurf.object()->pimpl->surface;

  FAIL_RETURN(sphereSweepInputMultiScale(inputID, frame, depthSurface, pyramids, depthPyramid, panoDef, depthDef,
                                         SGMPostProcessing::Off, BilateralFilterPostProcessing::Off, stream));

  return GPU::splatInputWithDepthIntoPano(panoDef, pano, depthSurface, inputSurfaces, stream);
}

static Status loadRGBAFromSurface(cv::Mat& mat, GPU::Surface& surface) {
  const int2 size{static_cast<int>(surface.width()), static_cast<int>(surface.height())};
  if (mat.type() != CV_8UC4 || mat.rows != size.y || mat.cols != size.x) {
    mat = cv::Mat(size.y, size.x, CV_8UC4);
  }

  return GPU::memcpyBlocking(mat.ptr<uint32_t>(), surface);
}

static Status loadF32C1FromSurface(cv::Mat& mat, GPU::Surface& surface) {
  const int2 size{static_cast<int>(surface.width()), static_cast<int>(surface.height())};
  if (mat.type() != CV_32FC1 || mat.rows != size.y || mat.cols != size.x) {
    mat = cv::Mat(size.y, size.x, CV_32FC1);
  }

  return GPU::memcpyBlocking(mat.ptr<float>(), surface);
}

Status sphereSweepInputSGMSingleScale(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                                      const std::map<videoreaderid_t, Core::SourceSurface*>& inputSurfaces,
                                      const Core::PanoDefinition& panoDef, GPU::Stream& stream, const float scale,
                                      const int uniquenessRatio) {
  PotentialValue<GPU::HostBuffer<SGM::CostType>> potHostCostBuf =
      GPU::HostBuffer<SGM::CostType>::allocate(dst.width() * dst.height() * GPU::numSphereSweeps(), "SGM score volume");
  FAIL_RETURN(potHostCostBuf.status());
  GPU::HostBuffer<SGM::CostType> hostCostVolume(potHostCostBuf.releaseValue());

  FAIL_RETURN(GPU::sphereSweepInputSGM(sourceID, frame, dst, hostCostVolume, inputSurfaces, panoDef, stream, scale));

  PotentialValue<GPU::HostBuffer<SGM::DispType>> potHostDispBuf =
      GPU::HostBuffer<SGM::DispType>::allocate(dst.width() * dst.height(), "SGM disparity");
  FAIL_RETURN(potHostDispBuf.status());
  GPU::HostBuffer<SGM::DispType> hostDisparity(potHostDispBuf.releaseValue());

  // set host ptr in a cv::Mat container
  const std::vector<int> sizes{(int)dst.height(), (int)dst.width(), (int)GPU::numSphereSweeps()};
  cv::Mat costVolume((int)sizes.size(), sizes.data(), CV_16UC1, hostCostVolume.hostPtr());
  cv::Mat disparity((int)dst.height(), (int)dst.width(), CV_16S, hostDisparity.hostPtr());
  disparity = cv::Scalar::all(SGM::COST_VOLUME_INIT_VALUE);
  cv::Mat dummy;  // needed by SGM

  // load input image from device
  cv::Mat inputMat;
  FAIL_RETURN(loadRGBAFromSurface(inputMat, *inputSurfaces.find(sourceID)->second->pimpl->surface));

  const int P1 = 50;
  const float P2Alpha = 0.25f / 3;
  const int P2Gamma = 200;
  const int P2Min = 50;
  SGM::aggregateDisparityVolumeWithAdaptiveP2SGM<cv::Vec<uchar, 4>>(
      inputMat, costVolume, Rect::fromInclusiveTopLeftBottomRight(0, 0, dst.height() - 1, dst.width() - 1), disparity,
      dummy, 0, GPU::numSphereSweeps(), P1, P2Alpha, P2Gamma, P2Min, uniquenessRatio, true, SGM::SGMmode::SGM_8DIRS);
  Logger::get(Logger::Info) << "SGM aggregation done" << std::endl;

  // Median filter the ouput disparity
  cv::Mat disparityCloned = disparity.clone();
  cv::medianBlur(disparityCloned, disparity, 3);
  Logger::get(Logger::Info) << "Median filter done" << std::endl;

  // use SGM disparity
  FAIL_RETURN(GPU::sphereSweepInputDisparityToDepth(sourceID, frame, dst, hostDisparity, true, inputSurfaces, panoDef,
                                                    stream, scale));
  Logger::get(Logger::Info) << "Disparity to depth done" << std::endl;

  FAIL_RETURN(hostCostVolume.release());
  FAIL_RETURN(hostDisparity.release());

  return Status();
}

Status sphereSweepInputMultiScale(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                                  const std::vector<InputPyramid>& inputPyramids, const DepthPyramid& depthPyramid,
                                  const Core::PanoDefinition& panoDef, const Core::DepthDefinition& depthDef,
                                  const SGMPostProcessing sgmPostProcessing,
                                  const BilateralFilterPostProcessing bilateralFilterPostProcessing,
                                  GPU::Stream& stream) {
  if (panoDef.numVideoInputs() > 6) {
    return Status{Origin::Stitcher, ErrType::ImplementationError,
                  "Sphere sweep only implemented for 6 inputs maximum (hardcoded)"};
  }

  if (depthDef.getNumPyramidLevels() != (int)depthPyramid.numLevels()) {
    return Status{Origin::Stitcher, ErrType::ImplementationError, "Depth Pyramid has wrong number of levels"};
  }

  const size_t numLevels = depthDef.getNumPyramidLevels();

  // TODO don't allocate 0-level
  GPU::Surface* depthSurf = numLevels == 1 ? &dst : &depthPyramid.getLevel(numLevels - 1).gpuSurf();
  ;

  for (int level = (int)numLevels - 1; level >= 0; level--) {
    std::map<videoreaderid_t, Core::SourceSurface*> surfaceMap;
    for (videoreaderid_t i = 0; i < panoDef.numVideoInputs(); i++) {
      surfaceMap[i] = inputPyramids[i].getLevel(level).surf();
    }

    // TODO don't allocate 0-level
    GPU::Surface* nextDepth = level == 0 ? &dst : &depthPyramid.getLevel(level).gpuSurf();

    GPU::Surface* filteredDepth = nullptr;
    std::unique_ptr<Core::SourceSurface> bilateralFilterTemporarySurface;

    if (bilateralFilterPostProcessing == BilateralFilterPostProcessing::On) {
      // temporary buffer to avoid a copy
      Potential<SourceSurface> potSurface = OffscreenAllocator::createDepthSurface(
          nextDepth->width(), nextDepth->height(), "Depth surface post-processing");
      FAIL_RETURN(potSurface.status());
      // hold the
      bilateralFilterTemporarySurface.reset(potSurface.release());

      // store nextDepth in filteredDepth for the filter output and have nextDepth reference the temporary buffer for
      // the sphere sweeps
      filteredDepth = nextDepth;
      nextDepth = bilateralFilterTemporarySurface->pimpl->surface;
    }

    if (level == (int)numLevels - 1) {
      if (sgmPostProcessing == SGMPostProcessing::On) {
        // SGM on coarsest level
        // TODO use a non-zero uniquenessRatio, and fill in the holes - currently using 0 to get a dense depth map with
        // less confidence
        FAIL_RETURN(sphereSweepInputSGMSingleScale(sourceID, frame, *nextDepth, surfaceMap, panoDef, stream,
                                                   (float)depthPyramid.getLevel(level).scale(),
                                                   0 /* uniquenessRatio */));
      } else {
        // No SGM on coarsest level
        FAIL_RETURN(GPU::sphereSweepInput(sourceID, frame, *nextDepth, surfaceMap, panoDef, stream,
                                          (float)depthPyramid.getLevel(level).scale()));
      }
    } else {
      // other finer levels
      FAIL_RETURN(GPU::sphereSweepInputStep(sourceID, frame, *nextDepth, *depthSurf, surfaceMap, panoDef, stream,
                                            (float)depthPyramid.getLevel(level).scale()));
    }

    if (bilateralFilterPostProcessing == BilateralFilterPostProcessing::On) {
      FAIL_RETURN(GPU::depthJointBilateralFilter(*filteredDepth, *nextDepth,
                                                 *inputPyramids[sourceID].getLevel(level).surf(), stream));
#ifdef DUMP_DEPTH_IMAGE
      {
        stream.synchronize();
        std::string filename = "/tmp/depth-raw-" + std::to_string(sourceID) + "-" + std::to_string(level) + ".png";
        Debug::dumpDepthSurface(filename.c_str(), *nextDepth, depthPyramid.getLevel(level).width(),
                                depthPyramid.getLevel(level).height());
      }
#endif
      // restore nextDepth
      std::swap(nextDepth, filteredDepth);
    }
#ifdef DUMP_DEPTH_IMAGE
    {
      stream.synchronize();
      std::string filename = "/tmp/depth-" + std::to_string(sourceID) + "-" + std::to_string(level) + ".png";
      Debug::dumpDepthSurface(filename.c_str(), *nextDepth, depthPyramid.getLevel(level).width(),
                              depthPyramid.getLevel(level).height());
    }
#endif

#ifdef DUMP_IMAGE_PYRAMID
    {
      stream.synchronize();
      std::string filename = "/tmp/pyramid-" + std::to_string(sourceID) + "-" + std::to_string(level) + ".png";
      Debug::dumpRGBATexture(filename.c_str(), *surfaceMap[0]->pimpl->surface, inputPyramids[0].getLevel(level).width(),
                             inputPyramids[0].getLevel(level).height());
    }
#endif

    depthSurf = nextDepth;
  }

  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
