// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "voronoiMaskMerger.hpp"

#include "core/transformGeoParams.hpp"

#include "imageMerger.hpp"
#include "imageMapping.hpp"
#include "panoRemapper.hpp"

#include "gpu/buffer.hpp"
#include "gpu/image/blur.hpp"
#include "gpu/image/imgExtract.hpp"
#include "gpu/image/imgInsert.hpp"
#include "gpu/image/imageOps.hpp"
#include "gpu/core1/voronoi.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/logging.hpp"

//#define DEBUGALPHA

#ifdef DEBUGALPHA
#ifndef _MSC_VER
static const std::string DEBUG_FOLDER = "/tmp/voronoi/";
#else
static const std::string DEBUG_FOLDER = "";
#endif
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "util/debugUtils.hpp"
#include <sstream>
#endif

namespace VideoStitch {
namespace Core {

static const float MAX_GRADIENT_MASK_TRANSITION_POWER = 5.0f;

std::pair<float, float> VoronoiMaskMerger::transitionParameters(int feather) {
  double maxTransitionDistance = M_PI;
  double minTransitionDistance = 0.0001;

  auto featherSq = pow((float)feather / 100.f, 2.0);

  // be able to set the maximium transition to get rid of ghosting when you have huge overlaps
  // maximum transition size, independent of how much the images overlap
  double transitionDistance = minTransitionDistance + featherSq * double(maxTransitionDistance - minTransitionDistance);

  // from feather 100 to 0, go from the smoothest possible transition to a steep one
  float power = 2.0f + ((float)(100 - feather) / 100.f) * (MAX_GRADIENT_MASK_TRANSITION_POWER - 2.0f);

  // backwards compatibility: blend_radius does not translate cleanly to feather when > 0, so we just ignore it
  // blend_radius default value was -1, which is equivalent to feather 100, so most projects output won't change

  return {(float)transitionDistance, power};
}

#ifdef DEBUGALPHA
void dumpDebugIndexBuffer(TextureTarget t, GPU::Buffer<const uint32_t> buffer, const ImageMapping& fromIm,
                          std::string name) {
  const int64_t width = fromIm.getOutputRect(t).getWidth();
  const int64_t height = fromIm.getOutputRect(t).getHeight();

  std::stringstream ss;
  ss << DEBUG_FOLDER << name << "-" << fromIm.getImId() << "-" << t << ".png";
  Debug::dumpRGBAIndexDeviceBuffer(ss.str().c_str(), buffer, width, height);
}

#endif

Status VoronoiMaskMerger::setup(const PanoDefinition& pano, GPU::Buffer<const uint32_t> inputsMask,
                                const ImageMapping& fromIm, const ImageMerger* const nextMerger, GPU::Stream stream) {
  // Get pixel size from input image
  size_t fromImOutputSize = fromIm.getOutputRect(EQUIRECTANGULAR).getArea();
  auto devTmpBuffer = GPU::uniqueBuffer<uint32_t>(fromImOutputSize, "Voronoi Mask Merger");
  FAIL_RETURN(devTmpBuffer.status());
  auto devWorkBuffer1 = GPU::uniqueBuffer<uint32_t>(fromImOutputSize, "Voronoi Mask Merger");
  FAIL_RETURN(devWorkBuffer1.status());
  auto devWorkBuffer2 = GPU::uniqueBuffer<uint32_t>(fromImOutputSize, "Voronoi Mask Merger");
  FAIL_RETURN(devWorkBuffer2.status());

  FAIL_RETURN(alpha[EQUIRECTANGULAR].recreate(fromImOutputSize, "Voronoi Mask Merger"));

  // copy the relevant portion of the setup image in devTmpBuffer
  FAIL_RETURN(Image::imgExtractFrom(devTmpBuffer.borrow(), fromIm.getOutputRect(EQUIRECTANGULAR).getWidth(),
                                    fromIm.getOutputRect(EQUIRECTANGULAR).getHeight(), inputsMask, pano.getWidth(),
                                    pano.getHeight(), fromIm.getOutputRect(EQUIRECTANGULAR).left(),
                                    fromIm.getOutputRect(EQUIRECTANGULAR).top(), fromIm.wraps(), stream));
#ifdef DEBUGALPHA
  stream.synchronize();
  dumpDebugIndexBuffer(EQUIRECTANGULAR, devTmpBuffer.borrow_const(), fromIm, "Copied Mask");
#endif

  if (!nextMerger) {
    FAIL_RETURN(setInitialImageMask(alpha[EQUIRECTANGULAR].borrow(), devTmpBuffer.borrow(),
                                    fromIm.getOutputRect(EQUIRECTANGULAR).getWidth(),
                                    fromIm.getOutputRect(EQUIRECTANGULAR).getHeight(), 1 << fromIm.getImId(), stream));

  } else {
    // We only need wrapping computations when the image wraps and fills the whole pano. Else wrapping extraction takes
    // care of the wrapping.
    bool needsWrappingVoronoi = fromIm.getOutputRect(EQUIRECTANGULAR).getWidth() >= pano.getWidth();
    auto edtParams = transitionParameters(feather);

    PanoDimensions panoDim = getPanoDimensions(pano);
    PanoRegion region;
    region.panoDim = panoDim;
    region.viewLeft = (int32_t)fromIm.getOutputRect(EQUIRECTANGULAR).left();
    region.viewTop = (int32_t)fromIm.getOutputRect(EQUIRECTANGULAR).top();
    region.viewWidth = (int32_t)fromIm.getOutputRect(EQUIRECTANGULAR).getWidth();
    region.viewHeight = (int32_t)fromIm.getOutputRect(EQUIRECTANGULAR).getHeight();

    FAIL_RETURN(computeMask(alpha[EQUIRECTANGULAR].borrow(), devTmpBuffer.borrow(), devWorkBuffer1.borrow(),
                            devWorkBuffer2.borrow(), region, 1 << fromIm.getImId(), nextMerger->getIdMask(),
                            needsWrappingVoronoi, edtParams.first, edtParams.second, stream));
  }

#ifdef DEBUGALPHA
  {
    stream.synchronize();
    std::stringstream ss;
    ss << DEBUG_FOLDER << "Voronoi Mask-" << fromIm.getImId() << "-equirectangular.png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str(), alpha[EQUIRECTANGULAR].borrow(),
                                                     fromIm.getOutputRect(EQUIRECTANGULAR).getWidth(),
                                                     fromIm.getOutputRect(EQUIRECTANGULAR).getHeight());
  }
#endif

  return stream.synchronize();
}

Status VoronoiMaskMerger::updateAsync() { return Status::OK(); }

Status VoronoiMaskMerger::setParameters(const std::vector<double>& params) {
  if (params.size() != 1) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration, "Unexpected number of arguments in voronoi mask merger"};
  }
  feather = (int)params[0];
  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
