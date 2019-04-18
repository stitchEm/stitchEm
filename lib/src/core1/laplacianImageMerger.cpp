// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "laplacianImageMerger.hpp"

#include "imageMapping.hpp"
#include "inputsMap.hpp"
#include "inputsMapCubemap.hpp"

#include "core/pyramid.hpp"
#include "gpu/image/sampling.hpp"
#include "gpu/image/blur.hpp"
#include "gpu/image/imgExtract.hpp"
#include "gpu/image/imgInsert.hpp"
#include "gpu/core1/voronoi.hpp"
#include "gpu/memcpy.hpp"
#include "parse/json.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/logging.hpp"

#include <cassert>
#include <sstream>

//#define DEBUGMASKS

#if defined(DEBUGMASKS)
#include <sstream>
#include "util/pnm.hpp"
#include "util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Core {

LaplacianImageMerger::Factory::Factory(int feather, int levels, int64_t baseSize, int gaussianRadius, int filterPasses,
                                       MaskMerger::MaskMergerType maskMergerType)
    : feather(feather),
      levels(levels),
      baseSize(baseSize),
      gaussianRadius(gaussianRadius),
      filterPasses(filterPasses),
      maskMergerType(maskMergerType) {}

ImageMergerFactory* LaplacianImageMerger::Factory::clone() const {
  return new Factory(feather, levels, baseSize, gaussianRadius, filterPasses, maskMergerType);
}

std::string LaplacianImageMerger::Factory::hash() const {
  std::stringstream ss;
  ss << "v1_LaplacianImageMerger" << feather << levels << " " << baseSize << " " << gaussianRadius << " "
     << filterPasses << " " << (int)maskMergerType;
  return ss.str();
}

uint32_t LaplacianImageMerger::Factory::getBlockAlignment() const {
  /* make sure all levels are multiple of 2 */
  return 32;
}

Potential<ImageMerger> LaplacianImageMerger::Factory::create(const PanoDefinition& pano, ImageMapping& fromIm,
                                                             const ImageMerger* to, bool progressive) const {
  LaplacianPyramid<uint32_t>* const* globalPyramids = nullptr;
  if (to == nullptr) {
    LaplacianPyramid<uint32_t>* globPyr[7];

    if (pano.getProjection() == PanoProjection::Cubemap || pano.getProjection() == PanoProjection::EquiangularCubemap) {
      for (int target = CUBE_MAP_POSITIVE_X; target <= CUBE_MAP_NEGATIVE_Z; ++target) {
        Potential<LaplacianPyramid<uint32_t>> pyr = LaplacianPyramid<uint32_t>::create(
            "global-" + toString((TextureTarget)target), pano.getLength(), pano.getLength(),
            computeNumLevels(pano.getLength(), pano.getLength()), LaplacianPyramid<uint32_t>::ExternalFirstLevel,
            progressive ? LaplacianPyramid<uint32_t>::Multiple : LaplacianPyramid<uint32_t>::SingleShot, gaussianRadius,
            filterPasses, false);
        FAIL_RETURN(pyr.status());
        globPyr[target] = pyr.release();
      }
      globPyr[EQUIRECTANGULAR] = nullptr;
    } else {
      Potential<LaplacianPyramid<uint32_t>> pyr = LaplacianPyramid<uint32_t>::create(
          "global-equirectangular", pano.getWidth(), pano.getHeight(),
          computeNumLevels(pano.getWidth(), pano.getHeight()), LaplacianPyramid<uint32_t>::ExternalFirstLevel,
          progressive ? LaplacianPyramid<uint32_t>::Multiple : LaplacianPyramid<uint32_t>::SingleShot, gaussianRadius,
          filterPasses, true);
      FAIL_RETURN(pyr.status());
      globPyr[EQUIRECTANGULAR] = pyr.release();

      for (int target = CUBE_MAP_POSITIVE_X; target <= CUBE_MAP_NEGATIVE_Z; ++target) {
        globPyr[target] = nullptr;
      }
    }

    globalPyramids = globPyr;
  } else {
    globalPyramids = static_cast<const LaplacianImageMerger*>(to)->globalPyramids;
  }

  return Potential<ImageMerger>(new LaplacianImageMerger(pano, fromIm, to, globalPyramids, feather, gaussianRadius,
                                                         filterPasses, maskMergerType));
}

Ptv::Value* LaplacianImageMerger::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue("laplacian"));
  res->push("feather", new Parse::JsonValue(feather));
  if (levels != -1) {
    res->push("levels", new Parse::JsonValue(levels));
  }
  res->push("base_size", new Parse::JsonValue(baseSize));
  res->push("gaussian_radius", new Parse::JsonValue(gaussianRadius));
  res->push("filter_passes", new Parse::JsonValue(filterPasses));
  res->push("mask_merger", new Parse::JsonValue((int)maskMergerType));
  return res;
}

Potential<ImageMergerFactory> LaplacianImageMerger::Factory::parse(const Ptv::Value& value) {
  int feather = DEFAULT_BLENDING_FEATHER;
  int levels = -1;
  int64_t baseSize = DEFAULT_BASE_LAPLACIAN_SIZE;
  int gaussianRadius = DEFAULT_LAPLACIAN_GAUSSIAN_RADIUS;
  int filterPasses = DEFAULT_LAPLACIAN_BLUR_PASSES;
#define POPULATE_INT_PROPAGATE_WRONGTYPE(config_name, varName)                                 \
  if (Parse::populateInt("LaplacianImageMergerFactory", value, config_name, varName, false) == \
      Parse::PopulateResult_WrongType) {                                                       \
    return {Origin::Stitcher, ErrType::InvalidConfiguration,                                   \
            "Invalid type for '" config_name "' in LaplacianMergerFactory, expected int"};     \
  }
  POPULATE_INT_PROPAGATE_WRONGTYPE("feather", feather);
  POPULATE_INT_PROPAGATE_WRONGTYPE("levels", levels);
  POPULATE_INT_PROPAGATE_WRONGTYPE("base_size", baseSize);
  POPULATE_INT_PROPAGATE_WRONGTYPE("gaussian_radius", gaussianRadius);
  POPULATE_INT_PROPAGATE_WRONGTYPE("filter_passes", filterPasses);
  MaskMerger::MaskMergerType maskMergerType;
  int maskType;
  switch (Parse::populateInt("GradientImageMergerFactory", value, "mask_merger", maskType, false)) {
    case Parse::PopulateResult_WrongType:
      return {Origin::Stitcher, ErrType::InvalidConfiguration,
              "Invalid type for 'mask_merger' configuration, expected int"};
    case Parse::PopulateResult_DoesNotExist:
      maskMergerType = MaskMerger::getDefaultMaskMerger();
      break;
    default:
      maskMergerType = (MaskMerger::MaskMergerType)maskType;
      break;
  }

#undef POPULATE_INT_PROPAGATE_WRONGTYPE

  feather = std::max(feather, 0);
  feather = std::min(feather, 100);

  // Backwards compatibility: prefer levels if specified.
  if (levels >= 1) {
    baseSize = -1;
  } else if (baseSize < 1) {
    Logger::get(Logger::Error) << "LaplacianImageMergerFactory: base_size < 1 makes no sense, setting to 1"
                               << std::endl;
    baseSize = 1;
  }
  if (filterPasses < 1) {
    Logger::get(Logger::Error) << "LaplacianImageMergerFactory: filter_passes < 1 makes no sense, setting to 1"
                               << std::endl;
    filterPasses = 1;
  } else if (filterPasses > 5) {
    Logger::get(Logger::Error) << "LaplacianImageMergerFactory: filter_passes > 5 makes no sense, setting to 5"
                               << std::endl;
    filterPasses = 5;
  }

  return Potential<ImageMergerFactory>(
      new LaplacianImageMerger::Factory(feather, levels, baseSize, gaussianRadius, filterPasses, maskMergerType));
}

int LaplacianImageMerger::Factory::computeNumLevels(int64_t width, int64_t height) const {
  // Backwards compatibility: If we have a number of levels, just use that;
  if (levels > 0) {
    return levels > 4 ? 4 : levels;
  }
  // We're trying to get pano output's base level to be as mall as possible while larger than baseSize.
  // Note that we're not making sure that this is not too small for inputs.
  int numLevels = 0;
  while (width > baseSize && height > baseSize) {
    ++numLevels;
    width = (width + 1) / 2;
    height = (height + 1) / 2;
  }
  return numLevels > 4 ? 4 : numLevels;
}

LaplacianImageMerger::LaplacianImageMerger(const PanoDefinition& pano, ImageMapping& fromIm, const ImageMerger* to,
                                           LaplacianPyramid<uint32_t>* const* pyr, int feather, int gaussianRadius,
                                           int filterPasses, MaskMerger::MaskMergerType maskMergerType)
    : ImageMerger(fromIm.getImId(), to),
      gaussianRadius(gaussianRadius),
      filterPasses(filterPasses),
      width(pano.getWidth()),
      height(pano.getHeight()) {
  maskMerger.reset(MaskMerger::factor(maskMergerType));
  maskMerger->setParameters(std::vector<double>{(double)(feather)});

  for (int i = EQUIRECTANGULAR; i <= CUBE_MAP_NEGATIVE_Z; ++i) {
    globalPyramids[i] = pyr[i];
    pyramids[i] = nullptr;
  }
}

LaplacianImageMerger::~LaplacianImageMerger() {
  for (int i = EQUIRECTANGULAR; i <= CUBE_MAP_NEGATIVE_Z; ++i) {
    if (!to) {
      // The first merger is the owner of the global pyramids.
      delete globalPyramids[i];
    }
  }
}

Status LaplacianImageMerger::prepareMergeAsync(TextureTarget t, const ImageMapping& fromIm, GPU::Stream stream) const {
  if (!fromIm.getOutputRect(t).empty()) {
    pyramids[t]->start(GPU::Buffer<uint32_t>(), GPU::Buffer<uint32_t>(), stream);
    PROPAGATE_FAILURE_STATUS(pyramids[t]->compute(fromIm.getDeviceOutputBuffer(t), stream));
  }
  return Status::OK();
}

Status LaplacianImageMerger::mergeAsync(TextureTarget t, const PanoDefinition& /*pano*/,
                                        GPU::Buffer<uint32_t> panoDevOut, GPU::UniqueBuffer<uint32_t>& progressivePbo,
                                        const ImageMapping& fromIm, bool isFirstMerger, GPU::Stream stream) const {
  // The first merger must wipe the global pyramid before inserting into it.
  if (isFirstMerger) {
    // allocate on demand
    if (!progressivePbo.borrow().wasAllocated()) {
      FAIL_RETURN(progressivePbo.alloc(width * height, "Progressive Pixel Buffer Object"));
    }
    globalPyramids[t]->start(panoDevOut, progressivePbo.borrow(), stream);
  }

  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }

  // Now merge the pyramids:
  int64_t offsetX = fromIm.getOutputRect(t).left();
  int64_t offsetY = fromIm.getOutputRect(t).top();

  for (int i = 0; i <= globalPyramids[t]->numLevels(); ++i) {
    LaplacianPyramid<uint32_t>::LevelSpec<uint32_t>& globalLevel = globalPyramids[t]->getLevel(i);
    const LaplacianPyramid<uint32_t>::LevelSpec<uint32_t>& level = pyramids[t]->getLevel(i);
    GPU::Buffer<const unsigned char> curMask = (maskMerger->getAlphaPyramid(t))
                                                   ? maskMerger->getAlphaPyramid(t)->getLevel(i).data()
                                                   : GPU::Buffer<unsigned char>();

    if (i == globalPyramids[t]->numLevels()) {
      PROPAGATE_FAILURE_STATUS(Image::imgInsertInto(globalLevel.data(), globalLevel.width(), globalLevel.height(),
                                                    level.data(), level.width(), level.height(), offsetX, offsetY,
                                                    curMask, t == EQUIRECTANGULAR && fromIm.wraps(),
                                                    false,  // never wrap vertically
                                                    stream));
    } else {
      PROPAGATE_FAILURE_STATUS(Image::imgInsertInto10bit(globalLevel.data(), globalLevel.width(), globalLevel.height(),
                                                         level.data(), level.width(), level.height(), offsetX, offsetY,
                                                         curMask, t == EQUIRECTANGULAR && fromIm.wraps(),
                                                         false,  // never wrap vertically
                                                         stream));
    }

    // make sure that offsets are always even, so that subsampling/upsampling is consistent
    // Illustration in dimension 1: insert a image of size 8 at offsetX==3 into a larger image.
    //  level 0:   | | | |X|X|X| | |       offsetX=3   | | | | | | |X|X|X| | | | | | | | | | | |
    //  level 1:   |   |XXX|XXX|   |       offsetX=1   |   |   |XXX|XXX|   |   |   |   |   |   |
    //  level 2:   |XXXXXXX|XXXXXXX|       offsetX=0   |XXXXXXX|XXXXXXX|       |       |       |
    //
    //  => everything is shifted.
    assert((offsetX & 1) == 0);
    assert((offsetY & 1) == 0);
    offsetX /= 2;
    offsetY /= 2;
  }
  return Status::OK();
}

Status LaplacianImageMerger::reconstruct(TextureTarget t, const PanoDefinition&, GPU::Buffer<uint32_t>,
                                         bool progressive, GPU::Stream stream) const {
  return globalPyramids[t]->collapse(!progressive, stream);
}

bool LaplacianImageMerger::isMultiScale() const { return true; }

// --------------------- Configuration

Status LaplacianImageMerger::setup(const PanoDefinition& pano, InputsMap& inputsMap, const ImageMapping& fromIm,
                                   GPU::Stream stream) {
  if (fromIm.getOutputRect(EQUIRECTANGULAR).empty()) {
    return Status::OK();
  }

  Potential<LaplacianPyramid<uint32_t>> fStatus = LaplacianPyramid<uint32_t>::create(
      "local-equirectangular-" + std::to_string(fromIm.getImId()), fromIm.getOutputRect(EQUIRECTANGULAR).getWidth(),
      fromIm.getOutputRect(EQUIRECTANGULAR).getHeight(), globalPyramids[EQUIRECTANGULAR]->numLevels(),
      LaplacianPyramid<uint32_t>::InternalFirstLevel, LaplacianPyramid<uint32_t>::SingleShot, gaussianRadius,
      filterPasses,
      // We only need wrapping computations when the image fills the whole pano. Else wrapping extraction takes care of
      // the wrapping.
      fromIm.getOutputRect(EQUIRECTANGULAR).getWidth() >= pano.getWidth());

  if (!fStatus.ok()) {
    return {Origin::Stitcher, ErrType::SetupFailure, "Could not set up laplacian image merger", fStatus.status()};
  }
  pyramids[EQUIRECTANGULAR].reset(fStatus.release());

  if (!to) {
    // is first merger, nothing to do
    return stream.synchronize();
  }

  FAIL_RETURN(maskMerger->setupMask(pano, inputsMap.getMask(), fromIm, to, stream));

  // Construct gaussian pyramid mask
  FAIL_RETURN(maskMerger->buildPyramidMask(
      fromIm, std::to_string(fromIm.getImId()), globalPyramids[EQUIRECTANGULAR]->numLevels(),
      // NOTE: The filter size of the mask and the image should be different or else, the merging result would be
      // similar to linear blending
      DEFAULT_GAUSSIAN_BLUR_RADIUS, filterPasses,
      // We only need wrapping computations when the image fills the whole pano. Else wrapping extraction takes care of
      // the wrapping.
      fromIm.getOutputRect(EQUIRECTANGULAR).getWidth() >= pano.getWidth(), stream));

  FAIL_RETURN(stream.synchronize());
  return Status::OK();
};

Status LaplacianImageMerger::setupCubemap(const PanoDefinition& pano, InputsMap& inputsMap, const ImageMapping& fromIm,
                                          GPU::Stream stream) {
  for (int t = CUBE_MAP_POSITIVE_X; t <= CUBE_MAP_NEGATIVE_Z; ++t) {
    TextureTarget target = (TextureTarget)t;
    if (fromIm.getOutputRect(target).empty()) {
      continue;
    }

    Potential<LaplacianPyramid<uint32_t>> fStatus = LaplacianPyramid<uint32_t>::create(
        "local-" + toString(target) + "-" + std::to_string(fromIm.getImId()), fromIm.getOutputRect(target).getWidth(),
        fromIm.getOutputRect(target).getHeight(), globalPyramids[target]->numLevels(),
        LaplacianPyramid<uint32_t>::InternalFirstLevel, LaplacianPyramid<uint32_t>::SingleShot, gaussianRadius,
        filterPasses, false);

    if (!fStatus.ok()) {
      return {Origin::Stitcher, ErrType::SetupFailure, "Could not set up laplacian image merger", fStatus.status()};
    }
    pyramids[target].reset(fStatus.release());
  }

  if (!to) {
    // is first merger, nothing to do
    return stream.synchronize();
  }

  FAIL_RETURN(maskMerger->setupMaskCubemap(pano, inputsMap.getMask(), fromIm, to, stream));

  // Construct gaussian pyramid mask
  // NB
  // It's important to ensure alpha layer continuity to first make
  // an equirectangular pyramid, then reproject each layer, instead of
  // first projecting the alpha layer then constructing a gaussian pyramid
  // for each face.
  // The reason is the convolution kernel code is not aware of the adjacency
  // between the cubemap's faces, creating a potential discontinuity at each faces border.
  FAIL_RETURN(maskMerger->buildPyramidMaskCubemap(pano, fromIm, std::to_string(fromIm.getImId()),
                                                  globalPyramids[CUBE_MAP_POSITIVE_X]->numLevels(),
                                                  // NOTE: The filter size of the mask and the image should be different
                                                  // or else, the merging result would be similar to linear blending
                                                  DEFAULT_GAUSSIAN_BLUR_RADIUS, filterPasses, false, stream));

  FAIL_RETURN(stream.synchronize());

  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
