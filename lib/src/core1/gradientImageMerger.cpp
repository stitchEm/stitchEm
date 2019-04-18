// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gradientImageMerger.hpp"

#include "maskMerger.hpp"
#include "imageMapping.hpp"
#include "inputsMap.hpp"
#include "inputsMapCubemap.hpp"

#include "parse/json.hpp"
#include "gpu/image/blur.hpp"
#include "gpu/image/imgExtract.hpp"
#include "gpu/image/imgInsert.hpp"
#include "gpu/buffer.hpp"
#include "gpu/core1/voronoi.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

#include <memory>
#include <sstream>

//#define DEBUGMASKS

#ifdef DEBUGMASKS
#include "util/pnm.hpp"
#include "util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace Core {
GradientImageMerger::Factory::Factory(int feather, MaskMerger::MaskMergerType maskMergerType)
    : feather(feather), maskMergerType(maskMergerType) {}

Potential<ImageMerger> GradientImageMerger::Factory::create(const PanoDefinition& pano, ImageMapping& fromIm,
                                                            const ImageMerger* to, bool) const {
  return Potential<ImageMerger>(new GradientImageMerger(pano, fromIm, to, feather, maskMergerType));
}

ImageMergerFactory* GradientImageMerger::Factory::clone() const { return new Factory(feather, maskMergerType); }

std::string GradientImageMerger::Factory::hash() const {
  std::stringstream ss;
  ss << "v1_GradientImageMerger" << feather << " mask_merger" << (int)maskMergerType;
  return ss.str();
}

Ptv::Value* GradientImageMerger::Factory::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("type", new Parse::JsonValue("gradient"));
  res->push("feather", new Parse::JsonValue(feather));
  res->push("mask_merger", new Parse::JsonValue((int)maskMergerType));
  return res;
}

Potential<ImageMergerFactory> GradientImageMerger::Factory::parse(const Ptv::Value& value) {
  int feather = DEFAULT_BLENDING_FEATHER;
  if (Parse::populateInt("GradientImageMergerFactory", value, "feather", feather, false) ==
      Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'feather' configuration, expected integer"};
  }

  MaskMerger::MaskMergerType maskMergerType;
  int maskType;
  switch (Parse::populateInt("GradientImageMergerFactory", value, "mask_merger", maskType, false)) {
    case Parse::PopulateResult_WrongType:
      return {Origin::Stitcher, ErrType::InvalidConfiguration,
              "Invalid type for 'mask_merger' configuration, expected integer"};
    case Parse::PopulateResult_DoesNotExist:
      maskMergerType = MaskMerger::getDefaultMaskMerger();
      break;
    default:
      maskMergerType = (MaskMerger::MaskMergerType)maskType;
      break;
  }

  feather = std::max(feather, 0);
  feather = std::min(feather, 100);
  return Potential<ImageMergerFactory>(new GradientImageMerger::Factory(feather, maskMergerType));
}

Status GradientImageMerger::mergeAsync(TextureTarget t, const PanoDefinition& pano, GPU::Buffer<uint32_t> pbo,
                                       GPU::UniqueBuffer<uint32_t>&, const ImageMapping& fromIm,
                                       bool /* isFirstMerger */, GPU::Stream stream) const {
  if (fromIm.getOutputRect(t).empty()) {
    return Status::OK();
  }
  // insert into pano image
  size_t width, height;
  if (t == EQUIRECTANGULAR) {
    width = pano.getWidth();
    height = pano.getHeight();
    if (maskMerger->getAlpha(t).wasAllocated() && (GradientImageMerger::warpMergeType() == Format::Gradient)) {
      /* merge was already done at the same time as warp */
      return Status::OK();
    }
  } else {
    width = pano.getLength();
    height = pano.getLength();
  }
  return Image::imgInsertInto(pbo, width, height, fromIm.getDeviceOutputBuffer(t), fromIm.getOutputRect(t).getWidth(),
                              fromIm.getOutputRect(t).getHeight(), fromIm.getOutputRect(t).left(),
                              fromIm.getOutputRect(t).top(), maskMerger->getAlpha(t),
                              fromIm.getOutputRect(t).right() >= (int64_t)pano.getLength(), false, stream);
}

GradientImageMerger::GradientImageMerger(const PanoDefinition& /*pano*/, ImageMapping& fromIm, const ImageMerger* to,
                                         int feather, MaskMerger::MaskMergerType maskMergerType)
    : ImageMerger(fromIm.getImId(), to) {
  maskMerger.reset(MaskMerger::factor(maskMergerType));
  maskMerger->setParameters(std::vector<double>{(double)(feather)});
}

GradientImageMerger::~GradientImageMerger() {}

Status GradientImageMerger::setup(const PanoDefinition& pano, InputsMap& inputsMap, const ImageMapping& fromIm,
                                  GPU::Stream stream) {
  SIMPLEPROFILE_MS("GradientImageMerger::setup() : Compute Mask");

  if (fromIm.getOutputRect(EQUIRECTANGULAR).empty()) {
    return Status::OK();
  }

  GPU::Buffer<uint32_t> inputsMask = inputsMap.getMask();
  FAIL_RETURN(maskMerger->setupMask(pano, inputsMask, fromIm, to, stream));
  // make sure kernels are done before destroying the tmp buffers
  FAIL_RETURN(stream.synchronize());

#ifdef DEBUGMASKS
  if (!fromIm.getOutputRect(EQUIRECTANGULAR).empty()) {
    std::string outputFile =
        "gradient-mask-" + std::to_string(fromIm.getImId()) + "-" + std::to_string(EQUIRECTANGULAR) + ".png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(outputFile, maskMerger->getAlpha(EQUIRECTANGULAR),
                                                     fromIm.getOutputRect(EQUIRECTANGULAR).getWidth(),
                                                     fromIm.getOutputRect(EQUIRECTANGULAR).getHeight());
  }
#endif

  return Status::OK();
}

Status GradientImageMerger::setupCubemap(const PanoDefinition& pano, InputsMap& inputsMap, const ImageMapping& fromIm,
                                         GPU::Stream stream) {
  SIMPLEPROFILE_MS("GradientImageMerger::setup() : Compute Mask");

  GPU::Buffer<uint32_t> inputsMask = inputsMap.getMask();
  FAIL_RETURN(maskMerger->setupMaskCubemap(pano, inputsMask, fromIm, to, stream));
  // make sure kernels are done before destroying the tmp buffers
  FAIL_RETURN(stream.synchronize());

#ifdef DEBUGMASKS
  if (!fromIm.getOutputRect(target).empty()) {
    std::string outputFile =
        "gradient-mask-" + std::to_string(fromIm.getImId()) + "-" + std::to_string(target) + ".png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(outputFile, maskMerger->getAlpha(target),
                                                     fromIm.getOutputRect(target).getWidth(),
                                                     fromIm.getOutputRect(target).getHeight());
  }
#endif

  return Status::OK();
}

}  // namespace Core
}  // namespace VideoStitch
