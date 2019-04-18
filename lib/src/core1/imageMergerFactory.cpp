// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/imageMergerFactory.hpp"

#include "core1/arrayImageMerger.hpp"
#include "core1/checkerboardImageMerger.hpp"
#include "core1/diffImageMerger.hpp"
#include "core1/exposureDiffImageMerger.hpp"
#include "core1/gradientImageMerger.hpp"
#include "core1/laplacianImageMerger.hpp"
#include "core1/stackImageMerger.hpp"
#include "core1/noblendImageMerger.hpp"

#include "coredepth/sphereSweepMerger.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

#include <cassert>
#include <mutex>
#include <iostream>

namespace VideoStitch {
namespace Core {

const std::vector<std::string>& ImageMergerFactory::availableMergers() {
  static std::vector<std::string> availableMergers;
  // Lazily fill in the list of mergers. TODO: use a better, macro-based, registration pattern.
  static std::mutex mutex;
  {
    std::unique_lock<std::mutex> lock(mutex);
    if (availableMergers.empty()) {
#ifndef VS_OPENCL
      availableMergers.push_back("laplacian");
#endif
      availableMergers.push_back("gradient");
      availableMergers.push_back("noblendv1");
      availableMergers.push_back("diff");
      availableMergers.push_back("exposure_diff");
      availableMergers.push_back("checkerboard");
      availableMergers.push_back("array");
      availableMergers.push_back("stack");
      availableMergers.push_back("sphere_sweep");
    }
  }
  return availableMergers;
}

Potential<ImageMergerFactory> ImageMergerFactory::createMergerFactory(const Ptv::Value& value) {
  // Make sure value is an object.
  if (!Parse::checkType("ImageMergerFactory", value, Ptv::Value::OBJECT)) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'ImageMergerFactory' configuration, expected object"};
  }
  std::string type;
  if (Parse::populateString("ImageMergerFactory", value, "type", type, false) == Parse::PopulateResult_WrongType) {
    return {Origin::Stitcher, ErrType::InvalidConfiguration,
            "Invalid type for 'ImageMergerFactory' configuration, expected string"};
  }

  if (type == "laplacian") {
    return LaplacianImageMerger::Factory::parse(value);
  } else if (type == "gradient") {
    return GradientImageMerger::Factory::parse(value);
  } else if (type == "noblendv1") {
    return Potential<ImageMergerFactory>(new NoBlendImageMerger::Factory());
  } else if (type == "diff") {
    return Potential<ImageMergerFactory>(new DiffImageMerger::Factory());
  } else if (type == "exposure_diff") {
    return Potential<ImageMergerFactory>(new ExposureDiffImageMerger::Factory());
  } else if (type == "checkerboard") {
    return CheckerboardImageMerger::Factory::parse(value);
  } else if (type == "array") {
    return Potential<ImageMergerFactory>(new ArrayImageMerger::Factory());
  } else if (type == "stack") {
    return Potential<ImageMergerFactory>(new StackImageMerger::Factory());
  } else if (type == "sphere_sweep") {
    return Potential<ImageMergerFactory>(new SphereSweepMerger::Factory());
  } else {
    return {Origin::Stitcher, ErrType::InvalidConfiguration, "Unknown merger type: '" + type + "'"};
  }
}

Potential<PanoMerger> ImageMergerFactory::createDepth(const PanoDefinition& /* pano */) const {
  return {Origin::Stitcher, ErrType::ImplementationError, "ImageMergerFactory does not implement pano merging."};
};

bool ImageMergerFactory::equal(const ImageMergerFactory& other) const { return hash() == other.hash(); }

uint32_t ImageMergerFactory::getBlockAlignment() const { return ImageMerger::CudaBlockSize; }

namespace {
/**
 * A merger factory that cannot instanciate a merger. This can be useful when creating a merger without stitchers.
 */
class ImpotentMergerFactory : public ImageMergerFactory {
 public:
  virtual Potential<ImageMerger> create(const PanoDefinition& /*pano*/, ImageMapping& /*fromIm*/,
                                        const ImageMerger* /*to*/, bool) const {
    return Status{Origin::Stitcher, ErrType::ImplementationError, "Cannot create impotent merger"};
  }
  virtual ~ImpotentMergerFactory() {}
  Ptv::Value* serialize() const { return NULL; }

  virtual CoreVersion version() const { return Impotent; }
  virtual ImageMergerFactory* clone() const { return new ImpotentMergerFactory(); }
  virtual std::string hash() const { return "impotentMerger"; }
};
}  // namespace

Potential<ImageMergerFactory> ImageMergerFactory::newImpotentMergerFactory() {
  return Potential<ImageMergerFactory>(new ImpotentMergerFactory);
}

}  // namespace Core
}  // namespace VideoStitch
