// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "panoMerger.hpp"

#include "core1/imageMerger.hpp"

#include "core/surfacePyramid.hpp"

#include "parse/json.hpp"

#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/panoDef.hpp"

namespace VideoStitch {
namespace Core {

class SphereSweepMerger : public PanoMerger {
 public:
  class Factory : public ImageMergerFactory {
   public:
    static Potential<ImageMergerFactory> parse(const Ptv::Value&) { return new Factory(); }
    Potential<ImageMerger> create(const PanoDefinition&, ImageMapping&, const ImageMerger*, bool) const override {
      return {Origin::Stitcher, ErrType::ImplementationError, "Can't create sphere sweep ImageMerger"};
    }

    Potential<PanoMerger> createDepth(const PanoDefinition& pano) const override {
      return new SphereSweepMerger(pano);
    };

    virtual ~Factory() {}

    Ptv::Value* serialize() const override {
      Ptv::Value* res = Ptv::Value::emptyObject();
      res->push("type", new Parse::JsonValue("sphere_sweep"));
      return res;
    }

    CoreVersion version() const override { return Depth; }
    ImageMergerFactory* clone() const override { return new Factory(); }
    std::string hash() const override { return "depth_sphere_sweep"; }
  };

  /**
   * Creates a sphere sweep merger
   */
  explicit SphereSweepMerger(const PanoDefinition& pano);

  virtual ~SphereSweepMerger() {}

  Status computeAsync(const PanoDefinition& panoDef, PanoSurface& pano,
                      const std::map<videoreaderid_t, Core::SourceSurface*>& surfaces, GPU::Stream stream) override;

 private:
  std::vector<InputPyramid> pyramids;
  DepthPyramid depthPyramid;
};

}  // namespace Core
}  // namespace VideoStitch
