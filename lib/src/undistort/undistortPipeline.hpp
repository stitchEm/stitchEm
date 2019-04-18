// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "undistort/inputPipeline.hpp"

#include "libvideostitch/overrideDef.hpp"

namespace VideoStitch {
namespace Core {

/**
 * Pipeline to undistort the inputs
 */
class UndistortPipeline : public InputPipeline {
 public:
  virtual ~UndistortPipeline();

  static Potential<UndistortPipeline> createUndistortPipeline(const std::vector<Input::VideoReader*>& inputs,
                                                              const PanoDefinition& pano,
                                                              const OverrideOutputDefinition& overrideDef);

 private:
  UndistortPipeline(const std::vector<Input::VideoReader*>&, const PanoDefinition& pano,
                    const OverrideOutputDefinition& overrideDef);

  Status preprocessGroup(const std::map<videoreaderid_t, SourceSurface*>& /* src */,
                         GPU::Stream& /* stream */) override {
    return Status::OK();
  }

  virtual Status processInput(videoreaderid_t sourceID, int frame, GPU::Surface& dst,
                              const std::map<videoreaderid_t, SourceSurface*>& src, const InputDefinition& inputDef,
                              GPU::Stream& stream) const final override;

  std::vector<Transform*> transforms;
  const OverrideOutputDefinition overrideDef;
};

}  // namespace Core
}  // namespace VideoStitch
