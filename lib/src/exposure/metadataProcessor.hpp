// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "input/inputFrame.hpp"

#include "libvideostitch/panoDef.hpp"

namespace VideoStitch {
namespace Exposure {

class MetadataProcessor {
 public:
  // may return a nullptr if it fails or the pano doesn't need an update
  std::unique_ptr<Core::PanoDefinition> createUpdatedPano(const Input::MetadataChunk& metadata,
                                                          const Core::PanoDefinition& currentPano, FrameRate frameRate,
                                                          frameid_t currentStitchingFrame);

 private:
  // Update tone curves in potentialNewPano to stitch currentStitchingFrame
  // If potentialNewPano does not exist yet, but current Pano needs updated,
  // it is created by cloning currentPano.
  void createUpdatedPanoForCurrentFrame(std::unique_ptr<Core::PanoDefinition>& potentialNewPano,
                                        const Core::PanoDefinition& currentPano, FrameRate frameRate,
                                        frameid_t currentStitchingFrame) const;

  void pruneToneCurves(frameid_t currentStitchingFrame, FrameRate frameRate);
  void insertToneCurveMetadata(std::vector<std::map<videoreaderid_t, Metadata::ToneCurve>> newData);

  using ToneCurveByTime = std::map<mtime_t, Metadata::ToneCurve>;
  std::map<videoreaderid_t, ToneCurveByTime> toneCurves;
};

}  // namespace Exposure
}  // namespace VideoStitch
