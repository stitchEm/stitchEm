// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/panoStitcherBase.hpp"
#include "libvideostitch/inputDef.hpp"

namespace VideoStitch {

namespace Core {

class SourceSurface;
class PanoMerger;

/**
 * @brief Implementation of DepthStitcher.
 */
template <typename Output>
class DepthStitcher : public PanoStitcherImplBase<Output> {
 public:
  virtual ~DepthStitcher();

  /**
   * The DepthStitcher is invalid until it has been setup().
   * pano must live until the DepthStitcher is destroyed.
   */
  DepthStitcher(const std::string& name, const PanoDefinition& pano, Eye eye);

 private:
  Status redoSetupImpl(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                       const ImageFlowFactory& flowFactory, const std::map<readerid_t, Input::VideoReader*>& readers,
                       const StereoRigDefinition*) override;
  Status setupImpl(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                   const ImageFlowFactory& flowFactory, const std::map<readerid_t, Input::VideoReader*>& readers,
                   const StereoRigDefinition*) override;

  Status merge(frameid_t frame, const std::map<readerid_t, Input::PotentialFrame>& inputFrames,
               const std::map<readerid_t, Input::VideoReader*>& readers,
               const std::map<readerid_t, PreProcessor*>& preprocessors, PanoSurface& pano) override;

  ChangeCompatibility getCompatibility(const PanoDefinition& pano, const PanoDefinition& newPano) const override;

  Status setupTexArrayAsync(videoreaderid_t inputID, frameid_t frame, const Input::PotentialFrame& inputFrame,
                            const InputDefinition& inputDef, GPU::Stream& stream, Input::VideoReader* reader,
                            const PreProcessor* preprocessor);

  std::vector<GPU::Buffer<unsigned char>> devUnpackTmps;
  std::map<videoreaderid_t, Core::SourceSurface*> surfaces;

  PanoMerger* panoMerger;

  using PanoStitcherImplBase<Output>::getPano;
};
}  // namespace Core
}  // namespace VideoStitch
