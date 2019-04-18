// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoPipeline.hpp"

#include "buffer.hpp"

namespace VideoStitch {
namespace Core {

PanoPipeline::PanoPipeline(PanoStitcherImplBase<StitchOutput>* stitcher,
                           const std::vector<Input::VideoReader*>& readers, const std::vector<PreProcessor*>& preprocs,
                           PostProcessor* postproc)
    : VideoPipeline(readers, preprocs, postproc), stitcher(stitcher) {}

PanoPipeline::~PanoPipeline() { delete stitcher; }

Potential<PanoPipeline> PanoPipeline::createPanoPipeline(PanoStitcherImplBase<StitchOutput>* stitcher,
                                                         const std::vector<Input::VideoReader*>& readers,
                                                         const std::vector<PreProcessor*>& preprocs,
                                                         PostProcessor* postproc) {
  PanoPipeline* ret = new PanoPipeline(stitcher, readers, preprocs, postproc);

  Status initStatus = ret->init();

  if (!initStatus.ok()) {
    delete ret;
    ret = nullptr;
    return initStatus;
  }

  return ret;
}

Status PanoPipeline::stitch(mtime_t date, FrameRate frameRate,
                            std::map<readerid_t, Input::PotentialFrame>& inputBuffers, Output* output) {
  std::vector<ExtractOutput*> ext;
  return stitchAndExtract(date, frameRate, inputBuffers, output, ext, nullptr);
}

Status PanoPipeline::stitchAndExtract(mtime_t date, FrameRate frameRate,
                                      std::map<readerid_t, Input::PotentialFrame>& inputBuffers, Output* output,
                                      std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo) {
  FAIL_RETURN(extract(date, frameRate, inputBuffers, extracts, algo));

  return stitcher->stitch(date, frameRate.timestampToFrame(date), postproc, inputBuffers, readers, preprocs, output);
}

Status PanoPipeline::setup(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                           const ImageFlowFactory& flowFactory, const StereoRigDefinition*) {
  return stitcher->setup(mergerFactory, warperFactory, flowFactory, readers);
}

}  // namespace Core
}  // namespace VideoStitch
