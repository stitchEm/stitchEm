// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "stereoPipeline.hpp"

#include "buffer.hpp"

#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/imageFlowFactory.hpp"
#include "libvideostitch/imageWarperFactory.hpp"

#include <iostream>
#include <future>

namespace VideoStitch {
namespace Core {
StereoPipeline::StereoPipeline(PanoStitcherImplBase<StereoOutput>* left, PanoStitcherImplBase<StereoOutput>* right,
                               const std::vector<Input::VideoReader*>& readers,
                               const std::vector<PreProcessor*>& preprocs, PostProcessor* postproc)
    : VideoPipeline(readers, preprocs, postproc), leftStitcher(left), rightStitcher(right) {}

StereoPipeline::~StereoPipeline() {
  delete leftStitcher;
  delete rightStitcher;
}

Potential<StereoPipeline> StereoPipeline::createStereoPipeline(PanoStitcherImplBase<StereoOutput>* left,
                                                               PanoStitcherImplBase<StereoOutput>* right,
                                                               const std::vector<Input::VideoReader*>& readers,
                                                               const std::vector<PreProcessor*>& preprocs,
                                                               PostProcessor* postproc) {
  StereoPipeline* ret = new StereoPipeline(left, right, readers, preprocs, postproc);

  Status initStatus = ret->init();

  if (!initStatus.ok()) {
    delete ret;
    ret = nullptr;
    return initStatus;
  }

  return ret;
}

/**
 *
 * - do not load concurrently (avoid concurrent accesses to disk)
 * - do not write while loading
 * - loading, transmitting and mapping are independant for two different images
 * - merging requires the previous and current images to be mapped.
 * - keep frames independant for the moment, i.e. do not start images for next frames during current frame.
 *
 * Image1 | load | map | merge     |                                                           | load |
 * Image2 |      | load     | map   | merge    |
 * Image3 |                 | load | map |XXXXX| merge    |
 * Image4 |                        | load | map |XXXXXXXXX| merge  |
 * Image5 |                               | load       | map   |XXX| merge  |
 *
 *                                                                          | readback | write |
 *
 * The main thread orchestrates the loading, delegating asynchronous logic to cuda streams.
 */
Status StereoPipeline::stitch(mtime_t date, frameid_t frame, std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                              StereoOutput* output) {
  std::vector<ExtractOutput*> ext;
  return stitchAndExtract(date, frame, inputBuffers, output, ext, nullptr);
}

Status StereoPipeline::stitchAndExtract(mtime_t date, FrameRate frameRate,
                                        std::map<readerid_t, Input::PotentialFrame>& inputBuffers, StereoOutput* output,
                                        std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo) {
  Status leftStatus;
  Status rightStatus;

  FAIL_RETURN(extract(date, frameRate, inputBuffers, extracts, algo));

  auto frame = frameRate.timestampToFrame(date);

  // stereo in single gpu
  // can't fix the race condition, and it's not slower sequentially
  leftStatus = leftStitcher->stitch(date, frame, postproc, inputBuffers, readers, preprocs, output);
  rightStatus = rightStitcher->stitch(date, frame, postproc, inputBuffers, readers, preprocs, output);

  if (!leftStatus.ok() && !rightStatus.ok()) {
    return {Origin::Stitcher, ErrType::RuntimeError, "Stitching of both eyes failed", leftStatus};
  }
  if (!leftStatus.ok()) {
    return {Origin::Stitcher, ErrType::RuntimeError, "Stitching of left eye failed", leftStatus};
  }
  if (!rightStatus.ok()) {
    return {Origin::Stitcher, ErrType::RuntimeError, "Stitching of right eye failed", rightStatus};
  }

  return Status::OK();
}

Status StereoPipeline::setup(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                             const ImageFlowFactory& flowFactory, const StereoRigDefinition* rig) {
  auto leftHandle =
      std::async(std::launch::async, &PanoStitcherImplBase<StereoOutput>::setup, leftStitcher, std::cref(mergerFactory),
                 std::cref(warperFactory), std::cref(flowFactory), readers, rig);
  auto rightHandle =
      std::async(std::launch::async, &PanoStitcherImplBase<StereoOutput>::setup, rightStitcher,
                 std::cref(mergerFactory), std::cref(warperFactory), std::cref(flowFactory), readers, rig);

  // wait for both tasks to finish
  auto leftStatus = leftHandle.get();
  auto rightStatus = rightHandle.get();

  FAIL_RETURN(leftStatus);
  return rightStatus;
}
}  // namespace Core
}  // namespace VideoStitch
