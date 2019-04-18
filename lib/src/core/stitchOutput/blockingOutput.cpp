// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "blockingOutput.hpp"

#include "libvideostitch/algorithm.hpp"

namespace VideoStitch {
namespace Core {
/**
 * BlockingSourceOutput
 */
Status BlockingSourceOutput::pushVideo(mtime_t date) {
  auto res = delegate->pushVideo();
  delegate->streamSynchronize();
  //  Util::SimpleProfiler prof("  write frame", false, Logger::get(Logger::Verbose));
  this->pushVideoToWriters(date, delegate);
  return res;
}

GPU::Surface& BlockingSourceOutput::acquireFrame(mtime_t date, GPU::Stream& stream) {
  return delegate->acquireFrame(date, stream);
}

/**
 * BlockingStitchOutput
 */
Status BlockingStitchOutput::pushVideo(mtime_t date) {
  auto res = delegate->pushVideo();
  delegate->streamSynchronize();
  //  Util::SimpleProfiler prof("  write frame", false, Logger::get(Logger::Verbose));
  this->pushVideoToWriters(date, delegate);
  return res;
}

PanoSurface& BlockingStitchOutput::acquireFrame(mtime_t date) { return delegate->acquireFrame(date); }

/**
 * BlockingStereoOutput
 */
Potential<BlockingStereoOutput> BlockingStereoOutput::create(
    std::shared_ptr<PanoSurface> surf, const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
    const std::vector<std::shared_ptr<Output::StereoWriter>>& writers) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());
  Potential<StereoFrameBuffer> potentialLeft = StereoFrameBuffer::create(surf, writers);
  FAIL_RETURN(potentialLeft.status());
  Potential<StereoFrameBuffer> potentialRight = StereoFrameBuffer::create(surf, writers);
  FAIL_RETURN(potentialRight.status());
  BlockingStereoOutput* ret = new BlockingStereoOutput(surf, renderers, writers);
  ret->left = potentialLeft.release();
  ret->right = potentialRight.release();
  return ret;
}

Status BlockingStereoOutput::pushVideo(mtime_t date, Eye eye) {
  Status res;
  switch (eye) {
    case LeftEye:
      res = left->pushVideo();
      left->streamSynchronize();
      break;
    case RightEye:
      res = right->pushVideo();
      right->streamSynchronize();
      break;
  }
  pushVideoToWriters(date, std::make_pair(left, right));
  return res;
}

PanoSurface& BlockingStereoOutput::acquireLeftFrame(mtime_t date) { return left->acquireFrame(date); }

PanoSurface& BlockingStereoOutput::acquireRightFrame(mtime_t date) { return right->acquireFrame(date); }
}  // namespace Core
}  // namespace VideoStitch
