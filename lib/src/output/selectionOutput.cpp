// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "common/container.hpp"

#include "libvideostitch/stitchOutput.hpp"

#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#if defined(_MSC_VER) && defined(VS_OPENCL)
#pragma warning(disable : 4100)
#endif

namespace VideoStitch {
namespace Output {

struct host_tag {};
struct cuda_tag {};

/**
 * Implementation of StereoWriter selection either Left or Right eye.
 */
template <typename Buffer>
class SelectionOutput : public StereoWriter, public AudioWriter {
 public:
  SelectionOutput(VideoWriter* delegateWriter, Eye eye)
      : Output(delegateWriter->getName()),
        StereoWriter(delegateWriter->getWidth(), delegateWriter->getHeight(), delegateWriter->getFrameRate(),
                     delegateWriter->getWidth(), delegateWriter->getHeight(), delegateWriter->getPixelFormat(),
                     delegateWriter->getExpectedOutputBufferType()),
        AudioWriter((delegateWriter->getAudioWriter()) ? delegateWriter->getAudioWriter()->getSamplingRate()
                                                       : Audio::SamplingRate::SR_NONE,
                    (delegateWriter->getAudioWriter()) ? delegateWriter->getAudioWriter()->getSamplingDepth()
                                                       : Audio::SamplingDepth::SD_NONE,
                    (delegateWriter->getAudioWriter()) ? delegateWriter->getAudioWriter()->getChannelLayout()
                                                       : Audio::ChannelLayout::UNKNOWN),
        delegate(delegateWriter),
        eye(eye) {}

  ~SelectionOutput() {}

  virtual void pushEye(Eye e, const Frame& frame) override {
    if (eye == e) {
      delegate->pushVideo(frame);
    }
  }
  AudioWriter* getAudioWriter() const override { return delegate->getAudioWriter(); }

  void pushAudio(Audio::Samples& audioSamples) override {
    assert(getAudioWriter());
    return delegate->getAudioWriter()->pushAudio(audioSamples);
  }

 private:
  VideoWriter* delegate;
  const Eye eye;
};

Potential<StereoWriter> StereoWriter::createSelection(VideoWriter* writer, Eye eye, AddressSpace buffer) {
#ifdef VS_OPENCL
  return {Origin::Output, ErrType::UnsupportedAction, "Stereo output not supported in OpenCL backend"};
#else   // VS_OPENCL
  switch (buffer) {
    case Host:
      return Potential<StereoWriter>(new SelectionOutput<host_tag>(writer, eye));
    case Device:
    default:
      return Potential<StereoWriter>(new SelectionOutput<cuda_tag>(writer, eye));
  }
#endif  // VS_OPENCL
}

}  // namespace Output
}  // namespace VideoStitch
