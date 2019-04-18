// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "anaglyph.hpp"
#include "common/container.hpp"

#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/gpu_device.hpp"

#ifdef VS_OPENCL

namespace VideoStitch {
namespace Output {

Potential<StereoWriter> StereoWriter::createAnaglyphColor(VideoWriter* /*writer*/, AddressSpace /*buffer*/,
                                                          int /*device*/) {
  return {Origin::Output, ErrType::UnsupportedAction, "Anaglyph color creation not supported"};
}

}  // namespace Output
}  // namespace VideoStitch

#else  // VS_OPENCL

#include "cuda/memory.hpp"
#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace VideoStitch {
namespace Output {

struct host_tag {};
struct cuda_tag {};

/**
 * Implementation of StereoWriter.
 */
template <typename Buffer>
class AnaglyphColor : public StereoWriter, public AudioWriter {
 public:
  AnaglyphColor(VideoWriter* delegateWriter, int device)
      : Output(delegateWriter->getName()),
        StereoWriter(delegateWriter->getWidth(), delegateWriter->getHeight(), delegateWriter->getFrameRate(),
                     delegateWriter->getWidth(), delegateWriter->getHeight(), PixelFormat::RGBA,
                     delegateWriter->getExpectedOutputBufferType()),
        AudioWriter((delegateWriter->getAudioWriter()) ? delegateWriter->getAudioWriter()->getSamplingRate()
                                                       : Audio::SamplingRate::SR_NONE,
                    (delegateWriter->getAudioWriter()) ? delegateWriter->getAudioWriter()->getSamplingDepth()
                                                       : Audio::SamplingDepth::SD_NONE,
                    (delegateWriter->getAudioWriter()) ? delegateWriter->getAudioWriter()->getChannelLayout()
                                                       : Audio::ChannelLayout::UNKNOWN),
        delegate(delegateWriter),
        device(device) {
    transferBuffer = acquireBuffer();
  }

  ~AnaglyphColor() {
    for (auto buf : freeBuffersStack) {
      freeBuffer(buf, Buffer());
    }
    freeBuffer(transferBuffer, Buffer());
    delete delegate;
  }

  virtual void pushEye(Eye eye, const Frame& frame) override {
    switch (eye) {
      case LeftEye:
        pushEye<LeftEye>(frame);
        break;
      case RightEye:
        pushEye<RightEye>(frame);
        break;
    }
  }

  AudioWriter* getAudioWriter() const override { return delegate->getAudioWriter(); }

  void pushAudio(Audio::Samples& audioSamples) override {
    assert(getAudioWriter());
    return delegate->getAudioWriter()->pushAudio(audioSamples);
  }

 private:
  // Note that this will be called concurrently.
  template <Eye eye>
  void pushEye(const Frame& frame) {
    {
      std::unique_lock<std::mutex> lock(mutex);
      char* frameBuffer = nullptr;
      char*& frameBufferTmp = frameToBuffer[frame.pts];
      if (!frameBufferTmp) {
        frameBufferTmp = acquireBuffer();
        resetBuffer(frameBufferTmp, Buffer());
        frameToBuffer[frame.pts] = frameBufferTmp;
      }
      frameBuffer = frameBufferTmp;

      // Since values from left and right add up, this has to be performed under lock
      anaglyphRender(frameBuffer, (const char*)frame.planes[0], eye, Buffer());

      // When we have both eyes, write the frame.
      int& numEyesWritten = frameToNumEyesPushed[frame.pts];
      ++numEyesWritten;
      assert(numEyesWritten <= 2);
      if (numEyesWritten >= 2) {
        Frame f = {{frameBuffer, nullptr, nullptr},
                   {frame.pitches[0], frame.pitches[1], frame.pitches[2]},
                   frame.width,
                   frame.height,
                   frame.pts,
                   frame.fmt};
        delegate->pushVideo(f);
        freeBuffersStack.push_back(frameBuffer);
        frameToBuffer[frame.pts] = nullptr;
        numEyesWritten = 0;
      }
    }
  }

  void anaglyphRender(char*, const char*, Eye, host_tag) const {
    // not implemented yet
  }

  void anaglyphRender(char* dst, const char* src, Eye eye, cuda_tag) const {
    struct cudaPointerAttributes attrs;
    cudaPointerGetAttributes(&attrs, src);
    const char* eyeBuffer;
    if (attrs.device != device) {
      // transfer the eye frame to the target GPU
      cudaMemcpy(transferBuffer, src, VideoWriter::getExpectedFrameSizeFor(PixelFormat::RGBA, width, height),
                 cudaMemcpyDeviceToDevice);
      eyeBuffer = transferBuffer;
    } else {
      eyeBuffer = src;
    }
    switch (eye) {
      case LeftEye:
        anaglyphColorLeft(reinterpret_cast<uint32_t*>(dst), reinterpret_cast<const uint32_t*>(eyeBuffer), width,
                          height);
        break;
      case RightEye:
        anaglyphColorRight(reinterpret_cast<uint32_t*>(dst), reinterpret_cast<const uint32_t*>(eyeBuffer), width,
                           height);
        break;
    }
  }

  const VideoWriter& getDelegate() const { return *delegate; }

 private:
  /**
   * Get a free buffer (create one if needed). Requires mutex to be locked.
   */
  char* acquireBuffer() {
    if (freeBuffersStack.empty()) {
      return allocateBuffer(VideoWriter::getExpectedFrameSizeFor(PixelFormat::RGBA, width, height), Buffer());
    } else {
      char* tmp = freeBuffersStack.back();
      freeBuffersStack.pop_back();
      return tmp;
    }
  }

  char* allocateBuffer(size_t size, host_tag) { return new char[size]; }
  char* allocateBuffer(size_t size, cuda_tag) {
    GPU::useDefaultBackendDevice();
    char* ptr;
    Cuda::mallocVS((void**)&ptr, size, "Anaglyph");
    return ptr;
  }
  void resetBuffer(char* buf, host_tag) { memset(buf, 0, width * height * 4); }
  void resetBuffer(char* buf, cuda_tag) { cudaMemset(buf, 0, width * height * 4); }
  void freeBuffer(char* buf, host_tag) { delete[] buf; }
  void freeBuffer(char* buf, cuda_tag) { Cuda::freeVS(buf); }

  VideoWriter* delegate;
  int device;
  std::mutex mutex;
  std::map<mtime_t, char*> frameToBuffer;
  std::map<mtime_t, int> frameToNumEyesPushed;
  // Buffer memory pool.
  std::vector<char*> freeBuffersStack;
  char* transferBuffer;
};

Potential<StereoWriter> StereoWriter::createAnaglyphColor(VideoWriter* writer, AddressSpace buffer, int device) {
  switch (buffer) {
    case Host:
      return Potential<StereoWriter>(new AnaglyphColor<host_tag>(writer, device));
    case Device:
    default:
      return Potential<StereoWriter>(new AnaglyphColor<cuda_tag>(writer, device));
  }
}
}  // namespace Output
}  // namespace VideoStitch

#endif  // VS_OPENCL
