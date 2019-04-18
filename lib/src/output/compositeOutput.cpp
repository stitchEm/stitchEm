// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "compositeOutput.hpp"
#include "common/container.hpp"

#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/gpu_device.hpp"

#ifdef VS_OPENCL

namespace VideoStitch {
namespace Output {

StereoWriter::StereoWriter(unsigned width, unsigned height, FrameRate framerate, unsigned panoWidth,
                           unsigned panoHeight, VideoStitch::PixelFormat pixelFormat, AddressSpace outputType)
    : Output(""),
      width(width),
      height(height),
      framerate(framerate),
      panoWidth(panoWidth),
      panoHeight(panoHeight),
      format(pixelFormat),
      outputType(outputType) {
  // TODO_OPENCL_IMPL
}

Potential<StereoWriter> StereoWriter::createComposition(VideoWriter* /*writer*/, Layout /*layout*/,
                                                        AddressSpace /*buffer*/) {
  return {Origin::Output, ErrType::UnsupportedAction, "Stereo output not supported in OpenCL backend"};
}

int64_t StereoWriter::getExpectedFrameSize() const {
  return VideoWriter::getExpectedFrameSizeFor(format, width, height);
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

/**
 * Template helper.
 */
template <Eye eye>
struct EyeOffset {};

template <>
struct EyeOffset<LeftEye> {
  static const int64_t value;
};
const int64_t EyeOffset<LeftEye>::value = 0;

template <>
struct EyeOffset<RightEye> {
  static const int64_t value;
};
const int64_t EyeOffset<RightEye>::value = 1;

struct host_tag {};
struct cuda_tag {};

/**
 * Implementation of StereoWriter.
 */
template <typename Buffer>
class CompositeWriter : public StereoWriter, public AudioWriter {
 public:
  CompositeWriter(VideoWriter* delegateWriter, Layout layout)
      : Output(delegateWriter->getName()),
        StereoWriter(delegateWriter->getWidth(), delegateWriter->getHeight(), delegateWriter->getFrameRate(),
                     layout == HorizontalLayout ? delegateWriter->getWidth() / 2 : delegateWriter->getWidth(),
                     layout == VerticalLayout ? delegateWriter->getHeight() / 2 : delegateWriter->getHeight(),
                     delegateWriter->getPixelFormat(), delegateWriter->getExpectedOutputBufferType()),
        AudioWriter((delegateWriter->getAudioWriter()) ? delegateWriter->getAudioWriter()->getSamplingRate()
                                                       : Audio::SamplingRate::SR_NONE,
                    (delegateWriter->getAudioWriter()) ? delegateWriter->getAudioWriter()->getSamplingDepth()
                                                       : Audio::SamplingDepth::SD_NONE,
                    (delegateWriter->getAudioWriter()) ? delegateWriter->getAudioWriter()->getChannelLayout()
                                                       : Audio::ChannelLayout::UNKNOWN),
        delegate(delegateWriter),
        layout(layout) {}

  ~CompositeWriter() {
    for (auto buf : freeBuffersStack) {
      freeBuffer(buf, Buffer());
    }
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
    char* frameBuffer = NULL;
    {
      std::unique_lock<std::mutex> lock(mutex);
      char*& frameBufferTmp = frameToBuffer[frame.pts];
      if (!frameBufferTmp) {
        frameBufferTmp = acquireBuffer(format);
        frameToBuffer[frame.pts] = frameBufferTmp;
      }
      frameBuffer = frameBufferTmp;
    }
    // This needs not be locked (we already have a buffer and the other eye will write to different pixels).
    switch (layout) {
      case HorizontalLayout:
        writeHalfBufferHorizontal<eye>(frameBuffer, (const char*)frame.planes[0]);
        break;
      case VerticalLayout:
        writeHalfBufferVertical<eye>(frameBuffer, (const char*)frame.planes[0]);
        break;
    }
    {
      // When we have both eyes, write the frame.
      std::unique_lock<std::mutex> lock(mutex);
      int& numEyesWritten = frameToNumEyesPushed[frame.pts];
      ++numEyesWritten;
      assert(numEyesWritten <= 2);
      if (numEyesWritten >= 2) {
        Frame f = {{frameBuffer, nullptr, nullptr},
                   {frame.pitches[0], frame.pitches[1], frame.pitches[2]},
                   layout == HorizontalLayout ? frame.width * 2 : frame.width,
                   layout == VerticalLayout ? frame.height * 2 : frame.height,
                   frame.pts,
                   frame.fmt};
        delegate->pushVideo(f);
        freeBuffersStack.push_back(frameBuffer);
        frameToBuffer[frame.pts] = nullptr;
        numEyesWritten = 0;
      }
    }
  }

  const VideoWriter& getDelegate() const { return *delegate; }

  Layout getLayout() const { return layout; }

 private:
  /**
   * Get a free buffer (create one if needed). Requires mutex to be locked.
   */
  char* acquireBuffer(VideoStitch::PixelFormat format) {
    if (freeBuffersStack.empty()) {
      return allocateBuffer(VideoWriter::getExpectedFrameSizeFor(format, width, height), Buffer());
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
    Cuda::mallocVS((void**)&ptr, size, "CompositeOutput");
    return ptr;
  }
  void freeBuffer(char* buf, host_tag) { delete[] buf; }
  void freeBuffer(char* buf, cuda_tag) { Cuda::freeVS(buf); }

  // --- horizontal -----------
  template <Eye eye>
  void writeHalfBufferHorizontal(char* dst, const char* src) const {
    const int64_t srcWidth = delegate->getWidth() / 2;
    const int64_t xOffset = EyeOffset<eye>::value * srcWidth;
    switch (delegate->getPixelFormat()) {
      case VideoStitch::PixelFormat::RGBA:
      case VideoStitch::PixelFormat::BGRU:
        writeHalfBufferHorizontalInter<4>(dst, src, delegate->getHeight(), delegate->getWidth(), srcWidth, xOffset,
                                          Buffer());
        break;
      case VideoStitch::PixelFormat::UYVY:
      case VideoStitch::PixelFormat::YUY2:
        writeHalfBufferHorizontalInter<2>(dst, src, delegate->getHeight(), delegate->getWidth(), srcWidth, xOffset,
                                          Buffer());
        break;
      case VideoStitch::PixelFormat::RGB:
      case VideoStitch::PixelFormat::BGR:
        writeHalfBufferHorizontalInter<3>(dst, src, delegate->getHeight(), delegate->getWidth(), srcWidth, xOffset,
                                          Buffer());
        break;
      case VideoStitch::PixelFormat::YV12:
        writeHalfBufferHorizontalPlanar(dst, src, delegate->getHeight(), delegate->getWidth(), srcWidth, xOffset);
      case VideoStitch::PixelFormat::Grayscale:
        writeHalfBufferHorizontalInter<1>(dst, src, delegate->getHeight(), delegate->getWidth(), srcWidth, xOffset,
                                          Buffer());
        break;
      default:
        assert(false);
    }
  }

  // horizontal planar
  void writeHalfBufferHorizontalPlanar(char* dst, const char* src, const int64_t height, const int64_t dstWidth,
                                       const int64_t srcWidth, const int64_t xOffset) const {
    writeHalfBufferHorizontalInter<1>(dst, src, height, dstWidth, srcWidth, xOffset, Buffer());
    writeHalfBufferHorizontalInter<1>(dst + height * dstWidth, src + height * srcWidth, (height + 1) / 2,
                                      (dstWidth + 1) / 2, (srcWidth + 1) / 2, xOffset / 2, Buffer());
    writeHalfBufferHorizontalInter<1>(dst + height * dstWidth + ((height + 1) / 2) * ((dstWidth + 1) / 2),
                                      src + height * srcWidth + ((height + 1) / 2) * ((srcWidth + 1) / 2),
                                      (height + 1) / 2, (dstWidth + 1) / 2, (srcWidth + 1) / 2, xOffset / 2, Buffer());
  }

  // --- vertical -----------
  template <Eye eye>
  void writeHalfBufferVertical(char* dst, const char* src) const {
    const int64_t srcHeight = delegate->getHeight() / 2;
    const int64_t yOffset = EyeOffset<eye>::value * srcHeight;
    switch (delegate->getPixelFormat()) {
      case VideoStitch::PixelFormat::RGBA:
      case VideoStitch::PixelFormat::BGRU:
        writeHalfBufferVertical<4>(dst + 4 * yOffset * delegate->getWidth(), src, delegate->getWidth() * srcHeight,
                                   Buffer());
        break;
      case VideoStitch::PixelFormat::UYVY:
      case VideoStitch::PixelFormat::YUY2:
        writeHalfBufferVertical<2>(dst + 4 * yOffset * delegate->getWidth(), src, delegate->getWidth() * srcHeight,
                                   Buffer());
        break;
      case VideoStitch::PixelFormat::RGB:
      case VideoStitch::PixelFormat::BGR:
        writeHalfBufferVertical<3>(dst + 3 * yOffset * delegate->getWidth(), src, delegate->getWidth() * srcHeight,
                                   Buffer());
        break;
      case VideoStitch::PixelFormat::YV12:
        writeHalfBufferVerticalPlanar<eye>(dst, src, delegate->getWidth(), delegate->getHeight());
        break;
      case VideoStitch::PixelFormat::Grayscale:
        writeHalfBufferVertical<1>(dst + yOffset * delegate->getWidth(), src, delegate->getWidth() * srcHeight,
                                   Buffer());
        break;
      default:
        assert(false);
    }
  }

  // vertical planar
  template <Eye eye>
  void writeHalfBufferVerticalPlanar(char* dst, const char* src, const int64_t width, const int64_t height) const {
    const char* ySrc = src;
    const char* uSrc = src + width * height / 2;
    const char* vSrc = src + (width * height * 5) / 8;
    char* yDst = dst;
    char* uDst = dst + width * height;
    char* vDst = dst + (width * height * 5) / 4;
    // Y
    writeHalfBufferVertical<1>(yDst + EyeOffset<eye>::value * width * height / 2, ySrc, width * height / 2, Buffer());
    // U
    writeHalfBufferVertical<1>(uDst + EyeOffset<eye>::value * width * height / 8, uSrc, width * height / 8, Buffer());
    // V
    writeHalfBufferVertical<1>(vDst + EyeOffset<eye>::value * width * height / 8, vSrc, width * height / 8, Buffer());
  }

  /**
   * Host implementation ------------------------------
   */
  template <int pixelSize>
  void writeHalfBufferHorizontalInter(char* dst, const char* src, const int64_t height, const int64_t dstWidth,
                                      const int64_t srcWidth, const int64_t xOffset, host_tag) const {
    for (int64_t y = 0; y < height; ++y) {
      // TODO: try out memcpy()ing.
      for (int64_t x = 0; x < srcWidth; ++x) {
        for (int k = 0; k < pixelSize; ++k) {
          dst[pixelSize * (y * dstWidth + x + xOffset) + k] = src[pixelSize * (y * srcWidth + x) + k];
        }
      }
    }
  }

  template <int pixelSize>
  void writeHalfBufferVertical(char* dst, const char* src, const int64_t size, host_tag) const {
    memcpy(dst, src, pixelSize * size);
  }

  /**
   * Cuda implementation ------------------------------
   */
  template <int pixelSize>
  void writeHalfBufferHorizontalInter(char* dst, const char* src, const int64_t height, const int64_t dstWidth,
                                      const int64_t srcWidth, const int64_t xOffset, cuda_tag) const {
    VideoStitch::Output::writeHalfBufferHorizontalInter<pixelSize>(dst, src, height, dstWidth, srcWidth, xOffset);
  }

  template <int pixelSize>
  void writeHalfBufferVertical(char* dst, const char* src, const int64_t size, cuda_tag) const {
    cudaMemcpy(dst, src, pixelSize * size, cudaMemcpyDeviceToDevice);
  }

  VideoWriter* delegate;
  const Layout layout;
  std::mutex mutex;
  std::map<mtime_t, char*> frameToBuffer;
  std::map<mtime_t, int> frameToNumEyesPushed;
  // Buffer memory pool.
  std::vector<char*> freeBuffersStack;
};

Potential<StereoWriter> StereoWriter::createComposition(VideoWriter* writer, Layout layout, AddressSpace buffer) {
  switch (buffer) {
    case Host:
      return Potential<StereoWriter>(new CompositeWriter<host_tag>(writer, layout));
    case Device:
    default:
      return Potential<StereoWriter>(new CompositeWriter<cuda_tag>(writer, layout));
  }
}

StereoWriter::StereoWriter(unsigned width, unsigned height, FrameRate framerate, unsigned panoWidth,
                           unsigned panoHeight, VideoStitch::PixelFormat pixelFormat, AddressSpace outputType)
    : Output(""),
      width(width),
      height(height),
      framerate(framerate),
      panoWidth(panoWidth),
      panoHeight(panoHeight),
      format(pixelFormat),
      outputType(outputType) {}

int64_t StereoWriter::getExpectedFrameSize() const {
  return VideoWriter::getExpectedFrameSizeFor(format, width, height);
}
}  // namespace Output
}  // namespace VideoStitch

#endif  // VS_OPENCL
