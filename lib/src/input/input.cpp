// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/input.hpp"

#include "image/unpack.hpp"

#include <mutex>
#include <sstream>

// TODO: enable thread safety annotations.
#define GUARDED_BY(mutex)

namespace VideoStitch {
namespace Input {

Reader::Reader(readerid_t id) : id(id), latency(0) {}

Reader::~Reader() {}

VideoReader::~VideoReader() {}

frameid_t VideoReader::getFirstFrame() const { return firstFrame; }

frameid_t VideoReader::getLastFrame() const { return lastFrame; }

int64_t VideoReader::getWidth() const { return spec.width; }

int64_t VideoReader::getHeight() const { return spec.height; }

int64_t VideoReader::getFrameDataSize() const { return spec.frameDataSize; }

const VideoReader::Spec& VideoReader::getSpec() const { return spec; }

VideoReader::Spec& VideoReader::getSpec() { return spec; }

VideoReader* Reader::getVideoReader() const { return dynamic_cast<VideoReader*>(const_cast<Reader*>(this)); }

AudioReader::~AudioReader() {}

const AudioReader::Spec& AudioReader::getSpec() const { return spec; }

AudioReader::Spec& AudioReader::getSpec() { return spec; }

AudioReader* Reader::getAudioReader() const {
  AudioReader* audio = dynamic_cast<AudioReader*>(const_cast<Reader*>(this));
  if (audio) {
    AudioReader::Spec spec = audio->getSpec();
    if (spec.layout != Audio::UNKNOWN && spec.sampleRate != Audio::SamplingRate::SR_NONE &&
        spec.sampleDepth != Audio::SamplingDepth::SD_NONE) {
      return audio;
    }
  }
  return nullptr;
}

MetadataReader::~MetadataReader() {}

MetadataReader* Reader::getMetadataReader() const { return dynamic_cast<MetadataReader*>(const_cast<Reader*>(this)); }

SinkReader* Reader::getSinkReader() const { return dynamic_cast<SinkReader*>(const_cast<Reader*>(this)); }

mtime_t Reader::getLatency() {
  std::lock_guard<std::mutex> lock(this->latencyMutex);
  return this->latency;
}

void Reader::setLatency(mtime_t value) {
  std::lock_guard<std::mutex> lock(this->latencyMutex);
  this->latency = value;
}

bool Reader::updateLatency(mtime_t value) {
  std::lock_guard<std::mutex> lock(this->latencyMutex);
  if (value > this->latency) {
    this->latency = value;
    return true;
  }
  return false;
}

///////////////////////////////////////////////////////////////////////////////////

class VideoReader::Spec::Impl {
  friend class VideoReader;
  /**
   * A mutex that guards the mutable properties.
   */
  std::mutex mutex;
  /**
   * The display name.
   */
  std::string displayName GUARDED_BY(mutex);
};

class AudioReader::Spec::Impl {
  friend class AudioReader;
  /**
   * A mutex that guards the mutable properties.
   */
  std::mutex mutex;
  /**
   * The display name.
   */
  std::string displayName GUARDED_BY(mutex);
};

class MetadataReader::Spec::Impl {
  friend class MetadataReader;
  /**
   * A mutex that guards the mutable properties
   */
  std::mutex mutex;

  /**
   * The display name
   */
  std::string displayName GUARDED_BY(mutex);
};

///////////////////////////////////////////

VideoReader::Spec::Spec(int64_t width, int64_t height, int64_t frameDataSize, VideoStitch::PixelFormat format,
                        AddressSpace addressSpace, int frameNum, FrameRate frameRate, bool frameRateIsProcedural,
                        const unsigned char* maskHostBuffer, int flags)
    : width(width),
      height(height),
      frameDataSize(frameDataSize),
      format(format),
      addressSpace(addressSpace),
      frameNum(frameNum),
      frameRate(frameRate),
      frameRateIsProcedural(frameRateIsProcedural),
      flags(flags),
      maskHostBuffer(maskHostBuffer),
      pimpl(new Impl()) {}

AudioReader::Spec::Spec(Audio::ChannelLayout layout, Audio::SamplingRate sampleRate, Audio::SamplingDepth sampleDepth)
    : layout(layout), sampleRate(sampleRate), sampleDepth(sampleDepth), pimpl(new Impl()) {}

MetadataReader::Spec::Spec(FrameRate framerate) : frameRate(framerate), pimpl(new Impl()) {}

///////////////////////////////////////////

VideoReader::Spec::Spec()
    : width(0),
      height(0),
      frameDataSize(0),
      format(VideoStitch::RGBA),
      addressSpace(Host),
      frameNum(0),
      frameRate({-1, 1} /* Unknown */),
      frameRateIsProcedural(true),
      flags(0),
      maskHostBuffer(NULL),
      pimpl(new Impl()) {}

AudioReader::Spec::Spec()
    : layout(Audio::UNKNOWN),
      sampleRate(Audio::SamplingRate::SR_NONE),
      sampleDepth(Audio::SamplingDepth::SD_NONE),
      pimpl(new Impl()) {}

MetadataReader::Spec::Spec() : frameRate({-1, 1} /* Unknown */), pimpl(new Impl()) {}

///////////////////////////////////////////

VideoReader::Spec::~Spec() { delete pimpl; }

AudioReader::Spec::~Spec() { delete pimpl; }

MetadataReader::Spec::~Spec() { delete pimpl; }

///////////////////////////////////////////

VideoReader::Spec::Spec(const Spec& o)
    : width(o.width),
      height(o.height),
      frameDataSize(o.frameDataSize),
      format(o.format),
      addressSpace(o.addressSpace),
      frameNum(o.frameNum),
      frameRate(o.frameRate),
      frameRateIsProcedural(o.frameRateIsProcedural),
      flags(o.flags),
      maskHostBuffer(o.maskHostBuffer),
      pimpl(new Impl()) {}

AudioReader::Spec::Spec(const Spec& o)
    : layout(o.layout), sampleRate(o.sampleRate), sampleDepth(o.sampleDepth), pimpl(new Impl()) {}

MetadataReader::Spec::Spec(const Spec& o) : frameRate(o.frameRate), pimpl(new Impl()) {}

///////////////////////////////////////////

void VideoReader::Spec::setDisplayName(const char* name) { pimpl->displayName = name; }

void AudioReader::Spec::setDisplayName(const char* name) { pimpl->displayName = name; }

void MetadataReader::Spec::setDisplayName(const char* name) { pimpl->displayName = name; }

///////////////////////////////////////////

void VideoReader::Spec::getDisplayName(std::ostream& os) const { os << pimpl->displayName; }

void AudioReader::Spec::getDisplayName(std::ostream& os) const { os << pimpl->displayName; }

void MetadataReader::Spec::getDisplayName(std::ostream& os) const { os << pimpl->displayName; }

///////////////////////////////////////////

VideoReader::VideoReader(int64_t width, int64_t height, int64_t frameDataSize, VideoStitch::PixelFormat format,
                         AddressSpace addressSpace, FrameRate frameRate, int firstFrame, int lastFrame,
                         bool isProcedural, const unsigned char* maskHostBuffer,
                         int flags)
    : Reader(-1),  // never called : https://isocpp.org/wiki/faq/multiple-inheritance#virtual-inheritance-ctors
      firstFrame(firstFrame),
      lastFrame(lastFrame),
      spec(width, height, frameDataSize, format, addressSpace, lastFrame - firstFrame + 1, frameRate, isProcedural,
           maskHostBuffer, flags) {
  getSpec().setDisplayName("[unknown input]");
}

AudioReader::AudioReader(Audio::ChannelLayout layout, Audio::SamplingRate sampleRate,
                         Audio::SamplingDepth sampleDepth)
    : Reader(-1),  // never called : https://isocpp.org/wiki/faq/multiple-inheritance#virtual-inheritance-ctors
      spec(layout, sampleRate, sampleDepth) {}

MetadataReader::MetadataReader(FrameRate framerate)
    : Reader(-1),  // never called : https://isocpp.org/wiki/faq/multiple-inheritance#virtual-inheritance-ctors
      spec(framerate) {}

SinkReader::SinkReader()
    : Reader(-1) {}  // never called : https://isocpp.org/wiki/faq/multiple-inheritance#virtual-inheritance-ctors

///////////////////////////////////////////

Status VideoReader::perThreadInit() { return Status::OK(); }

void VideoReader::perThreadCleanup() {}

Status VideoReader::unpackDevBuffer(GPU::Surface& dst, const GPU::Buffer<const unsigned char>& src,
                                    GPU::Stream& stream) const {
  return Image::unpackCommonPixelFormat(spec.format, dst, src, spec.width, spec.height, stream);
}

Status VideoReader::unpackDevBuffer(VideoStitch::PixelFormat fmt, GPU::Surface& dst,
                                    const GPU::Buffer<const unsigned char>& src, uint64_t width, uint64_t height,
                                    GPU::Stream& stream) {
  return Image::unpackCommonPixelFormat(fmt, dst, src, width, height, stream);
}
}  // namespace Input
}  // namespace VideoStitch
