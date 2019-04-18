// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "status.hpp"
#include "audio.hpp"
#include "frame.hpp"
#include "imuData.hpp"
#include "orah/exposureData.hpp"
#include "sink.hpp"

#include <stdint.h>
#include <cstdlib>
#include <limits>
#include <iosfwd>
#include <memory>
#include <mutex>

namespace VideoStitch {

namespace GPU {

template <typename T>
class Buffer;
class Surface;
class Stream;
}  // namespace GPU

namespace Input {

class VideoReader;
class AudioReader;
class MetadataReader;
class SinkReader;

enum class ReadStatusCode {
  // generic states
  Ok,
  ErrorWithStatus,
  // custom reader states
  EndOfFile,
  // the call is marked non-blocking and the requested operation would block, or:
  // there's no data available yet
  TryAgain,
};

typedef Result<ReadStatusCode> ReadStatus;

/**
 * @brief The base pointer for audio and video readers.
 */
class VS_EXPORT Reader {
 public:
  virtual ~Reader();

  // The index of this reader in the InputDefinition list of the project.
  // Makes the bond between the configuration values and the realtime objects.
  const readerid_t id;  ///< id

  /**
   * Type-casting to video reader
   * Might return null for readers with restricted capabilities.
   */
  VideoReader* getVideoReader() const;
  /**
   * Type-casting to audio reader
   * Might return null for readers with restricted capabilities.
   */
  AudioReader* getAudioReader() const;

  MetadataReader* getMetadataReader() const;

  SinkReader* getSinkReader() const;

  mtime_t getLatency();
  void setLatency(mtime_t); /* force new latency value */
  /*
   * update latency only if it increase and return true
   * return false if latency unmodified
   */
  bool updateLatency(mtime_t);

 protected:
  /**
   * @param id id
   */
  explicit Reader(readerid_t id);

 private:
  mtime_t latency;
  std::mutex latencyMutex;
};

/**
 * @brief The common interface to be implemented by all input video readers.
 *
 * Note that this class is performance-critical, so if you think about implementing
 * a custom reader you should be aware of general computing performance issues and read
 * the design docs of the input library.
 *
 * In particular, most reader will be I/O bound, and therefore the design is such
 * that it's better to do heavy processing in:
 *  - initialization methods
 *  - readFrame()
 *
 * Any other method is usually trivial.
 *
 * Note that readers should avoid being asynchronous and doing work in threads since the lib
 * already handles handles that at a higher level and strives to reuse buffers to limit memory usage.
 *
 * The pipeline works as follow: first, the frame data, with size getFrameDataSize(),
 * is read using readFrame() and transmitted to the GPU. readFrame() should never write
 * more than getFrameDataSize() bytes into the output buffer.
 * Loading immediately begins for the next image in parallel.
 * At the same time, the getFrameDataSize() of data is transmitted to the GPU.
 * If the pixel format is unknown to VideoStitch, unpackDevBuffer() is then called on the GPU device buffer data.
 * At this point, the data must be in packed 32-bit RGBA form on the GPU.
 * Then the reader is left alone and the mapping is started.
 *
 * In particular, procedural readers should strive to do all the work on the GPU
 * (i.e. in unpackDevBuffer).
 *
 * NOTE: By convention, frames are zero-based.
 */
class VS_EXPORT VideoReader : public virtual Reader {
 public:
  /**
   * @brief A class that holds time-constant specifications.
   */
  struct VS_EXPORT Spec {
    /**
     * Constructs a spec.
     * @param width The reader width.
     * @param height The reader height.
     * @param frameDataSize Size in bytes of a single frame.
     * @param pixelFormat The pixel format.
     * @param frameNum The number of frames for this reader.
     * @param frameRate Frame rate, in frames per second. Negative means unknown.
     * @param frameRateIsProcedural True if the frame rate is procedural, False otherwise, meaning it does not come from
     * an actual reader that needs to work with other readers sharing the same frame rate.
     * @param maskHostBuffer Mask data.
     * @param flags Reserved. Must be 0.
     */
    Spec(int64_t width, int64_t height, int64_t frameDataSize, VideoStitch::PixelFormat pixelFormat,
         AddressSpace addressspace, int frameNum, FrameRate frameRate, bool frameRateIsProcedural,
         const unsigned char* maskHostBuffer, int flags = 0);

    Spec();

    /**
     * Copy from @a spec.
     */
    Spec(const Spec& spec);

    ~Spec();

    /**
     * The reader width.
     */
    const int64_t width;
    /**
     * The reader height.
     */
    const int64_t height;
    /**
     * Size in bytes of a single frame.
     */
    const int64_t frameDataSize;
    /**
     * The pixel format.
     */
    const VideoStitch::PixelFormat format;
    /**
     * Input frames location.
     */
    const AddressSpace addressSpace;
    /**
     * Number of frames for this reader.
     */
    const int frameNum;
    /**
     * Frame rate, in frames per second.
     * Negative values mean that the reader has no known frame rate.
     */
    const FrameRate frameRate;
    /**
     * Flag whether the frame rate is procedural or not
     * meaning it does not come from an actual reader that needs to work with other readers sharing the same frame rate
     */
    const bool frameRateIsProcedural;
    /**
     * Flags. Reserved.
     */
    const int flags;
    /**
     * Host buffer containing input mask, of size width * height.
     * NULL means no mask. Not owned.
     */
    const unsigned char* const maskHostBuffer;

    /**
     * Returns a display name for the reader. Never assume any format for that. EVER.
     * That means that the only thing that's allowed with that is to diplay it to the user.
     * No testing for equality, parsing...
     * Thread safe.
     * @param os Sink for the display name.
     */
    void getDisplayName(std::ostream& os) const;

    /**
     * Reader implementors: use that to set the display name. You're free to put anything, and change it. Make it small.
     * @param name The display name to set.
     */
    void setDisplayName(const char* name);

   private:
    Spec& operator=(const Spec&) = delete;

    class Impl;
    Impl* const pimpl;
  };

 public:
  virtual ~VideoReader();

  /**
   * Read a frame into videoFrame and advance to the next frame.
   * Optionally read an audio packet synchronized with this frame.
   * It is the Reader's job to ensure correct synchronization.
   *
   * @param date The capture/decode timestamp in microseconds.
   * @param videoFrame Where to write the video frame.
   *                   The provided buffer is of size at least getFrameDataSize().
   * @return a success status.
   */
  virtual ReadStatus readFrame(mtime_t& date, unsigned char* videoFrame) = 0;

  /**
   * Seek to the given frame.
   * @param frame Where to seek. Starts at the reader's first frame.
   * @return False on failure. In that case, the reader is free to leave the
   *                           current frame is in an undetermined state.
   *                           The client can call getCurFrame() or call
   *                           seekFrame() again to seek to another frame.
   * @note for implementors: On failure, it's advised to do the most efficient
   *                         thing given that the next call will probably be a
   *                         seek to the previous 'current' frame.
   */
  virtual Status seekFrame(frameid_t frame) = 0;

  /**
   * Returns the first frame in the sequence (inclusive).
   */
  frameid_t getFirstFrame() const;
  /**
   * Returns the last frame in the sequence (inclusive), or NO_LAST_FRAME.
   */
  frameid_t getLastFrame() const;

  /**
   * Convert the device memory from the internal pixel format to RGBA.
   * Use this function if your PixelFormat is Unknown to VideoStitch.
   *
   * Note that to be efficient, implementations of this method are not allowed
   * to do I/O. It is even encouraged that this method only contains a kernel call.
   * Implementations of this functions must be stateless (i.e. they cannot touch any
   * non-const state in the class).
   * The function should return only when the data in the buffer is ready to be
   * processed, or push a kernel call to the given CUDA stream.
   * The kernel call should use stream (cudaStream_t)stream, and should be asynchronous
   * for better performance.
   * @param dst Destination buffer
   * @param src source buffer
   * @param stream GPU stream on which the computations will run.
   */
  virtual Status unpackDevBuffer(GPU::Surface& dst, const GPU::Buffer<const unsigned char>& src,
                                 GPU::Stream& stream) const;

  /**
   * Returns the width of the reader.
   */
  int64_t getWidth() const;

  /**
   * Returns the height of the reader.
   */
  int64_t getHeight() const;

  /**
   * Returns the size of the data for one frame that
   * should be transfered over to the GPU.
   */
  int64_t getFrameDataSize() const;

  /**
   * Returns the (const) reader spec.
   */
  const Spec& getSpec() const;

  /**
   * Returns the reader spec.
   */
  Spec& getSpec();

  /**
   * Some readers need to allocate GPU memory behind the scene. Since GPU allocations are tied to threads,
   * any allocation that is done by the reader (including, but not restricted to, readFrame() or unpackDevBuffer())
   * will need to be done in the same thread that called these methods.
   * The controller will take care of this when the stitchers are released if implementors cleanup memory in this
   * method. You can skip implementing this method if you don't do such allocations.
   */
  virtual Status perThreadInit();

  /**
   * Performs per-thread cleanup. See perThreadInit().
   */
  virtual void perThreadCleanup();

  /**
   * Convert the device memory from a given known pixel format to RGBA.
   *
   * @param fmt Current pixel format of src Buffer, to be unpacked to RGBA.
   * @param dst Destination buffer
   * @param src read-only source buffer
   * @param width Frame width
   * @param height Frame height
   * @param stream GPU stream used for computations.
   */
  static Status unpackDevBuffer(VideoStitch::PixelFormat fmt, GPU::Surface& dst,
                                const GPU::Buffer<const unsigned char>& src, uint64_t width, uint64_t height,
                                GPU::Stream& stream);

 protected:
  /**
   * Constructor.
   * @param width The reader width. Readers cannot be resized.
   * @param height The reader height. Readers cannot be resized.
   * @param frameDataSize Size in pixels.
   * @param format Native pixel format of the Reader before unpacking.
   * @param addressSpace Address space of the produced frames.
   * @param frameRate Frame rate, in frames per second. Negative means unknown.
   * @param firstFrame First available frame.
   * @param lastFrame Last available frame.
   * @param isProcedural boolean whether the reader is procedural or not
   * @param maskHostBuffer Mask data.
   * @param flags Reserved. Must be 0.
   * @note Everything must be known at creation time.
   */
  VideoReader(int64_t width, int64_t height, int64_t frameDataSize, VideoStitch::PixelFormat format,
              AddressSpace addressSpace, FrameRate frameRate, frameid_t firstFrame, frameid_t lastFrame,
              bool isProcedural, const unsigned char* maskHostBuffer, int flags = 0);

 private:
  VideoReader();
  explicit VideoReader(const VideoReader&);

  /**
   * First available frame.
   */
  const unsigned firstFrame;

  /**
   * Last available frame.
   */
  const unsigned lastFrame;

  Spec spec;
};

/**
 * @brief The common interface to be implemented by all input audio readers.
 */
class VS_EXPORT AudioReader : public virtual Reader {
 public:
  /**
   * @brief A class that holds time-constant specifications.
   */
  struct VS_EXPORT Spec {
    /**
     * Constructs a spec.
     * @param layout Audio channels layout.
     * @param sampleRate Audio sampling frequency.
     * @param sampleDepth Format of the audio samples.
     */
    Spec(Audio::ChannelLayout layout, Audio::SamplingRate sampleRate, Audio::SamplingDepth sampleDepth);

    Spec();

    /**
     * Copy from @a spec.
     */
    Spec(const Spec& spec);

    ~Spec();

    /**
     * Audio channels layout
     */
    Audio::ChannelLayout layout;

    /**
     * Audio sampling rate.
     */
    Audio::SamplingRate sampleRate;

    /**
     * Audio channel depth and layout.
     */
    Audio::SamplingDepth sampleDepth;

    /**
     * Returns a display name for the reader. Never assume any format for that. EVER.
     * That means that the only thing that's allowed with that is to diplay it to the user.
     * No testing for equality, parsing...
     * Thread safe.
     * @param os Sink for the display name.
     */
    void getDisplayName(std::ostream& os) const;

    /**
     * Reader implementors: use that to set the display name. You're free to put anything, and change it. Make it small.
     * @param name The display name to set.
     */
    void setDisplayName(const char* name);

   private:
    Spec& operator=(const Spec&) = delete;

    class Impl;
    Impl* const pimpl;
  };

 public:
  virtual ~AudioReader();

  /**
   * Read a frame of audio samples and advance to the next frame.
   *
   * @param nbSamples    Read a maximum of 'nbSamples' samples of audio into buffer.
   * @param audioSamples Where the audio samples channel planes will be written.
   *                     Up to 16 planar channels are supported.
   *                     Audio channel samples are either interleaved into a sample
   *                     frame or planar, either contiguous or padded (see Spec).
   *                     The memory of each plane is not allocated.
   *                     The ownership is transfered to the stitcher.
   *                     Also contains the timestamp of the first sample of the
   *                     buffer in microseconds.
   * @return a success status.
   */
  virtual ReadStatus readSamples(size_t nbSamples, Audio::Samples& audioSamples) = 0;

  /**
   * Seek to the given time.
   * @param date Where to seek.
   * @return False on failure. In that case, the reader is free to leave the
   *                           current frame is in an undetermined state.
   *                           The client can call getCurFrame() or call
   *                           seekFrame() again to seek to another frame.
   * @note for implementors: On failure, it's advised to do the most efficient
   *                         thing given that the next call will probably be a
   *                         seek to the previous 'current' frame.
   */
  virtual Status seekFrame(mtime_t date) = 0;

  /**
   * Returns the number of samples (per channel) currently available in the
   * reader's buffer.
   * @return number of samples (per channel) available to be read
   */
  virtual size_t available() = 0;

  /**
   * Used to determine if the end of stream has been reached.
   * @return true if the end of stream is reached, false otherwise.
   */
  virtual bool eos() = 0;

  /**
   * Returns the (const) reader spec.
   */
  const Spec& getSpec() const;

  /**
   * Returns the reader spec.
   */
  Spec& getSpec();

 protected:
  /**
   * Constructor.
   * @param layout Audio channels layout.
   * @param sampleRate Audio sample rate
   * @param sampleDepth Audio sample depth
   * @note Everything must be known at creation time.
   */
  AudioReader(Audio::ChannelLayout layout, Audio::SamplingRate sampleRate, Audio::SamplingDepth sampleDepth);

 private:
  AudioReader();
  explicit AudioReader(const AudioReader&);

  Spec spec;
};

static inline std::shared_ptr<AudioReader> getAudioSharedReader(std::shared_ptr<Reader> reader) {
  std::shared_ptr<AudioReader> audio = std::dynamic_pointer_cast<AudioReader>(reader);
  if (audio) {
    AudioReader::Spec spec = audio->getSpec();
    if (spec.layout != Audio::UNKNOWN && spec.sampleRate != Audio::SamplingRate::SR_NONE &&
        spec.sampleDepth != Audio::SamplingDepth::SD_NONE) {
      return audio;
    }
  }
  return nullptr;
}

static inline std::shared_ptr<VideoReader> getVideoSharedReader(std::shared_ptr<Reader> reader) {
  return std::dynamic_pointer_cast<VideoReader>(reader);
}

static inline std::shared_ptr<MetadataReader> getMetadataSharedReader(std::shared_ptr<Reader> reader) {
  return std::dynamic_pointer_cast<MetadataReader>(reader);
}

static inline std::shared_ptr<SinkReader> getSinkSharedReader(std::shared_ptr<Reader> reader) {
  return std::dynamic_pointer_cast<SinkReader>(reader);
}
/////////////////////////////

class VS_EXPORT MetadataReader : public virtual Reader {
 public:
  enum class MetadataReadStatusCode {
    // generic states
    Ok,
    ErrorWithStatus,
    // custom reader states
    EndOfFile,
    MoreDataAvailable,
  };

  typedef Result<MetadataReadStatusCode> MetadataReadStatus;

  /**
   * @brief A class that holds time-constant specifications.
   */
  struct VS_EXPORT Spec {
    /**
     * Constructs a spec.
     * @param framerate
     */
    explicit Spec(FrameRate framerate);

    Spec();

    /**
     * Copy from @a spec.
     */
    explicit Spec(const Spec& spec);

    ~Spec();

    /**
     * IMU frame rate
     */
    const FrameRate frameRate;

    /**
     * Returns a display name for the reader. Never assume any format for that. EVER.
     * That means that the only thing that's allowed with that is to diplay it to the user.
     * No testing for equality, parsing...
     * Thread safe.
     * @param os Sink for the display name.
     */
    void getDisplayName(std::ostream& os) const;

    /**
     * Reader implementors: use that to set the display name. You're free to put anything, and change it. Make it small.
     * @param name The display name to set.
     */
    void setDisplayName(const char* name);

   private:
    Spec& operator=(const Spec&) = delete;

    class Impl;
    Impl* const pimpl;
  };

 public:
  virtual ~MetadataReader();

  virtual Status readIMUSamples(std::vector<VideoStitch::IMU::Measure>& imuData) = 0;

  /**
   * Exposure, WhiteBalance, ToneCurve return values:
   * `Ok` if one sample was read, and there are no more samples available in the reader
   * `ErrorWithStatus` if a generic error was encountered
   * `EndOfFile` if the stream has stopped and there are no more samples available in the reader
   * `MoreDataAvailable` if one sample was read, and more data is available. the read function should be called again.
   */
  virtual MetadataReadStatus readExposure(std::map<videoreaderid_t, Metadata::Exposure>& exposure) = 0;
  virtual MetadataReadStatus readWhiteBalance(std::map<videoreaderid_t, Metadata::WhiteBalance>& whiteBalance) = 0;
  virtual MetadataReadStatus readToneCurve(std::map<videoreaderid_t, Metadata::ToneCurve>& toneCurve) = 0;

  /**
   * Returns the (const) reader spec.
   */
  const Spec& getSpec() const { return spec; }

  /**
   * Returns the reader spec.
   */
  Spec& getSpec() { return spec; }

 protected:
  /**
   * Constructor.
   * @param framerate
   * @note Everything must be known at creation time.
   */
  explicit MetadataReader(FrameRate framerate);

 private:
  MetadataReader();
  explicit MetadataReader(const MetadataReader&);

  Spec spec;
};

class VS_EXPORT SinkReader : public virtual IO::Sink<Reader> {
 protected:
  explicit SinkReader();
};
}  // namespace Input
}  // namespace VideoStitch
