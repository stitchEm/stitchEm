// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "output.hpp"
#include "audio.hpp"  // XXX TODO FIXME remove me

#include <vector>
#include <thread>
#include <atomic>

#ifndef SWIG
EXPIMP_TEMPLATE template class VS_EXPORT std::vector<std::pair<int, const int32_t*>>;
#endif

namespace VideoStitch {
namespace Util {
class OpaquePtr;
class OnlineAlgorithm;
}  // namespace Util
class Status;

namespace GPU {
class Surface;
class Overlayer;
}  // namespace GPU

namespace Output {

/**
 * Creates a writer from the given configuration given as a Ptv::Value.
 * Available options depend on the type.
 *
 * @param config The writer configuration
 * @param name The writer id.
 * @param width Output width
 * @param height Output height
 * @param framerate Frame rate
 * @param rate Audio sampling rate
 * @param depth Audio sample size
 * @param layout Audio channels layout
 *
 * @return the Output, or an error if the name is not unique.
 */
Potential<Output> VS_EXPORT create(const Ptv::Value& config, const std::string& name, unsigned width, unsigned height,
                                   FrameRate framerate, Audio::SamplingRate rate = Audio::SamplingRate::SR_NONE,
                                   Audio::SamplingDepth depth = Audio::SamplingDepth::SD_NONE,
                                   Audio::ChannelLayout layout = Audio::STEREO);

/**
 * @brief A common interface to be implemented by all output writers.
 */
class VS_EXPORT VideoWriter : public virtual Output {
 public:
  virtual ~VideoWriter();

  /**
   * Push a video frame to the output.
   * @param videoFrame Video frame to be pushed.
   *                   IMPORTANT : it will be valid only during the duration of the call.
   */
  virtual void pushVideo(const Frame& videoFrame) = 0;

  /**
   * Get the width of the callback display.
   * @return The frame width.
   */
  unsigned getWidth() const { return width; }

  /**
   * Get the width of the panorama.
   * @return The underlying panorama width.
   */
  unsigned getPanoWidth() const { return width; }

  /**
   * Get the height of the callback.
   * @return The frame height.
   */
  unsigned getHeight() const { return height; }

  /**
   * Get the height of the panorama.
   * @return The underlying panorama height.
   */
  unsigned getPanoHeight() const { return height; }

  /**
   * Get the frame rate of the output (same as the Controller).
   * @return The frame rate.
   */
  FrameRate getFrameRate() const { return framerate; }

  /**
   * Get the format prefered by this callback.
   * @return The pixel format for this callback.
   */
  VideoStitch::PixelFormat getPixelFormat() const { return pixelFormat; }

  /**
   * Returns the expected data size for pushFrame().
   */
  int64_t getExpectedFrameSize() const;

  /**
   * Hardware area of expected memory
   */
  AddressSpace getExpectedOutputBufferType() const { return outputType; }

  /**
   * Returns the expected data size for pushFrame(), for a reader of the given size and type.
   * @param format Pixel format.
   * @param width Output width
   * @param height Output height
   */
  static int64_t getExpectedFrameSizeFor(VideoStitch::PixelFormat format, int64_t width, int64_t height);

  /**
   * Returns the writer latency in ms.
   */
  mtime_t getLatency() const { return latency; }

  /**
   * Update writer configuration
   */
  virtual void updateConfig(const Ptv::Value&) {}

 protected:
  /**
   * Constructor.
   * @param width The output width.
   * @param height The output height.
   * @param framerate Frame rate of the system
   * @param pixelFormat The pixel format.
   * @param outputType Memory location of the output buffer
   * @param rate Audio sampling rate
   * @param depth Audio sample size
   * @param layout Audio channels layout
   */
  VideoWriter(unsigned width, unsigned height, FrameRate framerate, VideoStitch::PixelFormat pixelFormat,
              AddressSpace outputType = Host);

  std::atomic<mtime_t> latency;

 private:
  const unsigned width;
  const unsigned height;
  FrameRate framerate;
  const VideoStitch::PixelFormat pixelFormat;
  const AddressSpace outputType;
};

/**
 * @brief A common interface to be implemented by all output writers.
 */
class VS_EXPORT AudioWriter : public virtual Output {
 public:
  virtual ~AudioWriter();

  /**
   * Push a frame to the output.
   * @param audioSamples Audio samples synchronous to the frame.
   */
  virtual void pushAudio(Audio::Samples& audioSamples) = 0;

  /**
   * Audio sampling rate
   */
  Audio::SamplingRate getSamplingRate() const { return rate; }

  /**
   * Audio sample size
   */
  Audio::SamplingDepth getSamplingDepth() const { return depth; }

  /**
   * Audio channels layout
   */
  Audio::ChannelLayout getChannelLayout() const { return layout; }

 protected:
  /**
   * Constructor.
   * @param rate Audio sampling rate
   * @param depth Audio sample size
   * @param layout Audio channels layout
   */
  AudioWriter(const Audio::SamplingRate rate, const Audio::SamplingDepth depth, const Audio::ChannelLayout layout);

 private:
  const Audio::SamplingRate rate;
  const Audio::SamplingDepth depth;
  const Audio::ChannelLayout layout;
};

/**
 * @brief A common interface to be implemented by all stereoscopic output writers.
 */
class VS_EXPORT StereoWriter : public virtual Output {
 public:
  /**
   * How the output should be laid out:
   */
  enum Layout {
    // Images are on top of each other.
    VerticalLayout,
    // Images are side-by-side.
    HorizontalLayout,
  };

  /**
   * Creates a stereoscopic writer from a simple writer.
   * This writer will compose the left and right eyes according to the specified layout,
   * then push the result to the wrapped writer.
   * Available options depend on the type.
   *
   * @param writer The wrapped writer. We take ownership.
   * @param layout The layout for the composition
   * @param buffer The location of the frame buffer
   *
   * @return the Writer, or an error if the name is not unique.
   */
  static Potential<StereoWriter> createComposition(VideoWriter* writer, Layout layout, AddressSpace buffer);

  /**
   * Creates a stereoscopic writer from a simple writer.
   * This writer will compose the left and right eyes according to the specified layout,
   * then push the result to the wrapped writer.
   * Available options depend on the type.
   *
   * @param writer The wrapped writer. We take ownership.
   * @param eye The eye selected
   * @param buffer The location of the frame buffer
   *
   * @return the Writer, or an error if the name is not unique.
   */
  static Potential<StereoWriter> createSelection(VideoWriter* writer, Eye eye, AddressSpace buffer);

  /**
   * Creates a stereoscopic writer from a simple writer.
   * This writer will create an anaglyph image for red-cyan glasses from the left and right eyes,
   * then push the result to the wrapped writer.
   * Available options depend on the type.
   *
   * @param writer The wrapped writer. We take ownership.
   * @param buffer The location of the frame buffer
   * @param device The GPU on which to produce the anaglyph image if not on host.
   *
   * @return the Writer, or an error if the name is not unique.
   */
  static Potential<StereoWriter> createAnaglyphColor(VideoWriter* writer, AddressSpace buffer, int device = -1);

  virtual ~StereoWriter();

  /**
   * Push an eye's frame to the output.
   * @param eye The eye to push to.
   * @param frame Video frame to be pushed to the eye.
   */
  virtual void pushEye(Eye eye, const Frame& frame) = 0;

  /**
   * Push an audio frame to the output.
   * @param audioSamples Audio samples.
   */
  virtual void pushAudio(Audio::Samples& audioSamples) = 0;

  /**
   * Get the width of the callback display.
   * @return The frame width.
   */
  unsigned getWidth() const { return width; }

  /**
   * Get the width of the panorama.
   * @return The underlying panorama width.
   */
  unsigned getPanoWidth() const { return panoWidth; }

  /**
   * Get the height of the callback.
   * @return The frame height.
   */
  unsigned getHeight() const { return height; }

  /**
   * Get the height of the panorama.
   * @return The underlying panorama height.
   */
  unsigned getPanoHeight() const { return panoHeight; }

  /**
   * Get the frame rate of the output (same as the Controller).
   * @return The frame rate.
   */
  FrameRate getFrameRate() const { return framerate; }

  /**
   * Get the format prefered by this callback.
   * @return The pixel format for this callback.
   */
  VideoStitch::PixelFormat getPixelFormat() const { return format; }

  /**
   * Hardware area of expected memory
   */
  AddressSpace getExpectedOutputBufferType() const { return outputType; }

  /**
   * Returns the expected data size for pushFrame().
   */
  int64_t getExpectedFrameSize() const;

  /**
   * Update writer configuration
   */
  virtual void updateConfig(const Ptv::Value&) {}

 protected:
  /**
   * Constructor.
   * @param width The output width.
   * @param height The output height.
   * @param framerate The system frame rate
   * @param panoWidth The underlying panorama width.
   * @param panoHeight The underlying panorama height.
   * @param pixelFormat The pixel format.
   * @param outputType Memory location of the output buffers
   * @param rate Audio sampling rate
   * @param depth Audio sample size
   * @param layout Audio channels layout
   */
  StereoWriter(unsigned width, unsigned height, FrameRate framerate, unsigned panoWidth, unsigned panoHeight,
               VideoStitch::PixelFormat pixelFormat, AddressSpace outputType);

  const unsigned width;             ///< Output width
  const unsigned height;            ///< Output height
  FrameRate framerate;              ///< System frame rate
  const unsigned panoWidth;         ///< Underlying panorama width
  const unsigned panoHeight;        ///< Underlying panorama height
  VideoStitch::PixelFormat format;  ///< Pixel format
  const AddressSpace outputType;    ///< Memory location of the output buffers
};

}  // namespace Output

namespace Core {

class SourceRenderer;
class PanoRenderer;

class PanoramaDefinitionUpdater;
class PanoDefinition;

/**
 * @brief An output class to run an online algorithm.
 */
class VS_EXPORT AlgorithmOutput {
 public:
  /**
   * Event handler.
   */
  struct Listener {
    /**
     * Called when the algo is done running successfully.
     * @param pano Result.
     */
    virtual void onPanorama(PanoramaDefinitionUpdater& pano) = 0;

    /**
     * Called on error.
     * @param status error status.
     */
    virtual void onError(const Status& status) = 0;

    /**
     * @brief ~Listener Virual destructor
     */
    virtual ~Listener() {}
  };

  /**
   * Creates.
   * @param algo algorithm to run. takes ownership.
   * @param pano input pano
   * @param handler see above
   * @param opaque For internal use.
   */
  AlgorithmOutput(Util::OnlineAlgorithm* algo, const PanoDefinition& pano, Listener& handler, Util::OpaquePtr** opaque);

  virtual ~AlgorithmOutput();

  /**
   * Cancels the current algorithm.
   * @note The algorithm is not really cancelled, but it will not call the listener when finished.
   */
  void cancel() { cancelled = true; }

  /**
   * Internal.
   * @param inputFrames reserved.
   */
  void onFrame(std::vector<std::pair<videoreaderid_t, GPU::Surface&>>& inputFrames, mtime_t date, FrameRate frameRate);

 private:
  static void asyncJob(AlgorithmOutput*, std::vector<std::pair<videoreaderid_t, GPU::Surface&>>, mtime_t date,
                       FrameRate frameRate);

  std::unique_ptr<Util::OnlineAlgorithm> algo;
  std::unique_ptr<PanoramaDefinitionUpdater> panoramaUpdater;
  Listener& listener;
  std::thread* worker;
  Util::OpaquePtr** ctx;
  std::atomic<bool> cancelled;

  AlgorithmOutput();
  AlgorithmOutput(const AlgorithmOutput&);
  const AlgorithmOutput& operator=(const AlgorithmOutput&);
};

/**
 * @brief An output class registering the user callbacks for receiving
 * video data.
 */
class VS_EXPORT ExtractOutput {
 public:
  class Pimpl;

 public:
  virtual ~ExtractOutput();

  /**
   * The OpenGL rendering callbacks.
   * @return false if two callbacks to be installed have identical identifiers.
   */
  bool setRenderers(const std::vector<std::shared_ptr<SourceRenderer>>&);

  /**
   * Install a single OpenGL callback without removing the ones already installed.
   * @return false if the identifier is not unique and the callback couldn't be installed.
   */
  bool addRenderer(std::shared_ptr<SourceRenderer>);

  /**
   * Remove an OpenGL renderer by its identifier.
   * @return false if this callback is not installed yet.
   */
  bool removeRenderer(const std::string&);

  /**
   * The video input callbacks. We take ownership.
   * The previous active writers are disabled, and their pointers are invalidated.
   * @return false if two callbacks to be installed have identical identifiers.
   */
  bool setWriters(const std::vector<std::shared_ptr<Output::VideoWriter>>&);

  /**
   * Install a single callback without removing the ones already installed.
   * As usual, ownership is transferred to the StitchOutput, forget about this pointer
   * afterward.
   * @return false if the identifier is not unique and the callback couldn't be installed.
   */
  bool addWriter(std::shared_ptr<Output::VideoWriter>);

  /**
   * Remove a callback by its identifier.
   * @return false if this callback is not installed yet.
   */
  bool removeWriter(const std::string&);

  /**
   * update a video callback by its identifier.
   * @return false if this callback is not installed yet.
   */
  bool updateWriter(const std::string&, const Ptv::Value&);

  /**
   * Implementation. Internal.
   */
  Pimpl* const pimpl;
  /**
   * Create with impl @a pimpl. Internal.
   */
  explicit ExtractOutput(Pimpl*);

 protected:
  ExtractOutput();

 private:
  ExtractOutput(const ExtractOutput&);
  const ExtractOutput& operator=(const ExtractOutput&);
};

/**
 * @brief An output class registering the user callbacks for receiving
 * panoramic audio and video data.
 */
template <class VideoWriter>
class VS_EXPORT StitcherOutput {
 public:
  typedef VideoWriter Writer;
  class Pimpl;

 public:
  virtual ~StitcherOutput();

  /**
   * The OpenGL rendering callbacks.
   * @return false if two callbacks to be installed have identical identifiers.
   */
  bool setRenderers(const std::vector<std::shared_ptr<PanoRenderer>>&);

  /**
   * Install a single OpenGL callback without removing the ones already installed.
   * @return false if the identifier is not unique and the callback couldn't be installed.
   */
  bool addRenderer(std::shared_ptr<PanoRenderer>);

  /**
   * Remove an OpenGL renderer by its identifier.
   * @return false if this callback is not installed yet.
   */
  bool removeRenderer(const std::string&);

  /**
   * The OpenGL compositing callback.
   */
  void setCompositor(const std::shared_ptr<GPU::Overlayer>&);

  /**
   * The video input callbacks. We take ownership.
   * The previous active writers are disabled, and their pointers are invalidated.
   * @return false if two callbacks to be installed have identical identifiers.
   */
  virtual bool setWriters(const std::vector<std::shared_ptr<Writer>>&);

  /**
   * Install a single video callback without removing the ones already installed.
   * As usual, ownership is transferred to the StitchOutput, forget about this pointer
   * afterward.
   * @return false if the identifier is not unique and the callback couldn't be installed.
   */
  bool addWriter(std::shared_ptr<Writer>);

  /**
   * Remove a video callback by its identifier.
   * @return false if this callback is not installed yet.
   */
  bool removeWriter(const std::string&);

  /**
   * update a video callback by its identifier.
   * @return false if this callback is not installed yet.
   */
  bool updateWriter(const std::string&, const Ptv::Value&);

  /**
   * Implementation. Internal.
   */
  Pimpl* const pimpl;
  /**
   * Create with impl @a pimpl. Internal.
   */
  explicit StitcherOutput(Pimpl*);

 protected:
  StitcherOutput();

 private:
  StitcherOutput(const StitcherOutput&);
  const StitcherOutput& operator=(const StitcherOutput&);
};

typedef class StitcherOutput<Output::VideoWriter> StitchOutput;
typedef class StitcherOutput<Output::StereoWriter> StereoOutput;
}  // namespace Core
}  // namespace VideoStitch
