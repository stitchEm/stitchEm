// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "status.hpp"
#include "frame.hpp"
#include "audio.hpp"

#include <algorithm>
#include <vector>

namespace VideoStitch {
namespace Ptv {
class Value;
}
namespace Output {
class Output;
}
namespace Input {
struct ProbeResult;
class Reader;
}  // namespace Input

namespace Plugin {

/**
 * Loads plugins from the given directory, adds them to available plugins, and returns the number of plugins that were
 * added. This can be called with several directories to load multiple sets of plugins. If several registered plugins
 * match (handle) a config, the last matching plugin will be used. Thread-safe.
 * @param dir: Directory where to look for plugins.
 */
VS_EXPORT int loadPlugins(const std::string& dir);

/**
 * Instance management with std::vector<>.
 *
 * All instances are referenced in instances. Use it if you want to
 * loop on all instances of T.
 *
 * Usage: class MyClass : public VectorInstanceManaged<MyClass> {};
 *
 * In order to loop on all instances:
 * for(MyClass::InstanceVector::const_iterator ... ) {}
 */
template <class T>
class VS_EXPORT VectorInstanceManaged {
 public:
  /**
   * Instance vector type.
   */
  typedef std::vector<T*> InstanceVector;

  virtual ~VectorInstanceManaged() {
    if (instances().empty()) {
      return;
    }
    typename InstanceVector::iterator lIt = std::find(instances().begin(), instances().end(), this);
    if (lIt != instances().end()) {
      instances().erase(lIt);
    }
  }

  /**
   * Returns the vector of all instances.
   */
  static const InstanceVector& Instances() { return instances(); }

 protected:
  explicit VectorInstanceManaged(T* object) { instances().push_back(object); }

 private:
  static InstanceVector& instances();
};

/**
 * Class that wraps a static plugin spec to make it a plugin.
 * T can be either VSReaderPluginBase or VSWriterPluginBase.
 */
template <typename T>
class VSPlugin : public T, public VectorInstanceManaged<VSPlugin<T>> {
 public:
  /**
   * Object type.
   */
  typedef typename T::Object Object;

  /**
   * Config type.
   */
  typedef typename T::Config Config;

  /**
   * 'handles' function pointer type.
   */
  typedef bool (*HandlesFnT)(const Ptv::Value*);

  /**
   * Factory function pointer type.
   */
  typedef Potential<Object>* (*CreateFnT)(const Ptv::Value*, Config);

  /**
   * Contructor.
   * @param name Plugin name.
   * @param handlesFn callback for 'handles'
   * @param createFn factory function
   */
  VSPlugin(const char* name, HandlesFnT handlesFn, CreateFnT createFn)
      : VectorInstanceManaged<VSPlugin<T>>(this), name(name), handlesFn(handlesFn), createFn(createFn) {}

  /**
   * Object 'handles' implementation.
   */
  bool handles(const Ptv::Value* config) const { return handlesFn(config); }

  /**
   * Object factory implementation.
   */
  Potential<Object>* create(const Ptv::Value* config, Config runtime) const { return createFn(config, runtime); }

  /**
   * Returns the name of the plugin.
   */
  const std::string& getName() const { return name; }

 private:
  VSPlugin(const VSPlugin&) = delete;
  VSPlugin& operator=(const VSPlugin&) = delete;

  const std::string name;
  const HandlesFnT handlesFn;
  const CreateFnT createFn;
};

/** ****************************************************************
 * Base class for discovery plugins.
 * The discovery plugin enables auto-detecting input signal characteristics
 * and discovering the capture device's capabilities, in order to build
 * easily a configuration for its associated reader plugin.
 */

/**
 * Struct used and managed by the discovery plugins.
 * It represents a logical device that can be used for capture or playback.
 */
struct DiscoveryDevice {
  /**
   * Capture, Playback or both
   */
  enum Type { UNKNOWN = 0x00, CAPTURE = 0x01, PLAYBACK = 0x02, CAPTURE_AND_PLAYBACK = CAPTURE | PLAYBACK };

  /**
   * Audio or Video
   */
  enum class MediaType {
    UNKNOWNN = 0x00,
    AUDIO = 0x01,
    VIDEO = 0x02,
    AUDIO_AND_VIDEO = MediaType::AUDIO | MediaType::VIDEO
  };

  MediaType mediaType = MediaType::UNKNOWNN;

  /**
   * Type of the device.
   */
  Type type;

  /**
   * DiscoveryDevice name to be used in configuration files and by reader/writer plugins.
   */
  std::string name;

  /**
   * Human readable name to be used for display.
   */
  std::string displayName;

  /**
   * Compare 2 devices for equality
   * @param lhs First device to compare
   * @param rhs Second device to compare
   */
  friend bool operator==(const DiscoveryDevice& lhs, const DiscoveryDevice& rhs) {
    return lhs.type == rhs.type && lhs.name == rhs.name && lhs.displayName == rhs.displayName;
  }
};

/**
 * Display characteristics for a discovery plugin.
 */
struct DisplayMode {
  /**
   * Define a display mode.
   * @param width Resolution
   * @param height Resolution
   * @param interleaved Interleaved vs planar
   * @param framerate Frame rate
   */
  DisplayMode(int64_t width, int64_t height, bool interleaved, FrameRate framerate, bool psf = false)
      : width(width), height(height), interleaved(interleaved), framerate(framerate), psf(psf) {}

  /**
   * Default constructor for containers.
   */
  DisplayMode() : width(0), height(0), interleaved(false), framerate({1, 1}), psf(false) {}

  /**
   * Display width.
   */
  int64_t width;
  /**
   * Display height.
   */
  int64_t height;
  /**
   * True if interleaved.
   */
  bool interleaved;
  /**
   * Display framerate.
   */
  FrameRate framerate;

  /**
   * True if it's progressive segmented frame. When this is true, interleaved should be false.
   */
  bool psf;

  /**
   * Compare 2 display modes
   * @param lhs First display mode to compare
   * @param rhs Second display mode to compare
   */
  friend bool operator<(const DisplayMode& lhs, const DisplayMode& rhs) {
    if (lhs.width != rhs.width) {
      return lhs.width < rhs.width;
    }

    if (lhs.height != rhs.height) {
      return lhs.height < rhs.height;
    }

    if (lhs.framerate.num != rhs.framerate.num || lhs.framerate.den != rhs.framerate.den) {
      return lhs.framerate.num * rhs.framerate.den < rhs.framerate.num * lhs.framerate.den;
    }

    if (lhs.interleaved != rhs.interleaved) {
      return lhs.interleaved == true;
    }

    if (lhs.psf != rhs.psf) {
      return lhs.psf == true;
    }
    return false;
  }

  /**
   * Compare 2 display modes for equality
   * @param lhs First display mode to compare
   * @param rhs Second display mode to compare
   */
  friend bool operator==(const DisplayMode& lhs, const DisplayMode& rhs) {
    return lhs.width == rhs.width && lhs.height == rhs.height && lhs.framerate.num == rhs.framerate.num &&
           lhs.framerate.den == rhs.framerate.den && lhs.interleaved == rhs.interleaved && lhs.psf == rhs.psf;
  }

  /**
   * Check if current display mode can support another one
   * @param rhs display mode to support
   */
  bool canSupport(const DisplayMode& rhs) {
    if ((width < rhs.width) || (height < rhs.height)) {
      return false;
    }

    if (framerate.num * rhs.framerate.den < rhs.framerate.num * framerate.den) {
      return false;
    }

    if (interleaved != rhs.interleaved) {
      return false;
    }

    if (psf != rhs.psf) {
      return false;
    }
    return true;
  }
};

/**
 * Autodetection event handler.
 */
class AutoDetection {
 public:
  virtual ~AutoDetection();
  /**
   * Called when a device is lost.
   * @param device the concerned device.
   */
  virtual void signalLost(const DiscoveryDevice& device) = 0;
  /**
   * Called when a device is detected.
   * @param device the concerned device.
   * @param displayMode display mode.
   */
  virtual void signalDetected(const DiscoveryDevice& device, const DisplayMode& displayMode) = 0;
};

/**
 * VS external Discovered plugin
 */
class VS_EXPORT VSDiscoveryPlugin : public VectorInstanceManaged<VSDiscoveryPlugin> {
 public:
  /**
   * 'discover' function pointer type.
   */
  typedef VSDiscoveryPlugin* (*DiscoverFnT)();

  VSDiscoveryPlugin() : VectorInstanceManaged<VSDiscoveryPlugin>(this) {}

  virtual ~VSDiscoveryPlugin();

  /**
   * Type of the devices managed by this plugin.
   */
  virtual std::string name() const = 0;

  /**
   *  More readable name of the devices managed by this plugin, to be used in GUI interfaces.
   */
  virtual std::string readableName() const = 0;

  /**
   * Lists all available capture devices.
   */
  virtual std::vector<DiscoveryDevice> inputDevices() = 0;
  /**
   * Lists all available playback devices.
   */
  virtual std::vector<DiscoveryDevice> outputDevices() = 0;
  /**
   * Lists all available devices.
   */
  std::vector<DiscoveryDevice> devices();
  /**
   * Lists all available cards.
   */
  virtual std::vector<std::string> cards() const = 0;

  /**
   * If the signal can be autodetected, install the handler.
   * @param handler event handler
   */
  virtual void registerAutoDetectionCallback(AutoDetection& handler) = 0;

  /**
   * Lists all supported display modes, regardless of the display format.
   * @param device the concerned device.
   */
  virtual std::vector<DisplayMode> supportedDisplayModes(const DiscoveryDevice& device) = 0;

  /**
   * Returns the display mode for the current video signal.
   * @param device the concerned device.
   */
  virtual DisplayMode currentDisplayMode(const DiscoveryDevice& device) = 0;

  /**
   * Lists all supported pixel formats, regardless of the display modes.
   * @param device the concerned device.
   */
  virtual std::vector<PixelFormat> supportedPixelFormat(const DiscoveryDevice& device) = 0;

  /**
   * List all supported HW/SW video codecs supported.
   */
  virtual std::vector<std::string> supportedVideoCodecs() { return std::vector<std::string>(); }

  /**
   * Lists all supported audio channels layouts.
   * @param device the concerned device.
   */
  virtual std::vector<int> supportedNbChannels(const DiscoveryDevice& device) = 0;

  /**
   * Lists all supported audio sampling rates, eg. 44100Hz and 48000Hz.
   * @param device the concerned device.
   */
  virtual std::vector<Audio::SamplingRate> supportedSamplingRates(const DiscoveryDevice& device) = 0;

  /**
   * Lists all supported audio samples formats, and whether they are planar/interleaved.
   * @param device the concerned device.
   */
  virtual std::vector<Audio::SamplingDepth> supportedSampleFormats(const DiscoveryDevice& device) = 0;
};

/** ****************************************************
 * Base class for reader plugins.
 */

/**
 * Runtime config.
 */
struct ReaderConfig {
  /**
   * Creates a config.
   * @param id id
   * @param ff First frame
   * @param lf last frame.
   * @param w input width.
   * @param h input height.
   */
  ReaderConfig(readerid_t id, int ff, int lf, int64_t w, int64_t h)
      : id(id), targetFirstFrame(ff), targetLastFrame(lf), width(w), height(h) {}

  /**
   * Index of the input in the panorama list.
   */
  readerid_t id;

  /**
   * First frame.
   */
  int targetFirstFrame;
  /**
   * Last frame.
   */
  int targetLastFrame;
  /**
   * Input width.
   */
  int64_t width;
  /**
   * Input height.
   */
  int64_t height;
};

/**
 * VS external reader plugin representation
 */
class VS_EXPORT VSReaderPluginBase {
 public:
  /**
   * Object type.
   */
  typedef Input::Reader Object;
  /**
   * Object Reader configuration.
   */
  typedef ReaderConfig Config;

  virtual ~VSReaderPluginBase();
};

/**
 * VS Reader plugin type.
 */
typedef VSPlugin<VSReaderPluginBase> VSReaderPlugin;

/**
 * Reader plugin able to probe.
 */
class VS_EXPORT VSProbeReaderPlugin : public VSReaderPlugin, public VectorInstanceManaged<VSProbeReaderPlugin> {
  typedef VectorInstanceManaged<VSProbeReaderPlugin> InstanceManagedType;

 public:
  /**
   * Readability typedef.
   */
  typedef Input::ProbeResult ProbeResult;
  /**
   * The type of a probe callback.
   */
  typedef ProbeResult (*ProbeFnT)(const std::string&);
  /**
   * 'handles' function pointer type.
   */
  typedef bool (*HandlesFnT)(const Ptv::Value*);
  /**
   * Factory function pointer type.
   */
  typedef Potential<Object>* (*CreateFnT)(const Ptv::Value*, ReaderConfig);

  /**
   * Creates a plugin.
   * @param name the plugin id
   * @param handlesFn handles callback
   * @param createFn create callback
   * @param pProbeFn probe callback.
   */
  VSProbeReaderPlugin(char const* name, HandlesFnT handlesFn, CreateFnT createFn, ProbeFnT pProbeFn);

  /**
   * Probes.
   * @param p What to probe.
   */
  ProbeResult probe(const std::string& p) const;

  /** Name Disambiguation between
      VectorInstanceManaged<VSProbeReaderPlugin> and
      VectorInstanceManaged<VSReaderPlugin>. */
  // @{
  typedef InstanceManagedType::InstanceVector InstanceVector;
  static const InstanceVector& Instances() { return InstanceManagedType::Instances(); }
  // @}
 private:
  /**
   * The probe callback.
   */
  const ProbeFnT probeFn;
};

/** ****************************************
 * Base class for writer plugins.
 */

/**
 * Runtime config.
 */
struct WriterConfig {
  /**
   * Creates a config.
   * @param name the plugin id
   * @param w Writer width.
   * @param h Writer height.
   * @param fr System frame rate.
   * @param rate Audio sampling rate
   * @param depth Audio sample size
   * @param layout Audio channels layout
   */
  WriterConfig(const std::string& name, unsigned w, unsigned h, FrameRate fr, const Audio::SamplingRate rate,
               const Audio::SamplingDepth depth, const Audio::ChannelLayout layout)
      : name(name), width(w), height(h), framerate(fr), rate(rate), depth(depth), layout(layout) {}

  /**
   * Callback identifier.
   */
  std::string name;
  /**
   * Output width.
   */
  unsigned width;
  /**
   * Output height.
   */
  unsigned height;
  /**
   * Frame Rate.
   */
  FrameRate framerate;
  /**
   * Audio sampling rate.
   */
  const Audio::SamplingRate rate;
  /**
   * Audio sample size.
   */
  const Audio::SamplingDepth depth;
  /**
   * Audio channels layout.
   */
  const Audio::ChannelLayout layout;
};

/**
 * VS external writer plugin representation
 */
class VS_EXPORT VSWriterPluginBase {
 public:
  /**
   * Object type.
   */
  typedef Output::Output Object;
  /**
   * Writer configuration
   */
  typedef WriterConfig Config;

  virtual ~VSWriterPluginBase();
};

/**
 * VS Writer plugin type.
 */
typedef VSPlugin<VSWriterPluginBase> VSWriterPlugin;
}  // namespace Plugin
}  // namespace VideoStitch
