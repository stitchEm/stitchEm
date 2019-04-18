// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "status.hpp"
#include "frame.hpp"
#include "audio.hpp"
#include "outputEventManager.hpp"

#include <string>
#include <vector>
#include <memory>

namespace VideoStitch {
namespace Ptv {
class Value;
}

namespace Output {

class VideoWriter;
class AudioWriter;

/**
 * @brief Base class for all client's callbacks.
 */
class VS_EXPORT Output {
 public:
  /**
   * Type-casting to video callback
   * Might return null for callbacks with restricted capabilities.
   */
  VideoWriter* getVideoWriter() const;
  /**
   * Type-casting to audio callback
   * Might return null for callbacks with restricted capabilities.
   */
  virtual AudioWriter* getAudioWriter() const;

  virtual ~Output();

  /**
   * @return The identifier of this callback.
   */
  std::string getName() const { return name; }

  /**
   * Initialization function.
   * Set the subscribers before calling that function if your
   * implementation is meant to emit Events at startup.
   */
  virtual void init() {}

  OutputEventManager& getOutputEventManager();

 protected:
  /**
   * Constructor.
   * @param nameParam The writer id.
   */
  explicit Output(const std::string& nameParam);

  OutputEventManager outputEventManager;

 private:
  char name[2048];

  Output() = delete;
  Output(const Output&) = delete;
};

/**
 * The basic config elements for an Output.
 */
struct VS_EXPORT BaseConfig {
  BaseConfig();

  /**
   * Returns true on success.
   */
  Status parse(const Ptv::Value& config);

  /**
   * Resets to default values.
   */
  void clear();

  /**
   * The format string, aka the type of this output.
   * Can be a registered output plugin or "null" to discard the output.
   */
  char strFmt[2048];

  /**
   * The base filename for this output.
   * Semantics depend on the plugin (see strFmt).
   */
  char baseName[2048];

  /**
   * The desired number of digits.
   */
  int numberNumDigits;

  /**
   * The dowsampling factor respective to the pano size.
   * 2 means a size of a fourth (downsampled twice on every dimension).
   */
  int downsamplingFactor;
};

}  // namespace Output
}  // namespace VideoStitch
