// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"

#include "input.hpp"
#include "plugin.hpp"

#include <string>
#include <vector>

namespace VideoStitch {
namespace Core {
class ReaderInputDefinition;
}

namespace Input {

/**
 * The result of probing for reader information.
 *
 * Default values should be false, false, -1, -1, -1, -1.
 */
struct ProbeResult {
  /**
   * Whether the probe result is valid.
   */
  bool valid;
  /**
   * Whether the filename is a template, i.e. it takes a frame as parameter.
   */
  bool filenameIsTemplate;
  /**
   * The first available frame.
   */
  frameid_t firstFrame;
  /**
   * The last available frame, inclusive.
   */
  frameid_t lastFrame;
  /**
   * The input width
   */
  int64_t width;
  /**
   * The input height.
   */
  int64_t height;
  /**
   * Wether the input has audio or not
   */
  bool hasAudio;

  /**
   * Wether the input has video or not
   */
  bool hasVideo;
};

/**
 * @brief A Reader factory class used to create readers for use by the stitcher depending on file type.
 */
class VS_EXPORT ReaderFactory {
 public:
  virtual ~ReaderFactory() {}

  /**
   * Creates a reader.
   * @param id The index of this input among the list of all inputs.
   * @param def The configuration of the input.
   */
  virtual Potential<Reader> create(readerid_t id, const Core::ReaderInputDefinition& def) const = 0;

  /**
   * Analyzes the given reader config to guess the underlying info.
   * @param config Reader config. See ReaderFactory::create.
   */
  virtual ProbeResult probe(const Ptv::Value& config) const = 0;
  /**
   * Same as above, but works with a filename.
   * @param filename THis wil lbe converted into a string config.
   */
  ProbeResult probe(const std::string& filename) const;

  /**
   * Returns the first frames the the readers from this factory will be able to generate.
   */
  virtual frameid_t getFirstFrame() const = 0;

  /**
   * Returns the number of frames the the readers from this factory will be able to generate.
   */
  virtual frameid_t getNumFrames() const = 0;
};

/**
 * @brief The default reader factory.
 */
class VS_EXPORT DefaultReaderFactory : public ReaderFactory {
 public:
  /**
   * Create a DefaultReaderFactory.
   * @param firstFrame Expected first frame. Must be >= 0.
   * @param lastFrame Expected last frame. If -1, the reader will try to detect the end.
   */
  DefaultReaderFactory(frameid_t firstFrame, frameid_t lastFrame);
  virtual ~DefaultReaderFactory();

  virtual Potential<Reader> create(readerid_t id, const Core::ReaderInputDefinition& def) const;
  virtual ProbeResult probe(const Ptv::Value& config) const;

  virtual frameid_t getFirstFrame() const;
  virtual frameid_t getNumFrames() const;

 private:
  DefaultReaderFactory(const DefaultReaderFactory&) = delete;
  DefaultReaderFactory& operator=(const DefaultReaderFactory&) = delete;

  const frameid_t firstFrame;
  const frameid_t lastFrame;
};
}  // namespace Input
}  // namespace VideoStitch
