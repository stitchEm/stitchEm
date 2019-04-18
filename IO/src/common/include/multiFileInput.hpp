// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <string>

#include "libvideostitch/input.hpp"
#include "libvideostitch/plugin.hpp"

namespace VideoStitch {
namespace Ptv {
class Value;
}

namespace Input {
/**
 * @brief An abstract helper for readers that read numbered image files instead of video.
 */
class MultiFileReader : public VideoReader {
 public:
  virtual Status seekFrame(frameid_t date);

  /**
   * Some readers need to read a string from config.
   * \returns 0 if config is not a string.
   */
  static std::string const* hasStringContent(Ptv::Value const* config);

 protected:
  /**
   * Probes the given filename template to see if the file range exist and possible detect the last frame.
   * @param fileNameTemplate The file name template. Same format as printf.
   * @return NULL on error. The result must be deleted.
   */
  static ProbeResult probe(const std::string& fileNameTemplate);

  /**
   * Creates a MultiFileReader.
   * @param fileNameTemplate The file name template. Same format as printf.
   * @param probeResult The result of a previous call to probe(). We take ownership of the pointer.
   * @param width See Reader class.
   * @param height See Reader class.
   * @param frameDataSize See Reader class.
   */
  MultiFileReader(const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t width, int64_t height,
                  int64_t frameDataSize, VideoStitch::PixelFormat);

  virtual ~MultiFileReader();

  /**
   * @returns true if and only if probeResult and runtime are
   * consistent for a Reader instantiation.
   */
  static bool checkProbeResult(const Input::ProbeResult& probeResult, const Plugin::VSReaderPlugin::Config& runtime);

  virtual ReadStatus readFrame(mtime_t& timestamp, unsigned char* data);

  /**
   * Returns the file name for the given frame.
   * @param frame The frame whose filename to get.
   */
  std::string filenameFromTemplate(int frame) const;

  /**
   * Static version of the above.
   * @param frame The frame whose filename to get. Has to be a template.
   * @param filenameIsTemplate true if @a fileNameTemplate is a template.
   */
  static std::string filenameFromTemplate(const std::string& fileNameTemplate, int frame);

  /**
   * Returns true if filename mathes the given template.
   * @param filename Filename to test.
   * @param filenameTemplate template to test against
   * @param frame On output, contains the matching frame if the filename matches, -1 else.
   * @return True if the filename matches.
   */
  static bool matchesFilenameTemplate(const char* filename, const std::string& filenameTemplate, int& frame);

  void resetDisplayName();

  const std::string fileNameTemplate;

 protected:
  virtual ReadStatus readFrameInternal(unsigned char* data) = 0;

  /**
   * Returns the display type (e.g. JPEG) of the file.
   * @param os Output stream.
   */
  virtual void getDisplayType(std::ostream& os) const = 0;

  bool filenameIsTemplate;
  int curFrame;
};
}  // namespace Input
}  // namespace VideoStitch
