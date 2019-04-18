// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "multiFileInput.hpp"

#include "filesystem.hpp"

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"

#include <iostream>
#include <cassert>
#include <limits>
#include <memory>
#include <vector>

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

namespace VideoStitch {
namespace Input {

bool MultiFileReader::checkProbeResult(const Input::ProbeResult& probeResult,
                                       const Plugin::VSReaderPlugin::Config& runtime) {
  return (probeResult.valid && runtime.targetFirstFrame >= probeResult.firstFrame &&
          probeResult.lastFrame >= runtime.targetLastFrame);
}

std::string const* MultiFileReader::hasStringContent(Ptv::Value const* config) {
  std::string const* l_return = 0;
  if (config && config->getType() == Ptv::Value::STRING) l_return = &config->asString();
  return l_return;
}

bool MultiFileReader::matchesFilenameTemplate(const char* filename, const std::string& filenameTemplate, int& frame) {
  frame = -1;
  const size_t templateSize = filenameTemplate.size();
  size_t filenameIndex = 0;
  size_t templateIndex = 0;
  while (filename[filenameIndex] != '\0' && templateIndex < templateSize) {
    if (filenameTemplate[templateIndex] == '%') {
      ++templateIndex;
      if (templateIndex >= templateSize) {
        // Invalid syntax.
        return false;
      }
      if (filenameTemplate[templateIndex] == '%') {
        // Escaped percent sign, continue.
        continue;
      }
      // Conversion specifier, parse it.
      int width = 0;
      for (; templateIndex < templateSize; ++templateIndex) {
        if (filenameTemplate[templateIndex] == 'i' || filenameTemplate[templateIndex] == 's') {
          ++templateIndex;
          break;
        }
        const int digit = filenameTemplate[templateIndex] - (int)'0';
        if (digit < 0 || digit > 9) {
          return false;
        }
        width = 10 * width + digit;
      }
      if (width <= 0) {
        return false;
      }
      // Now check that the next width filename chars are numbers.
      int num09Chars = 0;
      int tmpFrame = 0;
      while (num09Chars < width && filename[filenameIndex] != '\0' && filename[filenameIndex] >= '0' &&
             filename[filenameIndex] <= '9') {
        tmpFrame = 10 * tmpFrame + (filename[filenameIndex] - (int)'0');
        ++num09Chars;
        ++filenameIndex;
      }
      if (num09Chars != width) {
        return false;
      }
      if (!(frame == -1 || frame == tmpFrame)) {
        return false;
      }
      frame = tmpFrame;
    } else if (filename[filenameIndex] != filenameTemplate[templateIndex]) {
      return false;
    }
    ++filenameIndex;
    ++templateIndex;
  }
  // All template input must have been consumed.
  return templateIndex == templateSize;
}

ProbeResult MultiFileReader::probe(const std::string& fileNameTemplate) {
  // Scan all files in the directory and find the min and max matching ones.
  std::string directory;
  std::string filenameT;
  Util::getBaseDir(fileNameTemplate, &directory, &filenameT);
  Util::DirectoryLister lister(directory);
  if (!lister.ok()) {
    Logger::get(Logger::Warning) << "Could not open directory " << directory << " for reading." << std::endl;
    return ProbeResult({false, false, -1, -1, -1, -1, false, false});
  }

  int minMatchingFrame = std::numeric_limits<int>::max();
  int maxMatchingFrame = -1;
  for (; !lister.done(); lister.next()) {
    int frame = -1;
    if (matchesFilenameTemplate(lister.file().c_str(), filenameT, frame)) {
      if (minMatchingFrame > frame) {
        minMatchingFrame = frame;
      }
      if (maxMatchingFrame < frame) {
        maxMatchingFrame = frame;
      }
    }
  }
  if (minMatchingFrame < 0) {
    // Not a template, we had a match without a frame.
    return ProbeResult({true, false, 0, NO_LAST_FRAME, -1, -1, false, true});
  } else if (maxMatchingFrame > 0) {
    return ProbeResult({true, true, minMatchingFrame, maxMatchingFrame, -1, -1, false, true});
  }
  // None matching.
  return ProbeResult({false, false, -1, -1, -1, -1, false, false});
}

MultiFileReader::MultiFileReader(const std::string& fileNameTemplate, const ProbeResult& probeResult, int64_t width,
                                 int64_t height, int64_t frameDataSize, VideoStitch::PixelFormat format)
    : Reader(-1),
      VideoReader(width, height, frameDataSize, format, Host, {60, 1} /*fps*/, int(probeResult.firstFrame),
                  int(probeResult.lastFrame), true /* procedural, frame rate is unknown */, nullptr),
      fileNameTemplate(fileNameTemplate),
      filenameIsTemplate(probeResult.filenameIsTemplate),
      curFrame(0) {}

MultiFileReader::~MultiFileReader() {}

void MultiFileReader::resetDisplayName() { getSpec().setDisplayName(fileNameTemplate.c_str()); }

Status MultiFileReader::seekFrame(frameid_t frameNumber) {
  // TODO return an error code when frame number does not exist
  curFrame = frameNumber;
  return Status::OK();
}

std::string MultiFileReader::filenameFromTemplate(int frame) const {
  if (!filenameIsTemplate) {
    return fileNameTemplate;
  }
  return filenameFromTemplate(fileNameTemplate, frame);
}

std::string MultiFileReader::filenameFromTemplate(const std::string& fileNameTemplate, int frame) {
  std::vector<char> buf(fileNameTemplate.size() + 64 + 1);
  /**
   * inject the current frame in the filename if the filename contains a format
   */
  std::string s2(fileNameTemplate);
  // find a % not followed by a %
  for (size_t i = 0; i < s2.size(); ++i) {
    if (s2[i] == '%') {
      if (i == s2.size() - 1 || s2[i + 1] == '%') {
        ++i;  // double %, skip
        continue;
      } else {
        // find first non-digit conversion specifier, change to i
        for (++i; i < s2.size(); ++i) {
          if (!('0' <= s2[i] && s2[i] <= '9')) {
            s2[i] = 'i';
            break;
          }
        }
        snprintf(buf.data(), buf.size(), s2.c_str(), frame);
        return std::string(buf.data());
      }
    }
  }
  return std::string(buf.data());
}

ReadStatus MultiFileReader::readFrame(mtime_t& timestamp, unsigned char* data) {
  const ReadStatus res = readFrameInternal(data);
  timestamp = (mtime_t)(curFrame++ * 1000000.0 * (double)getSpec().frameRate.den /
                        (double)getSpec().frameRate.num);  // timpestamp in µs
  return res;
}

}  // namespace Input
}  // namespace VideoStitch
