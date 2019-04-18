// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef EXTENSIONCHECKER_HPP_
#define EXTENSIONCHECKER_HPP_

#include "libvideostitch/ptv.hpp"

#include <algorithm>

// ending of path::tolower == lowercaseExtension
bool hasExtension(std::string const& path, std::string const& lowercaseExtension) {
  if (path.length() >= lowercaseExtension.length()) {
    std::string lowercasePathEnding(lowercaseExtension.length(), 0);
    std::transform(path.end() - lowercaseExtension.length(), path.end(), lowercasePathEnding.begin(), ::tolower);

    return (lowercasePathEnding == lowercaseExtension);
  } else {
    return false;
  }
}

/**
 * Some readers need to read a string from config.
 * \returns 0 if config is not a string.
 */
std::string const* hasStringContent(VideoStitch::Ptv::Value const* config) {
  std::string const* l_return = 0;
  if (config && config->getType() == VideoStitch::Ptv::Value::STRING) l_return = &config->asString();
  return l_return;
}

#endif  //  EXTENSIONCHECKER_HPP_
