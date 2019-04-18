// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace VideoStitch {
namespace FileHelper {
/**
 * @brief fileExists checks the existence of a file.
 * @param filename The input full filename.
 * @return true if the file exists, false otherwise.
 */
VS_DISCOVERY_EXPORT bool fileExists(const std::string &filename);
}  // namespace FileHelper
}  // namespace VideoStitch
