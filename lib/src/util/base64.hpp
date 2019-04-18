// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// TODO: move to util.

#pragma once

#include <string>
#include <iostream>

namespace VideoStitch {
namespace Util {

/**
 * @brief A helper to encode data to base64.
 */
std::string base64Encode(const std::string& input);

/**
 * @brief A helper to decode data from base64.
 */
std::string base64Decode(const std::string& input);

}  // namespace Util
}  // namespace VideoStitch
