// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "common-config.hpp"

#include <string>

namespace VideoStitch {
namespace Helper {
/**
 * Tries to infer the filename from the extractImages tool.
 * @param filename The source filename.
 * @returns NULL if no pattern was detect, the source name otherwise. No check is performed whether the file actually
 * exists.
 */
std::string VS_COMMON_EXPORT detectExtractImagesPattern(const char *filename);

}  // namespace Helper
}  // namespace VideoStitch
