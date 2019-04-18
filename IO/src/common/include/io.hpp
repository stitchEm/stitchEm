// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <stdio.h>
#include <string>

namespace VideoStitch {
namespace Io {

/**
 * Opens the file whose name is specified in the parameter filename and associates it with a stream that can be
 * identified in future operations by the FILE pointer returned. The operations that are allowed on the stream and how
 * these are performed are defined by the mode parameter. filename and mode can contain unicode characters
 */
FILE* openFile(const std::string& filename, const std::string& mode);

}  // namespace Io
}  // namespace VideoStitch
