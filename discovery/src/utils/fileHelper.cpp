// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libgpudiscovery/fileHelper.hpp"

namespace VideoStitch {
namespace FileHelper {
bool fileExists(const std::string &filename) {
  std::ifstream file(filename.c_str());
  bool fileExists = file.good();
  file.close();
  return fileExists;
}
}  // namespace FileHelper
}  // namespace VideoStitch
