// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "io.hpp"

#ifdef _MSC_VER
#include <codecvt>
#endif

namespace VideoStitch {
namespace Io {

FILE* openFile(const std::string& filename, const std::string& mode) {
#ifdef _MSC_VER
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  std::wstring wideFilename;
  try {
    wideFilename = converter.from_bytes(filename);
  } catch (...) {
    // In case is already encoded.
    return fopen(filename.c_str(), mode.c_str());
  }
  const std::wstring wideMode = converter.from_bytes(mode);
  return _wfopen(wideFilename.c_str(), wideMode.c_str());
#else
  return fopen(filename.c_str(), mode.c_str());
#endif
}
}  // namespace Io
}  // namespace VideoStitch
