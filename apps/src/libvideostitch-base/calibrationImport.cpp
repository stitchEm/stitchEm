// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationImport.hpp"

#include <sstream>
#include <cstring>
#include <vector>

namespace VideoStitch {
namespace Helper {

void split(const char* str, char delim, std::vector<std::string>* res) {
  const char* lastP = str;
  for (const char* p = str; *p != '\0'; ++p) {
    if (*p == delim) {
      res->push_back(std::string(lastP, p - lastP));
      lastP = p + 1;
    }
  }
  res->push_back(std::string(lastP));
}

std::string detectExtractImagesPattern(const char* filename) {
  std::string fn(filename);
  if (fn.find('-') == std::string::npos)  // at least one dash
    return "";
  std::vector<std::string> lFn;
  split(filename, '.', &lFn);  //"source/A", "mp4-10", "jpg"
  if (lFn.size() < 3) return "";
  if (lFn[lFn.size() - 2].find('-') == std::string::npos)  // at least one dash
    return "";
  std::vector<std::string> lFnExt;
  split(lFn[lFn.size() - 2].c_str(), '-', &lFnExt);  //"source/A", {"mp4", "10"}, "jpg"
  if (lFnExt.size() != 2) return "";
  std::string frmNum = lFnExt[lFnExt.size() - 1];  //"10"
  int res;
  if (!(std::stringstream(frmNum) >> res))  // sanity check: is a number
    return "";
  lFn.pop_back();            // remove "jpg"
  lFn.pop_back();            // remove "mp4-10"
  lFn.push_back(lFnExt[0]);  // add "mp4"
  std::string srcFn;         //"source/A.mp4"
  for (size_t i = 0; i < lFn.size(); ++i) {
    srcFn += (i + 1 < lFn.size()) ? lFn[i] + "." : lFn[i];
  }
  return srcFn;
}

}  // namespace Helper
}  // namespace VideoStitch
