// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "util/base64.hpp"

namespace VideoStitch {
namespace Testing {

void testBase64(std::string input) {
  std::string encoded = Util::base64Encode(input);
  std::string decoded = Util::base64Decode(encoded);
  ENSURE_EQ(input, decoded);
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::testBase64("aFv'(-^757?");
  return 0;
}
