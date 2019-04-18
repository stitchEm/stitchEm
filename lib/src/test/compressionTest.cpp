// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <string>
#include <sstream>
#include <gpu/memcpy.hpp>
#include <util/pngutil.hpp>
#include <util/compressionUtils.hpp>
#include <util/base64.hpp>

//#define DUMP_TEST_RESULT

#if defined(DUMP_TEST_RESULT)
//#undef NDEBUG
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "../util/debugUtils.hpp"
#endif

/*
 * This test is used to check the compression rate of the polyline-based encoding and the PNG compresison
 */

namespace VideoStitch {
namespace Testing {

bool readImageFromFile(const std::string filename, int64_t& width, int64_t& height, std::vector<uint32_t>& data) {
  std::vector<unsigned char> imageBuffer;
  if (!VideoStitch::Util::PngReader::readRGBAFromFile(filename.c_str(), width, height, imageBuffer)) {
    return false;
  }
  data.clear();
  data.resize(width * height, 0);
  for (int i = 0; i < width * height; i++)
    if (imageBuffer[4 * i] > 0 || imageBuffer[4 * i + 1] || imageBuffer[4 * i + 2] > 0) {
      data[i] = 1;
    }
  return true;
}

void testCompression() {
#ifdef DUMP_TEST_RESULT
  std::string workingPath =
      "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/";
#else
  std::string workingPath = "";
#endif

  std::vector<std::string> compressionTests;
  for (int i = 0; i <= 20; i++) {
    compressionTests.push_back(workingPath + "data/compression/" + std::to_string(i) + ".png");
  }
  std::vector<float> compressionRates = {56.71f, 73.03f, 69.57f, 79.81f, 72.33f};
  for (int t = 4; t >= 0; t--) {
    int64_t width, height;
    std::vector<uint32_t> data;
    ENSURE(readImageFromFile(compressionTests[t], width, height, data) == true, "Cannot load input file");
    std::vector<unsigned char> ucdata(data.size());
    for (size_t i = 0; i < data.size(); i++) {
      ucdata[i] = (data[i] > 0 ? 1 : 0);
    }
    Util::PngReader pngReader;
    std::string maskMemory;
    ENSURE(pngReader.writeMaskToMemory(maskMemory, width, height, &ucdata[0]));
    std::string encodedBinary = Util::base64Encode(maskMemory);
    const size_t maskLength = encodedBinary.size();

    std::string contourEncodeds;
    Util::Compression::polylineEncodeBinaryMask((int)width, (int)height, ucdata, contourEncodeds);
    size_t contourLength = contourEncodeds.size();
    std::vector<unsigned char> mask;
    Util::Compression::polylineDecodeBinaryMask((int)width, (int)height, contourEncodeds, mask);
    const float difference = Util::Compression::binaryDifference(ucdata, mask);

#ifdef DUMP_TEST_RESULT
    std::transform(mask.begin(), mask.end(), mask.begin(),
                   [](unsigned char d) -> unsigned char { return (d > 0) ? 255 : 0; });
    pngReader.writeMonochromToFile(std::string(compressionTests[t] + "_extract_poly_reconstructed32.png").c_str(),
                                   width, height, &mask[0]);
#endif
    const float compressionRate = 100.0f - float(contourLength * 100.0f) / maskLength;
    printf("Test %d : Difference %.05f - Binary %zu vs. contour32 %zu : ~difference %.02f\n", t, difference, maskLength,
           contourLength, compressionRate);
    ENSURE(difference < 0.005f, " The difference is too big after decoding");
    ENSURE(compressionRate >= compressionRates[t], "Compression rate got worse");
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char** argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::testCompression();

  return 0;
}
