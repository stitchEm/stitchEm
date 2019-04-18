// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

// TODO move back to src/test/common once all tests are ported

#include "testing_common.hpp"

#include "backend/common/imageOps.hpp"

#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "util/pngutil.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/imageProcessingUtils.hpp"

#include <cmath>
#include <vector>

namespace VideoStitch {
namespace Testing {

void ENSURE(Status status, const char* msg = "") {
  if (!status.ok()) {
    std::cerr << "TEST FAILED WITH ERROR STATUS";
    if (*msg) {
      std::cerr << "\n Test message: \n" << msg << "\n";
    }
    std::cerr << std::endl;
    std::raise(SIGABRT);
  }
}

void ENSURE_RGBA210_EQ(uint32_t a, uint32_t b) {
  ENSURE_EQ(Image::RGB210::a(a), Image::RGB210::a(b));
  if (Image::RGB210::a(a)) {
    ENSURE_EQ(Image::RGB210::r(a), Image::RGB210::r(b));
    ENSURE_EQ(Image::RGB210::g(a), Image::RGB210::g(b));
    ENSURE_EQ(Image::RGB210::b(a), Image::RGB210::b(b));
  }
}

void ENSURE_RGBA210_ARRAY_EQ(const uint32_t* exp, const uint32_t* actual, std::size_t w, std::size_t h) {
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const uint32_t& expValue = exp[y * w + x];
      const uint32_t& actualValue = actual[y * w + x];
      if (Image::RGB210::a(expValue) != Image::RGB210::a(actualValue)) {
        std::cerr << "TEST FAILED: At index '(" << x << "," << y << ")', expected alpha=" << Image::RGB210::a(expValue)
                  << ", got alpha=" << Image::RGB210::a(actualValue) << std::endl;
        ENSURE_EQ(Image::RGB210::a(expValue), Image::RGB210::a(actualValue));
      } else if (Image::RGB210::a(expValue)) {
        if (!(expValue == actualValue)) {
          std::cerr << "TEST FAILED: At index '(" << x << "," << y << ")', expected '(" << Image::RGB210::r(expValue)
                    << "," << Image::RGB210::g(expValue) << "," << Image::RGB210::b(expValue) << ")', got '("
                    << Image::RGB210::r(actualValue) << "," << Image::RGB210::g(actualValue) << ","
                    << Image::RGB210::b(actualValue) << ")'" << std::endl;
          ENSURE_RGBA210_EQ(expValue, actualValue);
        }
      }
    }
  }
}

void ENSURE_RGBA8888_ARRAY_EQ(const uint32_t* exp, const uint32_t* actual, std::size_t w, std::size_t h) {
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const uint32_t& expValue = exp[y * w + x];
      const uint32_t& actualValue = actual[y * w + x];
      if (!(expValue == actualValue)) {
        std::cerr << "TEST FAILED: At index '(" << x << "," << y << ")', expected '(" << Image::RGBA::r(expValue) << ","
                  << Image::RGBA::g(expValue) << "," << Image::RGBA::b(expValue) << ")', got '("
                  << Image::RGBA::r(actualValue) << "," << Image::RGBA::g(actualValue) << ","
                  << Image::RGBA::b(actualValue) << ")'" << std::endl;
        ENSURE_EQ(expValue, actualValue);
      }
    }
  }
}

void ENSURE_PNG_FILE_EQ(const std::string& filename, const std::vector<uint32_t>& data) {
  std::vector<unsigned char> tmp;
  int64_t width = 0;
  int64_t height = 0;
  if (!Util::PngReader::readRGBAFromFile(filename.c_str(), width, height, tmp)) {
    std::cerr << "Image '" << filename << "': failed to setup reader." << std::endl;
    ENSURE_EQ(1, 0);
  }
  ENSURE_EQ(tmp.size(), (size_t)(width * height * 4), "Sizes do not match");
  ENSURE_EQ(tmp.size(), data.size() * 4, "Sizes do not match");

  std::vector<uint32_t> rgba;
  ENSURE(Util::ImageProcessing::packImageRGBA(tmp, rgba));
  for (size_t k = 0; k < data.size(); k++) {
    ENSURE_EQ(rgba[k], data[k]);
  }
}

void ENSURE_PNG_FILE_EQ(const std::string& filename, const std::vector<unsigned char>& data) {
  std::vector<uint32_t> rgba;
  ENSURE(data.size() % 4 == 0, "Invalid data size");
  ENSURE(Util::ImageProcessing::packImageRGBA(data, rgba));
  ENSURE_PNG_FILE_EQ(filename, rgba);
}

void ENSURE_PNG_FILE_EQ(const std::string& filename, const GPU::Buffer<const uint32_t>& buffer) {
  std::vector<uint32_t> data(buffer.numElements());
  ENSURE(GPU::memcpyBlocking(&data[0], buffer, buffer.numElements() * sizeof(uint32_t)));
  ENSURE_PNG_FILE_EQ(filename, data);
}

/*
 * Threshold of the average normalized difference, range [0..1]
 */
void ENSURE_ARRAY_SIMILARITY(const unsigned char* exp, const unsigned char* actual, std::size_t s,
                             const float threshold = 0.01f) {
  float totalDifference = 0.0f;
  for (std::size_t i = 0; i < s; i++) {
    totalDifference += std::abs(float(exp[i]) - float(actual[i])) / 255.0f;
  }
  totalDifference /= s;
  if (totalDifference >= threshold) {
    std::cout << "*** Difference: " << totalDifference << std::endl;
  }
  ENSURE(totalDifference < threshold, "Difference is larger than threshold");
}

/*
 * Threshold of the average normalized difference, range [0..1]
 */
void ENSURE_PNG_FILE_AND_RGBA_BUFFER_SIMILARITY(const std::string& filename, const std::vector<unsigned char>& rgba,
                                                const float threshold = 0.01f,
                                                const std::string& msg = std::string("")) {
  std::vector<unsigned char> image;
  int64_t width = 0;
  int64_t height = 0;
  if (!Util::PngReader::readRGBAFromFile(filename.c_str(), width, height, image)) {
    std::cerr << "Image '" << filename << "': failed to setup reader." << std::endl;
    ENSURE_EQ(1, 0, msg.c_str());
  }
  ENSURE_EQ(image.size(), (size_t)(width * height * 4), msg.c_str());
  ENSURE_EQ(image.size(), rgba.size(), msg.c_str());

  ENSURE_ARRAY_SIMILARITY(&rgba[0], &image[0], rgba.size(), threshold);
}

/*
 * Threshold of the average normalized difference, range [0..1]
 */
void ENSURE_PNG_FILE_AND_RGBA_BUFFER_SIMILARITY(const std::string& filename, const GPU::Buffer<const uint32_t>& buffer,
                                                const float threshold = 0.01f,
                                                const std::string& msg = std::string("")) {
  std::vector<uint32_t> data(buffer.numElements());
  ENSURE(GPU::memcpyBlocking(&data[0], buffer, buffer.numElements() * sizeof(uint32_t)));
  std::vector<unsigned char> rgba;
  Util::ImageProcessing::unpackImageRGBA(data, rgba);

  ENSURE_PNG_FILE_AND_RGBA_BUFFER_SIMILARITY(filename, rgba, threshold, msg);
}

}  // namespace Testing
}  // namespace VideoStitch
