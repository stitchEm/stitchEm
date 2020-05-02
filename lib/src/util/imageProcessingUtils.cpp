// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/imageProcessingUtils.hpp"

#include "pngutil.hpp"

#include "autocrop/autoCrop.hpp"

#include "backend/common/imageOps.hpp"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

namespace VideoStitch {
namespace Util {

Status ImageProcessing::findCropCircle(const int& width, const int& height, void* data, int& x, int& y, int& r,
                                       const VideoStitch::Ptv::Value* algoConfig, const std::string* dumpFile) {
  x = 0;
  y = 0;
  r = 0;
  cv::Mat originalImage(cv::Size(width, height), CV_8UC4, data);
  cv::Mat inputImage;
  cv::cvtColor(originalImage, inputImage, cv::COLOR_RGBA2BGR);
  std::unique_ptr<VideoStitch::Ptv::Value> fake(VideoStitch::Ptv::Value::stringObject("fake"));
  VideoStitch::AutoCrop::AutoCropConfig config(algoConfig ? algoConfig : fake.get());

  AutoCrop::AutoCrop autoCrop(config);
  cv::Point3i circle;
  FAIL_RETURN(autoCrop.findCropCircle(inputImage, circle));
  if (dumpFile) {
    FAIL_RETURN(autoCrop.dumpCircleFile(circle, *dumpFile));
  }
  x = circle.x;
  y = circle.y;
  r = circle.z;
  return Status::OK();
}

Status ImageProcessing::readImage(const std::string& filename, int64_t& width, int64_t& height, int& channelCount,
                                  std::vector<unsigned char>& data) {
  std::string lowerStr = filename;
  std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
  if (lowerStr.substr(lowerStr.size() - 3) == "png") {
    if (!Util::PngReader::readRGBAFromFile(filename.c_str(), width, height, data)) {
      channelCount = 4;
      return {Origin::Input, ErrType::InvalidConfiguration, "Could not read RGBA image: '" + filename + "'"};
    }
  } else {
    return {Origin::Input, ErrType::ImplementationError, "Expected PNG image, got: '" + filename + "'"};
  }
  return Status::OK();
}

#define COLOR_COUNT 10
const float reds[COLOR_COUNT] = {0, 123, 235, 67, 12, 234, 233, 52, 90, 192};
const float greens[COLOR_COUNT] = {32, 65, 128, 12, 0, 223, 178, 96, 182, 155};
const float blues[COLOR_COUNT] = {128, 0, 53, 23, 245, 111, 103, 120, 32, 96};

template <class T>
void ImageProcessing::convertIndexToRGBA(const std::vector<T>& hostRGBA, std::vector<unsigned char>& data,
                                         const int displayBit) {
  data.resize(hostRGBA.size() * 4, 0);
  for (size_t j = 0; j < hostRGBA.size(); j++) {
    int value = hostRGBA[j];
    if (displayBit >= 0) {
      value = (value & displayBit);
    }
    int count = 0;
    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
    while (value > 0) {
      if ((value & 1) > 0) {
        r += reds[count % COLOR_COUNT];
        g += greens[count % COLOR_COUNT];
        b += blues[count % COLOR_COUNT];
        a += 1.0f;
      }
      value = value >> 1;
      count++;
    }
    if (a == 0.0f) {
      data[4 * j + 3] = 0;
      a = 1;
    } else {
      data[4 * j + 3] = 255;
    }
    data[4 * j + 0] = (unsigned char)(std::min(r / a, 255.0f));
    data[4 * j + 1] = (unsigned char)(std::min(g / a, 255.0f));
    data[4 * j + 2] = (unsigned char)(std::min(b / a, 255.0f));
  }
}

template void ImageProcessing::convertIndexToRGBA(const std::vector<uint32_t>& hostRGBA,
                                                  std::vector<unsigned char>& data, const int displayBit);
template void ImageProcessing::convertIndexToRGBA(const std::vector<int>& hostRGBA, std::vector<unsigned char>& data,
                                                  const int displayBit);
template void ImageProcessing::convertIndexToRGBA(const std::vector<unsigned char>& hostRGBA,
                                                  std::vector<unsigned char>& data, const int displayBit);

Status ImageProcessing::packImageRGBA(const std::vector<unsigned char>& data, std::vector<uint32_t>& rgba) {
  if (data.size() % 4 != 0) {
    return {Origin::Unspecified, ErrType::InvalidConfiguration,
            "ImageProcessing::packImageRGBA - Input size is invalid"};
  }
  rgba.clear();
  for (size_t i = 0; i < data.size() / 4; i++) {
    rgba.push_back(Image::RGBA::pack(data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]));
  }
  return Status::OK();
}

void ImageProcessing::unpackImageRGBA(const std::vector<uint32_t>& rgba, std::vector<unsigned char>& data) {
  data.clear();
  for (size_t i = 0; i < rgba.size(); i++) {
    data.push_back((unsigned char)Image::RGBA::r(rgba[i]));
    data.push_back((unsigned char)Image::RGBA::g(rgba[i]));
    data.push_back((unsigned char)Image::RGBA::b(rgba[i]));
    data.push_back((unsigned char)Image::RGBA::a(rgba[i]));
  }
}

uint32_t ImageProcessing::packRGBAColor(const uint32_t r, const uint32_t g, const uint32_t b, const uint32_t a) {
  return Image::RGBA::pack(r, g, b, a);
}

uint32_t ImageProcessing::packRGB210Color(const uint32_t r, const uint32_t g, const uint32_t b, const uint32_t a) {
  return Image::RGB210::pack(r, g, b, a);
}

}  // namespace Util
}  // namespace VideoStitch
