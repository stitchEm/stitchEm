// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "compressionUtils.hpp"
#include "geometryProcessingUtils.hpp"
#include "polylineEncodingUtils.hpp"
#include "util/base64.hpp"

namespace VideoStitch {
namespace Util {

float Compression::binaryDifference(const std::vector<unsigned char>& im0, const std::vector<unsigned char>& im1) {
  if (im0.size() == 0) {
    return 0.0f;
  }
  float diff = 0.0f;
  for (size_t i = 0; i < im0.size(); i++) {
    diff += std::abs(float(im0[i]) - float(im1[i]));
  }
  return diff / im0.size();
}

Status Compression::polylineEncodeBinaryMask(const int width, const int height, const std::vector<unsigned char>& data,
                                             std::string& values) {
  std::vector<std::vector<cv::Point>> contours;
  const cv::Size size(width, height);
  values = std::string("");
  if (width * height != (int)data.size()) {
    return {Origin::Input, ErrType::InvalidConfiguration,
            "Corrupted input mask in polylines encoding (width * height != data.size())"};
  }
  FAIL_RETURN(
      Util::GeometryProcessing::findImageContours<unsigned char>(size, data, 1, contours, cv::CHAIN_APPROX_TC89_KCOS));

  std::vector<int> encodedSizes;
  for (size_t i = 0; i < contours.size(); i++) {
    std::string contourEncoded;
    Util::PolylineEncoding::polylineEncode(contours[i], contourEncoded);
    encodedSizes.push_back((int)contourEncoded.size());
    values += contourEncoded;
  }

  // Encode vector info
  std::string vectorInfo = PolylineEncoding::polylineEncodeValue((int32_t)encodedSizes.size());

  for (size_t i = 0; i < encodedSizes.size(); i++) {
    vectorInfo += PolylineEncoding::polylineEncodeValue(encodedSizes[i]);
  }
  values = vectorInfo + values;
  return Status::OK();
}

Status Compression::polylineDecodeBinaryMask(const int width, const int height, const std::string& values,
                                             std::vector<unsigned char>& data, const bool toBinary) {
  std::vector<std::vector<cv::Point>> points;
  // Decode all components
  points.clear();
  PolylineEncoding::polylineDecodePolygon(values, points);
  const cv::Size size(width, height);
  if (width * height != (int)data.size()) {
    data.resize(width * height, 0);
  }
  FAIL_RETURN(Util::GeometryProcessing::drawPolygon(size, points, data));
  if (toBinary) {
    std::transform(data.begin(), data.end(), data.begin(),
                   [](unsigned char d) -> unsigned char { return (d == 255) ? 1 : 0; });
  }
  return Status::OK();
}

Status Compression::polylineEncodeBinaryMask(const int width, const int height, const std::string& data,
                                             std::string& values) {
  std::vector<unsigned char> v;
  std::copy(data.begin(), data.end(), std::back_inserter(v));
  return polylineEncodeBinaryMask(width, height, v, values);
}

Status Compression::polylineDecodeBinaryMask(const int width, const int height, const std::string& values,
                                             std::string& data) {
  std::vector<unsigned char> v;
  FAIL_RETURN(polylineDecodeBinaryMask(width, height, values, v));
  data = std::string("");
  std::copy(v.begin(), v.end(), std::back_inserter(data));
  return Status::OK();
}

Status Compression::convertMaskToEncodedMasks(const int width, const int height, const uint32_t* const fullData,
                                              std::vector<std::string>& datas) {
  // Find the maximum number
  uint32_t dataMax = *std::max_element(fullData, fullData + (width * height));
  // Get the position of max-bit;
  int count = 0;
  while (dataMax > 0) {
    dataMax = dataMax >> 1;
    count++;
  }
  datas.clear();
  std::vector<unsigned char> originalData;
  originalData.resize(width * height);

  for (int i = 0; i < count; i++) {
    uint32_t mask = 1 << i;
    for (int j = 0; j < width * height; j++) {
      originalData[j] = (fullData[j] & mask) ? 1 : 0;
    }
    std::string data;
    FAIL_RETURN(Util::Compression::polylineEncodeBinaryMask(width, height, originalData, data));
    datas.push_back(data);
  }
  return Status::OK();
}

Status Compression::convertEncodedMasksToMask(const int width, const int height, const std::vector<std::string>& datas,
                                              std::vector<uint32_t>& fullData) {
  if (width * height != (int)fullData.size()) {
    fullData.resize(width * height, 0);
  }
  std::vector<unsigned char> originalMask;
  for (size_t i = 0; i < datas.size(); i++) {
    FAIL_RETURN(Util::Compression::polylineDecodeBinaryMask(width, height, datas[i], originalMask, false));
    if (width * height != (int)originalMask.size()) {
      return {Origin::Input, ErrType::InvalidConfiguration, "Encoded mask is invalid."};
    }
    if (i == 0) {
      for (size_t j = 0; j < originalMask.size(); j++) {
        fullData[j] = (originalMask[j] == 255) ? (1 << i) : 0;
      }
    } else {
      for (size_t j = 0; j < originalMask.size(); j++) {
        fullData[j] |= (originalMask[j] == 255) ? (1 << i) : 0;
      }
    }
  }
  return Status::OK();
}

}  // namespace Util
}  // namespace VideoStitch
