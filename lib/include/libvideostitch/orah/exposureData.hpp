// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "../config.hpp"
#include "../status.hpp"

#include <array>
#include <cmath>

namespace VideoStitch {

namespace Metadata {

class VS_EXPORT IQMetadata {
 public:
  IQMetadata() = default;
  explicit IQMetadata(mtime_t ts) : timestamp(ts) {}
  mtime_t timestamp;
};

class VS_EXPORT Exposure : public IQMetadata {
 public:
  Exposure() = default;
  Exposure(mtime_t ts, uint16_t i, float sT, float sTM)
      : IQMetadata(ts), iso(i), shutterTime(sT), shutterTimeMax(sTM) {}

  uint16_t iso;
  float shutterTime;
  float shutterTimeMax;

  double computeExposure() const { return (double)shutterTime * (double)iso; }

  bool isValid() const { return iso && (shutterTime > 0); }

  double computeEv() const { return log2(computeExposure()); }

  bool operator==(const Exposure& other) const {
    return (timestamp == other.timestamp && iso == other.iso && shutterTime == other.shutterTime &&
            shutterTimeMax == other.shutterTimeMax);
  }

  friend std::ostream& operator<<(std::ostream& stream, const Exposure& exposure) {
    stream << "Exposure { ";
    stream << "Time stamp: " << exposure.timestamp << ", ";
    stream << "ISO: " << exposure.iso << ", ";
    stream << "Shutter time: " << exposure.shutterTime << ", ";
    stream << "Shutter time max: " << exposure.shutterTimeMax;
    stream << " }";
    return stream;
  }
};

class VS_EXPORT WhiteBalance : public IQMetadata {
 public:
  WhiteBalance() = default;
  WhiteBalance(mtime_t ts, unsigned r, unsigned g, unsigned b) : IQMetadata(ts), red(r), green(g), blue(b) {}

  unsigned red;
  unsigned green;
  unsigned blue;

  bool operator==(const WhiteBalance& other) const {
    return (timestamp == other.timestamp && red == other.red && green == other.green && blue == other.blue);
  }

  friend std::ostream& operator<<(std::ostream& stream, const WhiteBalance& wb) {
    stream << "WhiteBalance { ";
    stream << "Time stamp: " << wb.timestamp << ", ";
    stream << "WB red: " << wb.red << ", ";
    stream << "WB green: " << wb.green << ", ";
    stream << "WB blue: " << wb.blue;
    stream << " }";
    return stream;
  }
};

class VS_EXPORT ToneCurve : public IQMetadata {
 public:
  ToneCurve() = default;
  ToneCurve(mtime_t ts, const std::array<uint16_t, 256>& cv) : IQMetadata(ts) {
    std::copy(std::begin(cv), std::end(cv), std::begin(curve));
  }

  uint16_t curve[256];

  std::array<uint16_t, 256> curveAsArray() const {
    std::array<uint16_t, 256> array;
    std::copy(std::begin(curve), std::end(curve), std::begin(array));
    return array;
  }

  bool operator==(const ToneCurve& other) const {
    return (timestamp == other.timestamp && memcmp(curve, other.curve, sizeof(curve)) == 0);
  }

  friend std::ostream& operator<<(std::ostream& stream, const ToneCurve& tc) {
    stream << "ToneCurve { ";
    stream << "Time stamp: " << tc.timestamp << ", ";
    stream << "Tone curve: " << tc.curve[0];
    for (int i = 1; i < 256; ++i) {
      stream << "," << tc.curve[i];
    }
    stream << " }";
    return stream;
  }
};

}  // namespace Metadata
}  // namespace VideoStitch
