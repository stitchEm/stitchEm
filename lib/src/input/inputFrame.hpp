// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/buffer.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/imuData.hpp"
#include "libvideostitch/orah/exposureData.hpp"

namespace VideoStitch {
namespace Input {

struct PotentialFrame {
  Input::ReadStatus status;
  Core::Buffer frame;
};

struct MetadataChunk {
  std::vector<IMU::Measure> imu;
  std::vector<std::map<videoreaderid_t, Metadata::Exposure>> exposure;
  std::vector<std::map<videoreaderid_t, Metadata::WhiteBalance>> whiteBalance;
  std::vector<std::map<videoreaderid_t, Metadata::ToneCurve>> toneCurve;

  bool hasExposureData() const {
    for (const auto& map : exposure) {
      if (map.size()) {
        return true;
      }
    }
    for (const auto& map : whiteBalance) {
      if (map.size()) {
        return true;
      }
    }
    for (const auto& map : toneCurve) {
      if (map.size()) {
        return true;
      }
    }
    return false;
  }

  void clear() {
    imu.clear();
    exposure.clear();
    whiteBalance.clear();
    toneCurve.clear();
  }
};

}  // namespace Input
}  // namespace VideoStitch
