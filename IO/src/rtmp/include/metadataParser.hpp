// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/imuData.hpp"
#include "libvideostitch/orah/exposureData.hpp"

namespace VideoStitch {

class MetadataParser {
 public:
  static bool parse(const std::string& textLine, videoreaderid_t inputOffset,
                    std::pair<bool, IMU::Measure>& potentialMeasure,
                    std::map<videoreaderid_t, Metadata::Exposure>& exposure,
                    std::map<videoreaderid_t, Metadata::WhiteBalance>& whiteBalance,
                    std::map<videoreaderid_t, Metadata::ToneCurve>& toneCurve);
};

}  // namespace VideoStitch
