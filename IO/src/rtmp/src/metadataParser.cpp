// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "metadataParser.hpp"

#include "libvideostitch/logging.hpp"

namespace VideoStitch {

static const int SENSOR_UNINITIALIZED = -1;

#define FAIL_RETURN_CHECK_SENSOR(s, line)                                                                        \
  if (s == SENSOR_UNINITIALIZED) {                                                                               \
    Logger::get(Logger::Error) << "[MetadataParser] Exposure data with unspecified sensor encountered for line " \
                               << "'" << line << "' with inputOffset " << inputOffset << std::endl;              \
    return false;                                                                                                \
  }

bool MetadataParser::parse(const std::string& textLine, videoreaderid_t inputOffset,
                           std::pair<bool, IMU::Measure>& potentialMeasure,
                           std::map<videoreaderid_t, Metadata::Exposure>& exposure,
                           std::map<videoreaderid_t, Metadata::WhiteBalance>& whiteBalance,
                           std::map<videoreaderid_t, Metadata::ToneCurve>& toneCurve) {
  // Metadata header block:
  // "MetaStartMarker,idx=%u,ts=%u": fields are frame index and frame timestamp.

  // If there are no changes in meta data block values then it’s not added to meta data stream. Thus the minimum
  // possible metadata payload should look like this: “MetaStartMarker,idx=1,ts=1,sen=0,sen=1”

  std::stringstream lineStream(textLine);
  std::string cell;

  int sensor = SENSOR_UNINITIALIZED;

  IMU::Measure imuData;
  Metadata::Exposure exp;
  Metadata::WhiteBalance wb;

  potentialMeasure.first = false;

  mtime_t timestamp{0};

  // valid Orah 4i metadata line starts with marker
  std::getline(lineStream, cell, ',');
  if (cell.compare("MetaStartMarker") != 0) {
    return false;
  }

  while (std::getline(lineStream, cell, ',')) {
    try {
      std::size_t pos = 0;
      pos = cell.find("=");
      if (pos == std::string::npos) {
        // pre-0.9.18 tone curve format
        // TODO remove when all test cameras have been upgraded
        if (cell.find("tcStart") != std::string::npos) {
          Metadata::ToneCurve tc;
          tc.timestamp = timestamp;
          FAIL_RETURN_CHECK_SENSOR(sensor, textLine);
          // 0,4,8,13,17,22,26,31, ...
          for (int i = 0; i < 256; i++) {
            if (!std::getline(lineStream, cell, ',')) {
              Logger::get(Logger::Error) << "[MetadataParser] Incomplete pre-0.9.18 tone curve"
                                         << "<" << cell << ">" << std::endl;
              return false;
            }
            tc.curve[i] = uint16_t(stoi(cell));
          }
          toneCurve[sensor] = tc;
        } else {
          Logger::get(Logger::Error) << "[MetadataParser] Textline is not well formed: "
                                     << "<" << textLine << "> when processing <" << cell << ">" << std::endl;
          return false;
        }
      }

      std::string myKey = cell.substr(0, pos);
      std::string myVal = cell.substr(pos + 1, cell.size());

      if (myKey == "ts") {
        timestamp = mtime_t(stoi(myVal)) * 1000;
        continue;
      }

      if (myKey == "sen") {
        sensor = stoi(myVal);

        if (!(sensor == 0 || sensor == 1)) {
          Logger::get(Logger::Error) << "[MetadataParser] Invalid sensor " << sensor << " for line "
                                     << "'" << textLine << "' with inputOffset " << inputOffset << std::endl;
          return false;
        }

        // add input offset to get real input ID
        sensor += inputOffset;
        continue;
      }

      // ----------------- IMU -----------------
      // Metadata block for IMU:
      // “xgy=%hi,ygy=%hi,zgy=%hi,xac=%hi,yac=%hi,zac=%hi,xma=%hi,yma=%hi,zma=%hi,temp=%hi":
      // [xyz]gy is gyroscope data, [xyz]ac is accelerometer data, [xyz]ma is magnetometer data, temp is temperature.

      if (myKey == "xgy") {
        imuData.imu_gyr_x = stoi(myVal);
        continue;
      }
      if (myKey == "ygy") {
        imuData.imu_gyr_y = stoi(myVal);
        continue;
      }
      if (myKey == "zgy") {
        imuData.imu_gyr_z = stoi(myVal);
        continue;
      }
      if (myKey == "xac") {
        imuData.imu_acc_x = stoi(myVal);
        continue;
      }
      if (myKey == "yac") {
        imuData.imu_acc_y = stoi(myVal);
        continue;
      }
      if (myKey == "zac") {
        imuData.imu_acc_z = stoi(myVal);
        continue;
      }
      if (myKey == "xma") {
        imuData.imu_mag_x = stoi(myVal);
        continue;
      }
      if (myKey == "yma") {
        imuData.imu_mag_y = stoi(myVal);
        continue;
      }
      if (myKey == "zma") {
        imuData.imu_mag_z = stoi(myVal);
        continue;
      }
      if (myKey == "temp") {
        imuData.imu_temperature = stoi(myVal);

        // last key/val, flush IMU measure
        if (potentialMeasure.first) {
          Logger::get(Logger::Error) << "[Metadata] Two IMU measures in one text line not supported. Line: " << textLine
                                     << std::endl;
        }
        potentialMeasure.first = true;
        imuData.timestamp = timestamp;
        potentialMeasure.second = imuData;
        imuData = {};
        continue;
      }

      // ----------------- IQ parameters -----------------
      // 1st metadata block for IQ parameters:
      // “iso=%hu,sht=%1.8f,sht_mx=%1.8f”: the fields are: ISO value, shutter timer, shutter time max.
      if (myKey == "iso") {
        FAIL_RETURN_CHECK_SENSOR(sensor, textLine);
        exp.iso = (uint16_t)stoi(myVal);
        continue;
      }
      if (myKey == "sht") {
        FAIL_RETURN_CHECK_SENSOR(sensor, textLine);
        exp.shutterTime = stof(myVal);
        continue;
      }
      if (myKey == "sht_mx") {
        FAIL_RETURN_CHECK_SENSOR(sensor, textLine);
        exp.shutterTimeMax = stof(myVal);

        exp.timestamp = timestamp;
        exposure[sensor] = exp;
        exp = {};
        continue;
      }

      // 2nd metadata block for IQ parameters:
      // “r=%u,g=%u,b=%u”: there are WB scalers for red, green and blue colors.
      if (myKey == "r") {
        FAIL_RETURN_CHECK_SENSOR(sensor, textLine);
        wb.red = stoi(myVal);
        continue;
      }
      if (myKey == "g") {
        FAIL_RETURN_CHECK_SENSOR(sensor, textLine);
        wb.green = stoi(myVal);
        continue;
      }
      if (myKey == "b") {
        FAIL_RETURN_CHECK_SENSOR(sensor, textLine);
        wb.blue = stoi(myVal);

        wb.timestamp = timestamp;
        whiteBalance[sensor] = wb;
        wb = {};
        continue;
      }

      // 3rd metadata block for IQ parameters:
      // “tcStart=” + ”%hu,” times  tone curve array  size.
      if (myKey == "tcStart") {
        FAIL_RETURN_CHECK_SENSOR(sensor, textLine);

        Metadata::ToneCurve tc;
        tc.timestamp = timestamp;
        std::istringstream tcLine(myVal);
        // 0,4,8,13,17,22,26,31, ...
        for (int i = 0; i < 256; i++) {
          std::string tcVal;
          if (!std::getline(tcLine, tcVal, ';')) {
            Logger::get(Logger::Error) << "[MetadataParser] Incomplete tone curve"
                                       << "<" << cell << ">" << std::endl;
            return false;
          }
          tc.curve[i] = uint16_t(stoi(tcVal));
        }
        if (toneCurve.find(sensor) != toneCurve.end()) {
          Logger::get(Logger::Error)
              << "[MetadataParser] Two tone curves of same sensor in one text line not supported. Line: "
              << "<" << cell << ">" << std::endl;
        }
        toneCurve[sensor] = tc;
        continue;
      }

    } catch (const std::invalid_argument&) {
      Logger::get(Logger::Error) << "[MetadataParser] Textline is not well formed: <" << textLine
                                 << ">, invalid argument" << std::endl;
      return false;
    } catch (const std::out_of_range&) {
      Logger::get(Logger::Error) << "[MetadataParser] Textline is not well formed: <" << textLine << ">, out of range"
                                 << std::endl;
      return false;
    }
  }

  return true;
}

}  // namespace VideoStitch
