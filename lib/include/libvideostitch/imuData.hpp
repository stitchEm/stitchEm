// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IMUDATA_HPP
#define IMUDATA_HPP

#include "config.hpp"
#include "matrix.hpp"

#include <sstream>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>

namespace VideoStitch {

namespace IMU {

// +/- 2g on 16 bits => 2g: 32768 => 1g: 16384
static const double IMU_RATIO_ACC = 1. / 16384.;

// +/- 250 degrees/s on 16 bits => 250 degrees/s: 32768 => 1 degree/s: 131
static const double IMU_RATIO_GYR = M_PI / (131. * 180.);

class VS_EXPORT Measure {
 public:
  Measure()
      : timestamp(0),
        imu_acc_x(0),
        imu_acc_y(0),
        imu_acc_z(0),
        imu_gyr_x(0),
        imu_gyr_y(0),
        imu_gyr_z(0),
        imu_mag_x(0),
        imu_mag_y(0),
        imu_mag_z(0),
        imu_temperature(0) {}

  Measure(mtime_t ts, int acc_x, int acc_y, int acc_z, int gyr_x, int gyr_y, int gyr_z, int mag_x, int mag_y, int mag_z,
          int temperature)
      : timestamp(ts),
        imu_acc_x(acc_x),
        imu_acc_y(acc_y),
        imu_acc_z(acc_z),
        imu_gyr_x(gyr_x),
        imu_gyr_y(gyr_y),
        imu_gyr_z(gyr_z),
        imu_mag_x(mag_x),
        imu_mag_y(mag_y),
        imu_mag_z(mag_z),
        imu_temperature(temperature) {}

  mtime_t timestamp;
  int imu_acc_x, imu_acc_y, imu_acc_z;
  int imu_gyr_x, imu_gyr_y, imu_gyr_z;
  int imu_mag_x, imu_mag_y, imu_mag_z;
  int imu_temperature;

  bool operator==(const Measure& other) const {
    return (timestamp == other.timestamp &&

            imu_acc_x == other.imu_acc_x && imu_acc_y == other.imu_acc_y && imu_acc_z == other.imu_acc_z &&

            imu_gyr_x == other.imu_gyr_x && imu_gyr_y == other.imu_gyr_y && imu_gyr_z == other.imu_gyr_z &&

            imu_mag_x == other.imu_mag_x && imu_mag_y == other.imu_mag_y && imu_mag_z == other.imu_mag_z &&

            imu_temperature == other.imu_temperature);
  }

  Vector3<double> getAcc() const {
    return Vector3<double>(imu_acc_x * IMU_RATIO_ACC, imu_acc_y * IMU_RATIO_ACC, imu_acc_z * IMU_RATIO_ACC);
  }

  Vector3<double> getGyr() const {
    return Vector3<double>(imu_gyr_x * IMU_RATIO_GYR, imu_gyr_y * IMU_RATIO_GYR, imu_gyr_z * IMU_RATIO_GYR);
  }
};

static inline std::ostream& operator<<(std::ostream& stream, const Measure& imuData) {
  stream << "IMU data: timestamp: " << imuData.timestamp;
  stream << "    ACC(x,y,z): " << imuData.imu_acc_x << ", " << imuData.imu_acc_y << ", " << imuData.imu_acc_z;
  stream << "    GYR(x,y,z): " << imuData.imu_gyr_x << ", " << imuData.imu_gyr_y << ", " << imuData.imu_gyr_z;
  stream << "    MAG(x,y,z): " << imuData.imu_mag_x << ", " << imuData.imu_mag_y << ", " << imuData.imu_mag_z;
  stream << "    temperature: " << imuData.imu_temperature;
  return stream;
}

}  // end namespace IMU

}  // end namespace VideoStitch

#endif
