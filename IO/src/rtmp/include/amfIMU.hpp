// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef AMFIMU_HPP
#define AMFIMU_HPP

#include "amfIncludes.hpp"

namespace VideoStitch {

namespace IMU {

static const AVal av_imuid = mAVC("IMU Data");
static const AVal av_imu_acc_x = mAVC("imu_acc_x");
static const AVal av_imu_acc_y = mAVC("imu_acc_y");
static const AVal av_imu_acc_z = mAVC("imu_acc_z");
static const AVal av_imu_gyr_x = mAVC("imu_gyr_x");
static const AVal av_imu_gyr_y = mAVC("imu_gyr_y");
static const AVal av_imu_gyr_z = mAVC("imu_gyr_z");
static const AVal av_imu_mag_x = mAVC("imu_mag_x");
static const AVal av_imu_mag_y = mAVC("imu_mag_y");
static const AVal av_imu_mag_z = mAVC("imu_mag_z");
static const AVal av_imu_temperature = mAVC("imu_temperature");

}  // namespace IMU

}  // namespace VideoStitch

#endif  // AMFIMU_HPP
