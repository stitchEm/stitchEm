// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/orah/imuStabilization.hpp"

namespace VideoStitch {
namespace Testing {

void testFusionIMU() {
  VideoStitch::Stab::FusionIMU fusionIMU;
  Vector3<double> acc(0, 0, 1);
  Vector3<double> gyr(0, 0, 0);
  Quaternion<double> q = fusionIMU.init(acc, 0);
  ENSURE_APPROX_EQ(q.getQ0(), 1., 1e-5);
  ENSURE_APPROX_EQ(q.getQ1(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ2(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ3(), 0., 1e-5);

  const double angle = M_PI / 3;

  acc = Vector3<double>(0, std::sin(angle), std::cos(angle));
  q = fusionIMU.init(acc, 0);
  ENSURE_APPROX_EQ(2 * std::acos(q.getQ0()), angle, 1e-5);
  ENSURE_APPROX_EQ(2 * std::asin(q.getQ1()), angle, 1e-5);
  ENSURE_APPROX_EQ(q.getQ2(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ3(), 0., 1e-5);

  acc = Vector3<double>(0, -std::sin(angle), std::cos(angle));
  q = fusionIMU.init(acc, 0);
  ENSURE_APPROX_EQ(2 * std::acos(q.getQ0()), angle, 1e-5);
  ENSURE_APPROX_EQ(2 * std::asin(q.getQ1()), -angle, 1e-5);
  ENSURE_APPROX_EQ(q.getQ2(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ3(), 0., 1e-5);

  acc = Vector3<double>(std::sin(angle), 0, std::cos(angle));
  q = fusionIMU.init(acc, 0);
  ENSURE_APPROX_EQ(2 * std::acos(q.getQ0()), angle, 1e-5);
  ENSURE_APPROX_EQ(q.getQ1(), 0., 1e-5);
  ENSURE_APPROX_EQ(2 * std::asin(q.getQ2()), -angle, 1e-5);
  ENSURE_APPROX_EQ(q.getQ3(), 0., 1e-5);

  acc = Vector3<double>(-std::sin(angle), 0, std::cos(angle));
  q = fusionIMU.init(acc, 0);
  ENSURE_APPROX_EQ(2 * std::acos(q.getQ0()), angle, 1e-5);
  ENSURE_APPROX_EQ(q.getQ1(), 0., 1e-5);
  ENSURE_APPROX_EQ(2 * std::asin(q.getQ2()), angle, 1e-5);
  ENSURE_APPROX_EQ(q.getQ3(), 0., 1e-5);

  //////  use accelerometer only
  fusionIMU.setFusionFactor(1);  ///< discard gyroscope during fusion
  acc = Vector3<double>(0, 0, 1);
  q = fusionIMU.init(acc, 1000000);
  ENSURE_APPROX_EQ(q.getQ0(), 1., 1e-5);
  ENSURE_APPROX_EQ(q.getQ1(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ2(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ3(), 0., 1e-5);

  acc = Vector3<double>(0, std::sin(angle), std::cos(angle));
  gyr = Vector3<double>(1, 1, 1);
  q = fusionIMU.update(gyr, acc, 2000000);
  ENSURE_APPROX_EQ(2 * std::acos(q.getQ0()), angle, 1e-5);
  ENSURE_APPROX_EQ(2 * std::asin(q.getQ1()), angle, 1e-5);
  ENSURE_APPROX_EQ(q.getQ2(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ3(), 0., 1e-5);

  ////// use gyroscope only
  fusionIMU.setFusionFactor(0);  ///< discard accelerometer during fusion
  acc = Vector3<double>(0, 0, 1);
  q = fusionIMU.init(acc, 1000000);
  ENSURE_APPROX_EQ(q.getQ0(), 1., 1e-5);
  ENSURE_APPROX_EQ(q.getQ1(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ2(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ3(), 0., 1e-5);

  acc = Vector3<double>(0, 0, 1);
  gyr = Vector3<double>(angle, 0, 0);
  q = fusionIMU.update(gyr, acc, 1033333);  ///< integrate only during 1/30 of a second
  ENSURE_APPROX_EQ(2 * std::acos(q.getQ0()), angle / 30, 1e-5);
  ENSURE_APPROX_EQ(2 * std::asin(q.getQ1()), angle / 30, 1e-5);
  ENSURE_APPROX_EQ(q.getQ2(), 0., 1e-5);
  ENSURE_APPROX_EQ(q.getQ3(), 0., 1e-5);
}

void testImuHorizonLeveling() {
  std::vector<VideoStitch::IMU::Measure> vectMes;
  VideoStitch::Stab::IMUStabilization stab;
  double yaw, pitch, roll;
  Quaternion<double> q;

  stab.setLowPassIIR(1);  // no filtering
  const double angle = M_PI / 3;
  VideoStitch::IMU::Measure mes;
  mes.imu_acc_z = static_cast<int>(16384 * std::cos(angle));
  mes.imu_acc_y = static_cast<int>(16384 * std::sin(angle));
  mes.timestamp = 1000000;
  vectMes.push_back(mes);
  stab.addMeasures(vectMes);
  q = stab.computeHorizonLeveling();
  q.toEuler(yaw, pitch, roll);
  ENSURE_APPROX_EQ(yaw, 0., 1e-4);
  ENSURE_APPROX_EQ(pitch, 0., 1e-4);
  ENSURE_APPROX_EQ(roll, angle, 1e-4);

  mes = VideoStitch::IMU::Measure();

  mes.imu_acc_z = static_cast<int>(16384 * std::cos(angle));
  mes.imu_acc_x = static_cast<int>(16384 * std::sin(angle));
  mes.timestamp = 2000000;
  vectMes.clear();
  vectMes.push_back(mes);
  stab.addMeasures(vectMes);
  q = stab.computeHorizonLeveling();
  q.toEuler(yaw, pitch, roll);
  ENSURE_APPROX_EQ(yaw, 0., 1e-4);
  ENSURE_APPROX_EQ(pitch, angle, 1e-4);
  ENSURE_APPROX_EQ(roll, 0., 1e-4);
}

void testImuStabilization() {
  std::vector<VideoStitch::IMU::Measure> vectMes;
  double yaw, pitch, roll;
  Quaternion<double> q;
  const double angle = M_PI / 3;

  /// query without any element in the buffer
  {
    VideoStitch::Stab::IMUStabilization stab;
    stab.setLowPassIIR(1);    // no filtering
    stab.setFusionFactor(1);  // use accelerometer only

    q = stab.computeOrientation(1000000);  // buffer is empty
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, 0., 1e-4);
  }

  /// queries with one element in the buffer
  {
    VideoStitch::Stab::IMUStabilization stab;
    stab.setLowPassIIR(1);    // no filtering
    stab.setFusionFactor(1);  // use accelerometer only

    vectMes.resize(1);
    vectMes[0].imu_acc_z = static_cast<int>(16384 * std::cos(angle));
    vectMes[0].imu_acc_y = static_cast<int>(16384 * std::sin(angle));
    vectMes[0].timestamp = 500000;
    stab.addMeasures(vectMes);

    q = stab.computeOrientation(0);  // query before
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, 0., 1e-4);

    q = stab.computeOrientation(500000);  // query the exact element
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, angle, 1e-4);

    q = stab.computeOrientation(1000000);  // query after
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, 0., 1e-4);
  }

  /////////  queries with two or more elements in the buffer
  {
    VideoStitch::Stab::IMUStabilization stab;
    stab.setLowPassIIR(1);    // no filtering
    stab.setFusionFactor(1);  // use accelerometer only

    vectMes.clear();
    vectMes.resize(2);
    vectMes[0].imu_acc_z = 16384;
    vectMes[0].timestamp = 1000000;

    vectMes[1].imu_acc_z = static_cast<int>(16384 * std::cos(angle));
    vectMes[1].imu_acc_y = static_cast<int>(16384 * std::sin(angle));
    vectMes[1].timestamp = 2000000;

    stab.addMeasures(vectMes);  // put 2 elements in the buffer

    q = stab.computeOrientation(0);  // before first element
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, 0., 1e-4);

    q = stab.computeOrientation(1000000);  // query exactly the first element
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, 0., 1e-4);

    q = stab.computeOrientation(1500000);  // query between first and second elements
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, angle / 2., 1e-4);

    q = stab.computeOrientation(2000000);  // query exactly the second element
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, angle, 1e-4);

    q = stab.computeOrientation(3000000);  // after last element
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, 0., 1e-4);

    vectMes.clear();
    vectMes.resize(2);
    vectMes[0].imu_acc_z = static_cast<int>(16384 * std::cos(angle));
    vectMes[0].imu_acc_y = static_cast<int>(16384 * std::sin(angle));
    vectMes[0].timestamp = 3000000;
    vectMes[1].imu_acc_z = static_cast<int>(16384 * std::cos(angle));
    vectMes[1].imu_acc_y = static_cast<int>(16384 * std::sin(angle));
    vectMes[1].timestamp = 4000000;

    stab.addMeasures(vectMes);  // add 2 more elements

    q = stab.computeOrientation(3500000);  // this query will pop the first 2 elements
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, angle, 1e-4);

    q = stab.computeOrientation(2000000);  // is no longer valid (element was popped out)
    q.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(yaw, 0., 1e-4);
    ENSURE_APPROX_EQ(pitch, 0., 1e-4);
    ENSURE_APPROX_EQ(roll, 0., 1e-4);
  }
}

void testImuStabilizationWobblingCancellation() {
  std::vector<VideoStitch::IMU::Measure> vectMes;
  double yaw, pitch, roll;
  Quaternion<double> q;
  const double angle = M_PI / 3;

  VideoStitch::Stab::IMUStabilization stab;
  stab.setLowPassIIR(1);  // no filtering

  vectMes.clear();
  vectMes.resize(2);

  // first measurement
  vectMes[0].imu_acc_z = static_cast<int>(16384 * std::cos(angle));
  vectMes[0].imu_acc_y = static_cast<int>(16384 * std::sin(angle));
  vectMes[0].timestamp = 1000000;

  // second measurement: the accelerometer changes slightly, but the gyro remains very small
  vectMes[1].imu_gyr_x = vectMes[1].imu_gyr_y = vectMes[1].imu_gyr_z = 20;
  vectMes[1].imu_acc_z = static_cast<int>(15000 * std::cos(angle));
  vectMes[1].imu_acc_y = static_cast<int>(17000 * std::sin(angle));
  vectMes[1].timestamp = 2000000;

  stab.addMeasures(vectMes);  // put 2 elements in the buffer

  // query exactly the timestamp of the first element, retrieve the roll angle
  q = stab.computeOrientation(1000000);  // query exactly the first element
  q.toEuler(yaw, pitch, roll);
  ENSURE_APPROX_EQ(yaw, 0., 1e-4);
  ENSURE_APPROX_EQ(pitch, 0., 1e-4);
  ENSURE_APPROX_EQ(roll, angle, 1e-4);

  // query exactly the timestamp of the second element
  // the orientation does not change despite the noise in the sensor
  q = stab.computeOrientation(2000000);
  q.toEuler(yaw, pitch, roll);
  ENSURE_APPROX_EQ(yaw, 0., 1e-4);
  ENSURE_APPROX_EQ(pitch, 0., 1e-4);
  ENSURE_APPROX_EQ(roll, angle, 1e-4);
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  VideoStitch::Testing::testFusionIMU();
  VideoStitch::Testing::testImuHorizonLeveling();
  VideoStitch::Testing::testImuStabilization();
  VideoStitch::Testing::testImuStabilizationWobblingCancellation();

  return 0;
}
