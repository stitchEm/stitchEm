// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "stabilization/ukfQuaternion.hpp"

#include <fstream>

namespace VideoStitch {
namespace Testing {

/**
 * @brief testUkfQuaternion
 *
 * Check that UKF has converged to the correct orientation (less than 1e-2 degree error) after 3 iterations
 */
void testUkfQuaternionConvergence() {
  VideoStitch::Stabilization::UKF_Quaternion ukf;

  Vector3<double> targetOrientation(0, 0, -M_PI / 4);
  Quaternion<double> targetOrientationQ = Quaternion<double>::fromAxisAngle(targetOrientation);

  Eigen::Matrix<double, 9, 1> targetMeasurement;
  targetMeasurement << 0., 0., 0., 0., 0., 1., sqrt(2) / 2., sqrt(2) / 2., 0.;

  for (std::size_t i = 0; i < 3; ++i) {
    ukf.predict(0.02);
    ukf.measure(targetMeasurement);
  }
  Quaternion<double> qdiff = targetOrientationQ * ukf.getCurrentOrientation().conjugate();
  ENSURE_APPROX_EQ(0., qdiff.toAxisAngle().norm() * 180. / M_PI, 1e-2);
}

#ifndef __clang_analyzer__  // VSA-7040

/**
 * @brief testUkfQuaternionTrackingSequence
 *
 * Check that UKF is able to track correct orientation on a sequence (less than 0.3 degree error)
 */
void testUkfQuaternionTrackingSequence() {
  VideoStitch::Stabilization::UKF_Quaternion ukf;

  std::ifstream ifs;
  ifs.open("data/stabilization/imu_data.phone.txt", std::ios::in);
  std::string line;

  double previous_timestamp = 0;
  int current_index = 0;
  while (std::getline(ifs, line)) {
    current_index++;
    std::istringstream iss(line);
    unsigned long long timestamp;
    double acc_x, acc_y, acc_z, mag_x, mag_y, mag_z, gyr_x, gyr_y, gyr_z;
    double m00, m01, m02, m10, m11, m12, m20, m21, m22;
    iss >> timestamp;
    iss >> acc_x >> acc_y >> acc_z >> mag_x >> mag_y >> mag_z >> gyr_x >> gyr_y >> gyr_z;
    iss >> m00 >> m01 >> m02 >> m10 >> m11 >> m12 >> m20 >> m21 >> m22;
    Matrix33<double> targetOrientationM(m00, m01, m02, m10, m11, m12, m20, m21, m22);

    Quaternion<double> targetOrientationQ = Quaternion<double>::fromRotationMatrix(targetOrientationM).conjugate();

    double delta_t = timestamp - previous_timestamp;
    delta_t /= 1e9;
    previous_timestamp = timestamp;
    if (current_index == 1) {
      continue;
    }

    Vector3<double> mag(mag_x, mag_y, mag_z);
    Vector3<double> acc(acc_x, acc_y, acc_z);

    mag.normalize();
    acc.normalize();
    double scalProduct = dotVector<double>(mag, acc);
    mag -= acc * scalProduct;
    mag.normalize();

    Eigen::Matrix<double, 9, 1> measure;
    measure << gyr_x, gyr_y, gyr_z, acc(0), acc(1), acc(2), mag(0), mag(1), mag(2);

    ukf.predict(delta_t);
    ukf.measure(measure);

    Quaternion<double> qdiff = targetOrientationQ * ukf.getCurrentOrientation().conjugate();
    double diffAngleDeg = qdiff.toAxisAngle().norm() * 180. / M_PI;

    ENSURE(diffAngleDeg < 0.3, "Predicted value too far away from target value");
  }
  ifs.close();
}
#endif  // __clang_analyzer__

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  VideoStitch::Testing::testUkfQuaternionConvergence();
#ifndef __clang_analyzer__  // VSA-7043
  VideoStitch::Testing::testUkfQuaternionTrackingSequence();
#endif  // __clang_analyzer__

  return 0;
}
