// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IMU_STABILIZATION_HPP
#define IMU_STABILIZATION_HPP

#include "../imuData.hpp"
#include "../status.hpp"
#include "../quaternion.hpp"
#include "../matrix.hpp"
#include "../circularBuffer.hpp"

#include <mutex>
#include <vector>

namespace VideoStitch {
namespace Stab {

class IIRFilter;

enum class GyroBiasStatus { DISABLED = -1, OK = 0, IN_PROGRESS = 1, THRESHOLD_TOO_HIGH = 2, CAMERA_MOVING = 3 };

/*
 * The method is inspired by: https://github.com/richardstechnotes/RTIMULib-Arduino
 *
 * At each timestep, we fuse the current orientation computed using accelerometer only (accQ)
 * with the orientation computed by integrating the gyro over time from the previous position
 */

class VS_EXPORT FusionIMU {
 public:
  FusionIMU();

  /**
   * @brief call this the first time
   * @param acc : accelerometer data, expressed in the IMU coordinate system
   * @param timestamp : in microseconds
   * @return Quaternion in the IMU coordinate system
   */
  Quaternion<double> init(const Vector3<double>& acc, mtime_t timestamp);

  /**
   * @brief call this all the subsequent times, after init() has been called the first time
   * @param gyr : gyroscope data, expressed in the IMU coordinate system
   * @param acc : accelerometer data, expressed in the IMU coordinate system
   * @param timestamp : in microseconds
   * @return Quaternion in the IMU coordinate system
   */
  Quaternion<double> update(const Vector3<double>& gyr, const Vector3<double>& acc, mtime_t timestamp);

  /**
   * @brief Set the fusion factor used to combined measures (acc) and predicted (gyr) orientations
   * @param factor: 0: use gyroscope only   1: use accelerometer only
   *                Any value in between [0; 1] is valid. A value outside this range will trigger a log error
   *                and won't change the value of the attribute
   */
  void setFusionFactor(double factor);

  /**
   * @brief Returns true if the fusion uses the gyroscope measurements
   * This is true only if the fusion factor is below 1.
   * A fusion factor of exactly 1 uses accelerometer data only
   */
  bool usesGyroscope() const;

  /**
   * @brief Set the timestamp without changing the other internal variables
   * @param timestamp : in microseconds
   */
  void setTimestamp(mtime_t timestamp);

 private:
  bool initialized;

  void computeOrientationFromAcc(const Vector3<double>& accelerometer);
  Quaternion<double> RPY2Quat(const Vector3<double>& rpy) const;
  Vector3<double> Quat2RPY(const Quaternion<double>& q) const;

  Vector3<double> fusionRPY;  ///< fusion acc + gyr
  Quaternion<double> fusionQ;
  Vector3<double> accRPY;
  Quaternion<double> accQ;
  mtime_t lastTimestamp;

  double fusionFactor;
};

class VS_EXPORT IMUStabilization {
 public:
  IMUStabilization();
  ~IMUStabilization();

  void addMeasures(const std::vector<VideoStitch::IMU::Measure>& mes);
  Quaternion<double> computeOrientation(mtime_t timestamp);

  /**
   * @brief computes the horizon leveling using last IMU data
   */
  Quaternion<double> computeHorizonLeveling();

  void setLowPassIIR(double ratioNyquist);

  void setFusionFactor(double fusionFactor);

  void enableGyroBiasCorrection();
  void disableGyroBiasCorrection();
  bool isGyroBiasCorrectionValid() const;

  GyroBiasStatus getGyroBiasStatus() const;

 private:
  IMUStabilization(const IMUStabilization& other) = delete;
  IMUStabilization& operator=(const IMUStabilization&) = delete;

  /**
   * @brief Converts IMU quaternion into VS quaternion
   * @param qIMU: input quaternion, expressed in IMU coordinate system
   * @return Quaternion expressed in VS coordinate system
   */
  Quaternion<double> changeCoordinateSystemIMU2Lib(const Quaternion<double>& qIMU) const;

  /**
   * @brief computes the orientation in the IMU coordinate system
   * @param accelerometer: last available accelerometer data, expressed in IMU coordinate system
   * @return Quaternion expressed in IMU coordinate system
   */
  Quaternion<double> computeOrientationFromAccelerometer(const Vector3<double>& accelerometer) const;

  void resetGyroBias();

  std::mutex measuresLock;
  CircularBuffer<VideoStitch::IMU::Measure> measures;                     // raw imu measurements
  CircularBuffer<std::pair<mtime_t, Quaternion<double> > > orientations;  // timestamped quaternions
  std::size_t maxSizeBuffers;

  /// Filters for accelerometer and gyroscope data
  std::unique_ptr<IIRFilter> iirAx;
  std::unique_ptr<IIRFilter> iirAy;
  std::unique_ptr<IIRFilter> iirAz;

  std::unique_ptr<IIRFilter> iirGx;
  std::unique_ptr<IIRFilter> iirGy;
  std::unique_ptr<IIRFilter> iirGz;

  FusionIMU fusion;  ///< computes the fusion between accelerometer and gyroscope data

  bool m_enableGyroBiasCorrection;
  bool m_gyroBiasIsValid;
  int m_gyroBiasNb;  ///< number of gathered measurements for gyro bias computation
  Vector3<double> m_gyroBias;
  Vector3<double> m_previousAcc;
  std::vector<Vector3<double> > m_gyroBiasMeasurements;  ///< gyroscope values gathered for bias cancellation
  double m_threshold_noise_after_bias_correction;

  mutable std::mutex m_statusGyroBiasLock;
  GyroBiasStatus m_statusGyroBias;
};

}  // namespace Stab
}  // namespace VideoStitch

#endif
