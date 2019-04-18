// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/orah/imuStabilization.hpp"

#include "libvideostitch/logging.hpp"

#include "stabilization/iirFilter.hpp"

namespace VideoStitch {
namespace Stab {

// hard threshold to zero-out low gyroscope values
static const double THR_GYR_NOISE_WITHOUT_BIAS_CORRECTION_SQUARED = 500 * 500 * IMU::IMU_RATIO_GYR * IMU::IMU_RATIO_GYR;
static const double ACC_NOISE_DELTA_THR_SQUARED = 0.1 * 0.1;

// 0: use gyroscope only     1: use accelerometer only
static const double IMU_FUSION_FACTOR = 0.09;

// used to filter out accelerometer and gyroscope IMU data using IIR biquad filters
static const double IIR_LOWPASS_FILTER_RATIO_NYQUIST = 0.3;

// Number of measurements used to compute the gyroscope bias correction
static const int GYRO_BIAS_NB_REQUIRED_MEASUREMENTS = 150;

///////////////////////////////////////////////////////

FusionIMU::FusionIMU()
    : initialized(false),
      fusionRPY(0., 0., 0.),
      fusionQ(1., 0., 0., 0.),
      accRPY(0., 0., 0.),
      accQ(1., 0., 0., 0.),
      lastTimestamp(0),
      fusionFactor(IMU_FUSION_FACTOR) {}

Quaternion<double> FusionIMU::RPY2Quat(const Vector3<double> &rpy) const {
  double cosX_2 = std::cos(rpy(0) / 2.0f);
  double sinX_2 = std::sin(rpy(0) / 2.0f);
  double cosY_2 = std::cos(rpy(1) / 2.0f);
  double sinY_2 = std::sin(rpy(1) / 2.0f);
  double cosZ_2 = std::cos(rpy(2) / 2.0f);
  double sinZ_2 = std::sin(rpy(2) / 2.0f);

  double q0 = cosX_2 * cosY_2 * cosZ_2 + sinX_2 * sinY_2 * sinZ_2;
  double q1 = sinX_2 * cosY_2 * cosZ_2 - cosX_2 * sinY_2 * sinZ_2;
  double q2 = cosX_2 * sinY_2 * cosZ_2 + sinX_2 * cosY_2 * sinZ_2;
  double q3 = cosX_2 * cosY_2 * sinZ_2 - sinX_2 * sinY_2 * cosZ_2;
  Quaternion<double> q(q0, q1, q2, q3);
  q = q.normalize();
  return q;
}

Vector3<double> FusionIMU::Quat2RPY(const Quaternion<double> &q) const {
  double roll = std::atan2(2.0 * (q.getQ2() * q.getQ3() + q.getQ0() * q.getQ1()),
                           1 - 2.0 * (q.getQ1() * q.getQ1() + q.getQ2() * q.getQ2()));

  double pitch = std::asin(2.0 * (q.getQ0() * q.getQ2() - q.getQ1() * q.getQ3()));

  double yaw = std::atan2(2.0 * (q.getQ1() * q.getQ2() + q.getQ0() * q.getQ3()),
                          1 - 2.0 * (q.getQ2() * q.getQ2() + q.getQ3() * q.getQ3()));

  return Vector3<double>(roll, pitch, yaw);
}

void FusionIMU::computeOrientationFromAcc(const Vector3<double> &accelerometer) {
  Vector3<double> acc = accelerometer;
  acc.normalize();
  double roll = std::atan2(acc(1), acc(2));
  double pitch = -std::atan2(acc(0), std::sqrt(acc(1) * acc(1) + acc(2) * acc(2)));
  double yaw = fusionRPY(2);
  accRPY = Vector3<double>(roll, pitch, yaw);
  accQ = RPY2Quat(accRPY);

  int maxIndex = 0;
  double maxAbsVal = std::abs(accQ.getQ0());
  for (int i = 1; i < 4; ++i) {
    if (std::abs(accQ.getQ(i)) > maxAbsVal) {
      maxAbsVal = std::abs(accQ.getQ(i));
      maxIndex = i;
    }
  }

  if (((accQ.getQ(maxIndex) > 0) && (fusionQ.getQ(maxIndex) < 0)) ||
      ((accQ.getQ(maxIndex) < 0) && (fusionQ.getQ(maxIndex) > 0))) {
    accQ.negate();
    accRPY = Quat2RPY(accQ);
  }
}

Quaternion<double> FusionIMU::init(const Vector3<double> &acc, mtime_t timestamp) {
  initialized = true;
  lastTimestamp = timestamp;
  computeOrientationFromAcc(acc);
  fusionQ = accQ;
  fusionRPY = accRPY;
  return fusionQ;
}

Quaternion<double> FusionIMU::update(const Vector3<double> &gyr, const Vector3<double> &acc, mtime_t timestamp) {
  if (!initialized) {
    std::ostringstream oss;
    oss << "FusionIMU: make sure init() is called before update(). ";
    oss << "Using the first accelerometer values as an initilization" << std::endl;
    Logger::get(Logger::Warning) << oss.str();
    return init(acc, timestamp);
  }

  double delta_t = (timestamp - lastTimestamp) / 1000000.;
  if (delta_t <= 0) {
    std::ostringstream oss;
    oss << "FusionIMU: negative delta timestamp: " << timestamp << " - " << lastTimestamp << std::endl;
    Logger::get(Logger::Error) << oss.str();
    lastTimestamp = timestamp;
    return fusionQ;
  }

  lastTimestamp = timestamp;

  computeOrientationFromAcc(acc);

  Quaternion<double> qOffset(0, gyr(0), gyr(1), gyr(2));
  qOffset = fusionQ * qOffset;
  qOffset *= 0.5 * delta_t;
  fusionQ += qOffset;

  fusionQ = Quaternion<double>::slerp(fusionQ.normalize(), accQ, fusionFactor);

  fusionRPY = Quat2RPY(fusionQ);
  return fusionQ;
}

bool FusionIMU::usesGyroscope() const { return fusionFactor < 1.0; }

void FusionIMU::setTimestamp(mtime_t timestamp) { lastTimestamp = timestamp; }

void FusionIMU::setFusionFactor(double factor) {
  if ((factor < 0) || (factor > 1)) {
    std::ostringstream oss;
    oss << "FusionIMU::setFusionFactor invalid argument (outside of acceptable range [0; 1]) : ";
    oss << factor << std::endl;
    Logger::get(Logger::Error) << oss.str();
    return;
  }
  fusionFactor = factor;
}

/////////////////

IMUStabilization::IMUStabilization()
    : maxSizeBuffers(3600),
      iirAx(new IIRFilter),
      iirAy(new IIRFilter),
      iirAz(new IIRFilter),
      iirGx(new IIRFilter),
      iirGy(new IIRFilter),
      iirGz(new IIRFilter),
      m_enableGyroBiasCorrection(false),
      m_gyroBiasIsValid(false),
      m_gyroBiasNb(0),
      m_gyroBias(0., 0., 0.),
      m_previousAcc(0., 0., 0.),
      m_threshold_noise_after_bias_correction(THR_GYR_NOISE_WITHOUT_BIAS_CORRECTION_SQUARED),
      m_statusGyroBias(GyroBiasStatus::DISABLED) {
  iirAx->initLowPass(IIR_LOWPASS_FILTER_RATIO_NYQUIST);
  iirAy->initLowPass(IIR_LOWPASS_FILTER_RATIO_NYQUIST);
  iirAz->initLowPass(IIR_LOWPASS_FILTER_RATIO_NYQUIST);

  iirGx->initLowPass(IIR_LOWPASS_FILTER_RATIO_NYQUIST);
  iirGy->initLowPass(IIR_LOWPASS_FILTER_RATIO_NYQUIST);
  iirGz->initLowPass(IIR_LOWPASS_FILTER_RATIO_NYQUIST);
}

IMUStabilization::~IMUStabilization() {}

void IMUStabilization::setLowPassIIR(double ratioNyquist) {
  iirAx->initLowPass(ratioNyquist);
  iirAy->initLowPass(ratioNyquist);
  iirAz->initLowPass(ratioNyquist);
  iirGx->initLowPass(ratioNyquist);
  iirGy->initLowPass(ratioNyquist);
  iirGz->initLowPass(ratioNyquist);
}

void IMUStabilization::setFusionFactor(double fusionFactor) { fusion.setFusionFactor(fusionFactor); }

void IMUStabilization::enableGyroBiasCorrection() {
  resetGyroBias();
  m_enableGyroBiasCorrection = true;

  std::lock_guard<std::mutex> lock(m_statusGyroBiasLock);
  m_statusGyroBias = GyroBiasStatus::IN_PROGRESS;
}

void IMUStabilization::disableGyroBiasCorrection() {
  m_enableGyroBiasCorrection = false;

  std::lock_guard<std::mutex> lock(m_statusGyroBiasLock);
  m_statusGyroBias = GyroBiasStatus::DISABLED;
}

bool IMUStabilization::isGyroBiasCorrectionValid() const {
  bool enabledAndValid = m_enableGyroBiasCorrection && m_gyroBiasIsValid;

  std::lock_guard<std::mutex> lock(m_statusGyroBiasLock);
  if (enabledAndValid) {
    assert(m_statusGyroBias == GyroBiasStatus::OK);
  } else {
    assert(m_statusGyroBias != GyroBiasStatus::OK);
  }
  return enabledAndValid;
}

GyroBiasStatus IMUStabilization::getGyroBiasStatus() const {
  std::lock_guard<std::mutex> lock(m_statusGyroBiasLock);
  return m_statusGyroBias;
}

Quaternion<double> IMUStabilization::changeCoordinateSystemIMU2Lib(const Quaternion<double> &qIMU) const {
  return Quaternion<double>(qIMU.getQ0(), qIMU.getQ2(), -qIMU.getQ3(), -qIMU.getQ1());
}

void IMUStabilization::resetGyroBias() {
  m_gyroBiasNb = 0;
  m_gyroBiasMeasurements.clear();
  m_gyroBiasMeasurements.reserve(GYRO_BIAS_NB_REQUIRED_MEASUREMENTS);
  m_gyroBiasIsValid = false;
  m_threshold_noise_after_bias_correction = THR_GYR_NOISE_WITHOUT_BIAS_CORRECTION_SQUARED;
  m_gyroBias = Vector3<double>(0., 0., 0.);
}

Quaternion<double> IMUStabilization::computeOrientationFromAccelerometer(const Vector3<double> &accelerometer) const {
  Vector3<double> acc = accelerometer;
  acc.normalize();

  Vector3<double> g(0, 0, 1);
  Vector3<double> vec = crossVector(acc, g);
  double angle = std::acos(dotVector(g, acc));
  vec.normalize();
  vec = vec * angle;
  Quaternion<double> qIMU = Quaternion<double>::fromAxisAngle(vec);
  return qIMU;
}

void IMUStabilization::addMeasures(const std::vector<VideoStitch::IMU::Measure> &mes) {
  std::lock_guard<std::mutex> lock(measuresLock);

  for (auto m : mes) {
    measures.push(m);
    if (measures.size() > maxSizeBuffers) {
      measures.erase(1);
    }

    Vector3<double> acc = m.getAcc();
    Vector3<double> gyr = m.getGyr();

    if (m_enableGyroBiasCorrection && !m_gyroBiasIsValid) {
      if (m_gyroBiasNb == 0) {
        // first measurement is always OK
        m_previousAcc = acc;
      }
      // update gyro bias
      Vector3<double> deltaAcc = acc - m_previousAcc;
      m_previousAcc = acc;
      if ((deltaAcc.normSqr() < ACC_NOISE_DELTA_THR_SQUARED) &&
          (gyr.normSqr() < THR_GYR_NOISE_WITHOUT_BIAS_CORRECTION_SQUARED)) {
        m_gyroBias += gyr / GYRO_BIAS_NB_REQUIRED_MEASUREMENTS;
        m_gyroBiasMeasurements.push_back(gyr);

        m_gyroBiasNb++;
        if (m_gyroBiasNb == GYRO_BIAS_NB_REQUIRED_MEASUREMENTS) {
          double max_normsqr = -1;
          for (std::size_t indexGyr = 0; indexGyr < GYRO_BIAS_NB_REQUIRED_MEASUREMENTS; ++indexGyr) {
            double current_normsqr = (m_gyroBiasMeasurements[indexGyr] - m_gyroBias).normSqr();
            if (max_normsqr < current_normsqr) {
              max_normsqr = current_normsqr;
            }
          }

          m_threshold_noise_after_bias_correction = 3 * max_normsqr;
          if (m_threshold_noise_after_bias_correction > THR_GYR_NOISE_WITHOUT_BIAS_CORRECTION_SQUARED) {
            m_threshold_noise_after_bias_correction = THR_GYR_NOISE_WITHOUT_BIAS_CORRECTION_SQUARED;
            std::ostringstream oss;
            oss << "Gyroscope bias correction high, set default threshold instead: "
                << m_threshold_noise_after_bias_correction << " <> " << THR_GYR_NOISE_WITHOUT_BIAS_CORRECTION_SQUARED
                << std::endl;
            Logger::get(Logger::Info) << oss.str();
            disableGyroBiasCorrection();
            std::lock_guard<std::mutex> lock(m_statusGyroBiasLock);
            m_statusGyroBias = GyroBiasStatus::THRESHOLD_TOO_HIGH;
          } else {
            std::ostringstream oss;
            oss << "Threshold gyro noise: " << m_threshold_noise_after_bias_correction
                << "  (compare to: " << THR_GYR_NOISE_WITHOUT_BIAS_CORRECTION_SQUARED << ")" << std::endl;
            Logger::get(Logger::Verbose) << oss.str();
            m_gyroBiasIsValid = true;
            std::lock_guard<std::mutex> lock(m_statusGyroBiasLock);
            m_statusGyroBias = GyroBiasStatus::OK;
          }
        }
      } else {
        Logger::get(Logger::Warning) << "Could not compute gyroscope bias correction (camera is moving ?)" << std::endl;
        resetGyroBias();
        disableGyroBiasCorrection();
        std::lock_guard<std::mutex> lock(m_statusGyroBiasLock);
        m_statusGyroBias = GyroBiasStatus::CAMERA_MOVING;
      }
    }

    if (m_enableGyroBiasCorrection && m_gyroBiasIsValid) {
      // use gyro bias
      Logger::get(Logger::Debug) << "using computed gyro bias" << std::endl;
      gyr -= m_gyroBias;
      if (gyr.normSqr() < m_threshold_noise_after_bias_correction) {
        gyr = Vector3<double>(0, 0, 0);
      }

    } else {
      // no gyro bias, use default threshold
      Logger::get(Logger::Debug) << "no gyro bias, using default threshold" << std::endl;
      if (gyr.normSqr() < THR_GYR_NOISE_WITHOUT_BIAS_CORRECTION_SQUARED) {
        gyr = Vector3<double>(0, 0, 0);
      }
    }

    double axiir = iirAx->filterValue(acc(0));
    double ayiir = iirAy->filterValue(acc(1));
    double aziir = iirAz->filterValue(acc(2));

    double gxiir = iirGx->filterValue(gyr(0));
    double gyiir = iirGy->filterValue(gyr(1));
    double gziir = iirGz->filterValue(gyr(2));

    // disable the wobbling due to accelerometer noise: do not change the orientation if the camera is static
    if (fusion.usesGyroscope() && (gyr.normSqr() == 0)) {
      if (!orientations.empty()) {
        fusion.setTimestamp(m.timestamp);
        std::pair<mtime_t, Quaternion<double> > currentOrientation = orientations[orientations.size() - 1];
        currentOrientation.first = m.timestamp;
        orientations.push(currentOrientation);
        if (orientations.size() > maxSizeBuffers) {
          orientations.erase(1);
        }
        return;
      }
    }

    acc = Vector3<double>(axiir, ayiir, aziir);
    gyr = Vector3<double>(gxiir, gyiir, gziir);

    Quaternion<double> qIMU;
    if (measures.size() == 1) {
      qIMU = fusion.init(acc, m.timestamp);
    } else {
      qIMU = fusion.update(gyr, acc, m.timestamp);
    }

    Quaternion<double> qLib = changeCoordinateSystemIMU2Lib(qIMU);
    std::pair<mtime_t, Quaternion<double> > p = std::make_pair(m.timestamp, qLib);
    orientations.push(p);
    if (orientations.size() > maxSizeBuffers) {
      orientations.erase(1);
    }
  }
}

Quaternion<double> IMUStabilization::computeHorizonLeveling() {
  VideoStitch::IMU::Measure lastMes;
  {
    std::lock_guard<std::mutex> lock(measuresLock);
    if (measures.empty()) {
      return Quaternion<double>(1, 0, 0, 0);
    }
    lastMes = measures[measures.size() - 1];
  }

  Quaternion<double> qIMU = computeOrientationFromAccelerometer(lastMes.getAcc());
  Quaternion<double> qLib = changeCoordinateSystemIMU2Lib(qIMU);
  return qLib;
}

Quaternion<double> IMUStabilization::computeOrientation(mtime_t timestamp) {
  std::lock_guard<std::mutex> lock(measuresLock);
  assert(measures.size() == orientations.size());
  while ((orientations.size() > 1) && (orientations[1].first < timestamp)) {
    orientations.erase(1);
    measures.erase(1);
  }

  if (orientations.empty()) {
    Logger::get(Logger::Warning) << "Could not stabilize using IMU: orientations is empty" << std::endl;
    return Quaternion<double>(1., 0., 0., 0.);
  }

  if (orientations[0].first > timestamp) {
    std::ostringstream oss;
    oss << "Could not stabilize using IMU: current timestamp " << timestamp;
    oss << " is before the first IMU timestamp " << orientations[0].first << std::endl;
    Logger::get(Logger::Warning) << oss.str();
    return Quaternion<double>(1., 0., 0., 0.);
  }
  if (orientations[0].first == timestamp) {
    return orientations[0].second;
  }

  if (orientations.size() == 1) {
    if (orientations[0].first < timestamp) {
      std::ostringstream oss;
      oss << "Could not stabilize using IMU: current timestamp " << timestamp;
      oss << " is after the last IMU timestamp " << orientations[0].first << std::endl;
      Logger::get(Logger::Warning) << oss.str();
      return Quaternion<double>(1., 0., 0., 0.);
    }

    assert(false);  ///< we should never reach this point
  }

  if (orientations[1].first == timestamp) {
    return orientations[1].second;
  }

  assert(orientations[0].first != orientations[1].first);
  assert(orientations[0].first < timestamp);
  assert(orientations[1].first > timestamp);

  double t = static_cast<double>(timestamp - orientations[0].first) / (orientations[1].first - orientations[0].first);
  return Quaternion<double>::slerp(orientations[0].second, orientations[1].second, t);
}

}  // namespace Stab
}  // namespace VideoStitch
