// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/quaternion.hpp"

#include <Eigen/Dense>

#include <vector>

namespace VideoStitch {

namespace Stabilization {

typedef Eigen::Matrix<double, 9, 1> Vector9d;

/**
 * @brief The UKF_Quaternion class
 *
 * Inspired by: A Quaternion-Based Unscented Kalman Filter for Orientation Tracking (Edgar Kraft)
 */
class UKF_Quaternion {
 public:
  UKF_Quaternion();
  /**
   * @brief Propagate current state in time
   * @param delta_t : time in seconds
   */
  void predict(double delta_t);

  /**
   * @brief Incorporate a new measurement, update state and covariance matrix
   * @param v : 9 dimensional vector (gyr_xyz, acc_xyz, mag_xyz)
   */
  void measure(const Vector9d& v);

  Quaternion<double> getCurrentOrientation() const;
  Vector3<double> getCurrentAngularVelocity() const;
  Vector3<double> getCurrentAngularAcceleration() const;
  Eigen::Matrix<double, 9, 9> getCovarianceMatrix() const;

 protected:
  class qState {
   public:
    qState();
    void initFromVect(const Vector9d& v);
    void toVect(Vector9d& v) const;

    Quaternion<double> q;  // quaternion orientation
    Vector3<double> w;     // angular velocity
    Vector3<double> ww;    // angular acceleration
  };

  void computeSigmaPoints();
  void computeProcessModel(double delta_t);
  void computeAverageState();
  void computeCovarianceState();
  void computeMeasurements();
  void computeMeanAndCovMeasures();

  qState x;    // current a posteriori state vector
  qState xk_;  // a priori state vector

  Quaternion<double> b;  // orientation of the magnetic north
  Quaternion<double> g;  // orientation of the gravity

  Eigen::Matrix<double, 9, 9> P;    // current covariance of the state vector
  Eigen::Matrix<double, 9, 9> Pk_;  // a priori state vector covariance

  Eigen::Matrix<double, 9, 9> Q;  // process noise
  Eigen::Matrix<double, 9, 9> R;  // measurement noise

  Eigen::Matrix<double, 4, 18> M;  // temporary data used to compute Cholesky decomposition

  Eigen::Matrix<double, 9, 18> W_;  // temporary data used to compute a priori state vector covariance
  std::vector<Vector9d> W;          // temporary data used to compute sigma points
  std::vector<qState> X;            // sigma points {Xi}
  std::vector<qState> Y;            // propagated sigma points {Yi}
  Eigen::Matrix<double, 9, 18> Z;   // projection of the sigma points in the measurement space {Zi}
  std::vector<Vector3<double> > E;  // temporary data used to compute error vectors ei
  Vector9d zk_;                     // mean of {Zi}
  Eigen::Matrix<double, 9, 9> Pvv;  // innovation covariance
  Eigen::Matrix<double, 9, 9>
      Pxz;  // cross-correlation matrix (temporary data used for the computation of the Kalman gain)
  Eigen::Matrix<double, 9, 9> Kk;  // Kalman gain
};

}  // namespace Stabilization
}  // namespace VideoStitch
