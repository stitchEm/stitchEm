// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __clang_analyzer__  // VSA-7040

#include "ukfQuaternion.hpp"

namespace VideoStitch {
namespace Stabilization {
UKF_Quaternion::qState::qState() : q(1., 0., 0., 0.), w(0., 0., 0.), ww(0., 0., 0.) {}

void UKF_Quaternion::qState::initFromVect(const Vector9d &v) {
  Vector3<double> axisAngle(v(0), v(1), v(2));
  q = Quaternion<double>::fromAxisAngle(axisAngle);

  w = Vector3<double>(v(3), v(4), v(5));
  ww = Vector3<double>(v(6), v(7), v(8));
}

void UKF_Quaternion::qState::toVect(Vector9d &v) const {
  Vector3<double> axisAngle = q.toAxisAngle();
  v(0) = axisAngle(0);
  v(1) = axisAngle(1);
  v(2) = axisAngle(2);
  v(3) = w(0);
  v(4) = w(1);
  v(5) = w(2);
  v(6) = ww(0);
  v(7) = ww(1);
  v(8) = ww(2);
}

UKF_Quaternion::UKF_Quaternion() : b(0, 0, 1, 0), g(0, 0, 0, 1) {
  P = Eigen::Matrix<double, 9, 9>::Identity() * 2;
  Q = Eigen::Matrix<double, 9, 9>::Identity() * 0.01;
  R = Eigen::Matrix<double, 9, 9>::Identity() * 0.000001;
  Z = Eigen::Matrix<double, 9, 18>::Identity();

  W.resize(18);
  X.resize(18);
  Y.resize(18);
  E.resize(18, Vector3<double>(0, 0, 0));
}

Quaternion<double> UKF_Quaternion::getCurrentOrientation() const { return x.q; }

Vector3<double> UKF_Quaternion::getCurrentAngularVelocity() const { return x.w; }

Vector3<double> UKF_Quaternion::getCurrentAngularAcceleration() const { return x.ww; }

Eigen::Matrix<double, 9, 9> UKF_Quaternion::getCovarianceMatrix() const { return P; }

void UKF_Quaternion::predict(double delta_t) {
  computeSigmaPoints();
  computeProcessModel(delta_t);
  computeAverageState();
  computeCovarianceState();
  computeMeasurements();
  computeMeanAndCovMeasures();
}

void UKF_Quaternion::measure(const Vector9d &v) {
  Vector9d gain = Kk * (v - zk_);
  qState gainQ;
  gainQ.initFromVect(gain);

  x.q = gainQ.q * xk_.q;
  x.w = gainQ.w + xk_.w;
  x.ww = gainQ.ww + xk_.ww;

  P = Pk_ - Kk * (Pvv * Kk.transpose());
}

void UKF_Quaternion::computeSigmaPoints() {
  Eigen::Matrix<double, 9, 9> S = (P + Q).llt().matrixL();

  for (std::size_t i = 0; i < 9; ++i) {
    Vector9d Wi = sqrt(18.) * S.col(i);
    W[2 * i] = Wi;
    W[2 * i + 1] = -Wi;
  }

  for (std::size_t i = 0; i < 18; ++i) {
    Vector9d &Wi = W[i];
    Quaternion<double> q_wi = Quaternion<double>::fromAxisAngle(Vector3<double>(Wi(0), Wi(1), Wi(2)));
    X[i].q = x.q * q_wi;
    X[i].w = x.w + Vector3<double>(Wi(3), Wi(4), Wi(5));
    X[i].ww = x.ww + Vector3<double>(Wi(6), Wi(7), Wi(8));
  }
}

void UKF_Quaternion::computeProcessModel(double delta_t) {
  for (std::size_t i = 0; i < X.size(); ++i) {
    Vector3<double> wi_current = X[i].w + (X[i].ww * (delta_t * 0.5));

    Quaternion<double> q_delta = Quaternion<double>::fromAxisAngle(wi_current * delta_t);
    Y[i].q = q_delta * X[i].q;
    Y[i].w = X[i].w + X[i].ww * delta_t;
    Y[i].ww = X[i].ww;
  }
}

void UKF_Quaternion::computeAverageState() {
  Quaternion<double> qt = x.q;
  Vector3<double> wt(0, 0, 0);
  Vector3<double> wwt(0, 0, 0);

  for (std::size_t i = 0; i < Y.size(); ++i) {
    M(0, i) = Y[i].q.getQ0();
    M(1, i) = Y[i].q.getQ1();
    M(2, i) = Y[i].q.getQ2();
    M(3, i) = Y[i].q.getQ3();
    wt += Y[i].w;
    wwt += Y[i].ww;
  }
  wt /= static_cast<double>(Y.size());
  wwt /= static_cast<double>(Y.size());

  xk_.q = qt;
  xk_.w = wt;
  xk_.ww = wwt;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 4, 4> > es(M * M.transpose());
  auto v = es.eigenvectors();

  Quaternion<double> qavg(v(0, 3), v(1, 3), v(2, 3), v(3, 3));
  xk_.q = qavg;

  Quaternion<double> qavg_inv = qavg.conjugate();

  for (std::size_t i = 0; i < Y.size(); ++i) {
    E[i] = (Y[i].q * qavg_inv).toAxisAngle();
  }
}

void UKF_Quaternion::computeCovarianceState() {
  Pk_.setZero();
  for (std::size_t i = 0; i < 18; ++i) {
    W_(0, i) = E[i](0);
    W_(1, i) = E[i](1);
    W_(2, i) = E[i](2);
    W_(3, i) = Y[i].w(0) - xk_.w(0);
    W_(4, i) = Y[i].w(1) - xk_.w(1);
    W_(5, i) = Y[i].w(2) - xk_.w(2);
    W_(6, i) = Y[i].ww(0) - xk_.ww(0);
    W_(7, i) = Y[i].ww(1) - xk_.ww(1);
    W_(8, i) = Y[i].ww(2) - xk_.ww(2);
  }

  Pk_ = W_ * W_.transpose();
  Pk_ /= 18;
}

void UKF_Quaternion::computeMeasurements() {
  for (std::size_t i = 0; i < 18; ++i) {
    // gyr
    Z(0, i) = Y[i].w(0);
    Z(1, i) = Y[i].w(1);
    Z(2, i) = Y[i].w(2);

    // acc
    Quaternion<double> zacc = Y[i].q.rotate(g);
    Z(3, i) = zacc.getQ1();
    Z(4, i) = zacc.getQ2();
    Z(5, i) = zacc.getQ3();

    // mag
    Quaternion<double> zmag = Y[i].q.rotate(b);
    Z(6, i) = zmag.getQ1();
    Z(7, i) = zmag.getQ2();
    Z(8, i) = zmag.getQ3();
  }
}

void UKF_Quaternion::computeMeanAndCovMeasures() {
  zk_ = Z.rowwise().mean();
  Z.colwise() -= zk_;
  Pvv = (Z * Z.transpose()) / 18 + R;
  Pxz = (W_ * Z.transpose()) / 18;
  Kk = Pxz * Pvv.inverse();
}

}  // namespace Stabilization
}  // namespace VideoStitch

#endif  // __clang_analyzer__
