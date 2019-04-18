// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef QUATERNION_HPP_
#define QUATERNION_HPP_

#include "matrix.hpp"
#include "config.hpp"
#include "assert.h"

#include <cmath>
#include <iosfwd>

namespace VideoStitch {

/**
 * @brief A quaternion class.
 */
template <typename T>
class VS_EXPORT Quaternion {
 public:
  /**
   * Creates a unit quaternion (1).
   */
  Quaternion() {
    // yaw = pitch = roll = 0
    q0 = 1;
    q1 = q2 = q3 = 0;
  }

  /**
   * Creates a quaternion with given components.
   * @param q0 Real component
   * @param q1 i component
   * @param q2 j component
   * @param q3 k component
   */
  Quaternion(const T q0, const T q1, const T q2, const T q3) : q0(q0), q1(q1), q2(q2), q3(q3) {}

  /**
   * @brief Gives vector-like access to quaternion elements
   * @param i
   *    0: returns q0 (usually you should be using the method this->getQ0() instead)
   *    1: returns q1 (usually you should be using the method this->getQ1() instead)
   *    2: returns q2 (usually you should be using the method this->getQ2() instead)
   *    3: returns q3 (usually you should be using the method this->getQ3() instead)
   *
   * Any other value for @param i is invalid and will return a NaN
   */
  T getQ(int i) const {
    if (i == 0) {
      return q0;
    }
    if (i == 1) {
      return q1;
    }
    if (i == 2) {
      return q2;
    }
    if (i == 3) {
      return q3;
    }
    assert(false);
    return std::numeric_limits<T>::quiet_NaN();
  }

  /**
   * Scalar multiplication.
   * @param s scale factor
   */
  Quaternion operator*(const T s) const { return Quaternion(q0 * s, q1 * s, q2 * s, q3 * s); }

  /**
   * Scalar multiply/assign.
   * @param s scale factor
   */
  void operator*=(const T s) {
    q0 *= s;
    q1 *= s;
    q2 *= s;
    q3 *= s;
  }

  /**
   * Scalar division.
   * @param s inverse scale factor
   */
  Quaternion operator/(const T s) const { return Quaternion(q0 / s, q1 / s, q2 / s, q3 / s); }

  /**
   * Scalar division.
   * @param s inverse scale factor
   */
  void operator/=(const T s) {
    q0 /= s;
    q1 /= s;
    q2 /= s;
    q3 /= s;
  }

  /**
   * Equality operator.
   * WARNING: exact equality.
   * @param rhs right hand side
   */
  bool operator==(const Quaternion& rhs) const {
    return ((q0 - rhs.q0 == 0.0 && q1 - rhs.q1 == 0.0 && q2 - rhs.q2 == 0.0 && q3 - rhs.q3 == 0.0) ||
            (q0 + rhs.q0 == 0.0 && q1 + rhs.q1 == 0.0 && q2 + rhs.q2 == 0.0 && q3 + rhs.q3 == 0.0));
  }

  /**
   * Addition. Commutative.
   * @param rhs right hand side
   */
  Quaternion operator+(const Quaternion& rhs) const {
    return Quaternion(q0 + rhs.q0, q1 + rhs.q1, q2 + rhs.q2, q3 + rhs.q3);
  }

  /**
   * Add/assign.
   * @param rhs right hand side
   */
  void operator+=(const Quaternion& rhs) {
    q0 += rhs.q0;
    q1 += rhs.q1;
    q2 += rhs.q2;
    q3 += rhs.q3;
  }

  /**
   * Subtraction. Commutative.
   * @param rhs right hand side
   */
  Quaternion operator-(const Quaternion& rhs) const {
    return Quaternion(q0 - rhs.q0, q1 - rhs.q1, q2 - rhs.q2, q3 - rhs.q3);
  }

  /**
   * Subtract/assign.
   * @param rhs right hand side
   */
  void operator-=(const Quaternion& rhs) {
    q0 -= rhs.q0;
    q1 -= rhs.q1;
    q2 -= rhs.q2;
    q3 -= rhs.q3;
  }

  /**
   * Returns the opposite quaternion;
   */
  Quaternion operator-() const {
    return Quaternion(-q0, -q1, -q2, -q3);
    ;
  }

  /**
   * @brief negate
   *
   * if q is a rotation quaternion, then q.negate() represents the same rotation
   */
  void negate() {
    q0 = -q0;
    q1 = -q1;
    q2 = -q2;
    q3 = -q3;
  }

  /**
   * Dot product. Commutative.
   * @param rhs right hand side
   */
  T dot(const Quaternion& rhs) const { return q0 * rhs.q0 + q1 * rhs.q1 + q2 * rhs.q2 + q3 * rhs.q3; }

  /**
   * Cross product.
   * WARNING: Multiply semantics are q * r
   * @param rhs right hand side
   */
  Quaternion operator*(const Quaternion& rhs) const {
    return Quaternion(this->q0 * rhs.q0 - this->q1 * rhs.q1 - this->q2 * rhs.q2 - this->q3 * rhs.q3,
                      this->q0 * rhs.q1 + this->q1 * rhs.q0 + this->q2 * rhs.q3 - this->q3 * rhs.q2,
                      this->q0 * rhs.q2 - this->q1 * rhs.q3 + this->q2 * rhs.q0 + this->q3 * rhs.q1,
                      this->q0 * rhs.q3 + this->q1 * rhs.q2 - this->q2 * rhs.q1 + this->q3 * rhs.q0);
  }

  /**
   * Cross product / assign.
   * WARNING: Multiply semantics are q * r
   * @param rhs right hand side
   */
  void operator*=(const Quaternion& rhs) {
    const T mq0 = this->q0 * rhs.q0 - this->q1 * rhs.q1 - this->q2 * rhs.q2 - this->q3 * rhs.q3;
    const T mq1 = this->q0 * rhs.q1 + this->q1 * rhs.q0 + this->q2 * rhs.q3 - this->q3 * rhs.q2;
    const T mq2 = this->q0 * rhs.q2 - this->q1 * rhs.q3 + this->q2 * rhs.q0 + this->q3 * rhs.q1;
    const T mq3 = this->q0 * rhs.q3 + this->q1 * rhs.q2 - this->q2 * rhs.q1 + this->q3 * rhs.q0;
    q0 = mq0;
    q1 = mq1;
    q2 = mq2;
    q3 = mq3;
  }

  /**
   * Conjugation. Note that this is NOT in place.
   */
  Quaternion conjugate() const { return Quaternion(this->q0, -this->q1, -this->q2, -this->q3); }

  /**
   * Norm.
   */
  T norm() const { return std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3); }

  /**
   * Normalize. Note that this is NOT in place.
   */
  Quaternion normalize() { return *this / norm(); }

  /**
   * Normalize. Note that this is NOT in place.
   * Note: if the quaternion is known to be an unit quaternion,
   * use the conjugate directly
   */
  Quaternion inverse() const { return conjugate() / norm(); }

  /**
   * Exponential. Note that typical formulas (e.g. exp(a+b) = exp(a)exp(b)) don't hold because of non-commutativity.
   */
  Quaternion exp() const {
    T angle = std::sqrt(q1 * q1 + q2 * q2 + q3 * q3);
    T sin = std::sin(angle);
    if (sin != 0.0) {
      T coeff = sin / angle;
      return Quaternion(std::cos(angle), coeff * q1, coeff * q2, coeff * q3);
    }
    return Quaternion();
  }

  /**
   * Log. Note that typical formulas (e.g. log(a * b) = log(a)+ log(b)) don't hold because of non-commutativity.
   */
  Quaternion log() const {
    if (q1 == 0.0 && q2 == 0.0 && q3 == 0.0) {
      if (q0 > 0.0) {
        return Quaternion(::log(q0), 0.0, 0.0, 0.0);
      } else if (q0 < 0.0) {
        return Quaternion(::log(-q0), M_PI, 0.0, 0.0);
      } else {
        return Quaternion(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(),
                          std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());
      }
    } else {
      T imag = std::sqrt(q1 * q1 + q2 * q2 + q3 * q3);
      T norm = std::sqrt(imag * imag + q0 * q0);
      T t = std::atan2(imag, T(q0)) / imag;
      return Quaternion(::log(norm), t * q1, t * q2, t * q3);
    }
  }

  /**
   * @brief Initialize a rotation quaternion from an axis-angle representation
   * ||v|| is the angle
   * v / ||v|| is the axis
   */
  static Quaternion fromAxisAngle(const Vector3<T>& v) {
    T angle = v.norm();
    if (angle < 1e-6) {
      Quaternion<T> q;
      return q;
    }

    Quaternion q(std::cos(angle / 2), std::sin(angle / 2) * v(0) / angle, std::sin(angle / 2) * v(1) / angle,
                 std::sin(angle / 2) * v(2) / angle);
    if (q.q0 < 0) {
      q.negate();
    }
    return q;
  }

  /**
   * @brief toAxisAngle
   * @return axis angle representation
   *
   * The norm of the returned vector is the angle. The direction is the rotation axis.
   * This quaternion must be a rotation quaternion (norm 1). Otherwise, the null vector (0, 0 ,0) is returned
   */
  Vector3<T> toAxisAngle() const {
    T x, y, z;
    if (std::abs(this->norm() - 1) > 1e-3) {
      return Vector3<T>(0, 0, 0);
    }

    T angle = std::acos(q0);
    if (angle > M_PI / 2) {
      angle = 2 * angle - 2 * M_PI;
    } else {
      angle = 2 * angle;
    }

    T normV = std::sqrt(1 - q0 * q0);
    x = q1 / normV;
    y = q2 / normV;
    z = q3 / normV;

    return Vector3<T>(x * angle, y * angle, z * angle);
  }

  /**
   * @brief rotate rhs using this as the rotation
   * @param rhs quaternion to be rotated around this
   * @return rotated quaternion
   *
   * this must be a rotation quaternion (norm 1). Otherwise (1, 0, 0, 0) is returned
   * rhs.q0 must be 0 (pure imaginary quaternion). Otherwise (1, 0, 0, 0) is returned
   */
  Quaternion<T> rotate(const Quaternion<T>& rhs) const {
    if (std::abs(norm() - 1) > 1e-6) {
      return Quaternion<T>();
    }
    if (std::abs(rhs.q0) > 1e-6) {
      return Quaternion<T>();
    }

    return ((*this) * (rhs * conjugate()));
  }

  // ------------------- Conversions to Euler angles ------------
  // ftp://sbai2009.ene.unb.br/Projects/GPS-IMU/George/arquivos/Bibliografia/79.pdf
  //
  // We're using the Body 3-1-2 euler angle sequence in VideoStitch
  // In addition, we use the convention that rotation rotates the axes, not
  // the object, so the above formulae actually give the inverse rotation.

  /**
   * Creates from euler angles.
   * @param yaw yaw
   * @param pitch pitch
   * @param roll roll
   */
  static Quaternion fromEulerZXY(const T yaw, const T pitch, const T roll) {
    const T cy = std::cos(yaw * 0.5);
    const T cp = std::cos(pitch * 0.5);
    const T cr = std::cos(roll * 0.5);
    const T sy = std::sin(yaw * 0.5);
    const T sp = std::sin(pitch * 0.5);
    const T sr = std::sin(roll * 0.5);
    Quaternion<T> q = Quaternion(-cr * cp * cy - sr * sp * sy, cr * cy * sp + sr * cp * sy, cr * cp * sy - sr * cy * sp,
                                 -cr * sp * sy + cp * cy * sr);
    if (q.q0 < 0) {
      q.negate();
    }
    return q;
  }

  /**
   * @brief Creates a rotation quaternion from a 3x3 rotation matrix
   * @param R: rotation matrix
   * @return unit quaternion which represents the same rotation
   */
  static Quaternion fromRotationMatrix(const Matrix33<double>& R) {
    T yaw = 0, pitch = 0, roll = 0;
    R.toEuler(yaw, pitch, roll);
    return fromEulerZXY(yaw, pitch, roll);
  }

  /**
   * Converts to (canonical) Euler angles.
   * @param yaw output yaw
   * @param pitch output pitch
   * @param roll output roll
   */
  void toEuler(T& yaw, T& pitch, T& roll) const {
    // Body 3-1-2
    yaw = std::atan2(2.0 * (q1 * q3 - q0 * q2), q3 * q3 - q2 * q2 - q1 * q1 + q0 * q0);
    pitch = -std::asin(2.0 * (q2 * q3 + q0 * q1));
    roll = std::atan2(2.0 * (q1 * q2 - q0 * q3), q2 * q2 - q3 * q3 + q0 * q0 - q1 * q1);
  }

  /**
   * Converts to a (euler) rotation matrix.
   */
  Matrix33<double> toRotationMatrix() const {
    const T q0q0 = q0 * q0;
    const T q0q1 = q0 * q1;
    const T q0q2 = q0 * q2;
    const T q0q3 = q0 * q3;
    const T q1q1 = q1 * q1;
    const T q1q2 = q1 * q2;
    const T q1q3 = q1 * q3;
    const T q2q2 = q2 * q2;
    const T q2q3 = q2 * q3;
    const T q3q3 = q3 * q3;
    return Matrix33<double>(q0q0 + q1q1 - q2q2 - q3q3, 2 * q1q2 - 2 * q0q3, 2 * q1q3 + 2 * q0q2, 2 * q1q2 + 2 * q0q3,
                            q0q0 - q1q1 + q2q2 - q3q3, 2 * q2q3 - 2 * q0q1, 2 * q1q3 - 2 * q0q2, 2 * q2q3 + 2 * q0q1,
                            q0q0 - q1q1 - q2q2 + q3q3);
  }

  /**
   * Find a quaternion representing the rotation between two 3D vectors
   */
  static Quaternion fromTwoVectors(const Vector3<T>& v0, const Vector3<T>& v1) {
    Vector3<T> v0_normalized = v0;
    v0_normalized.normalize();
    Vector3<T> v1_normalized = v1;
    v1_normalized.normalize();

    const Vector3<T> w = crossVector(v0_normalized, v1_normalized);
    const T d01 = dotVector(v0_normalized, v1_normalized);
    Quaternion q = Quaternion(1.f + d01, w(0), w(1), w(2));
    return q.normalize();
  }

  // ------------------- Interpolations --------------------------

  // Ken Shoemake. Animating rotation with quaternion curves, SIGGRAPH '85

  /**
   * Spherical linear interpolation
   * @param q0 quaternion at t=0
   * @param q1 quaternion at t=1
   * @param t interpolation time in [0;1]
   */

  static Quaternion slerp(const Quaternion& q0, Quaternion q1, const T t) {
    T dot = q0.dot(q1);
    if (dot < 0.0) {
      // because the covering is double (q and -q map to the same rotation), the rotation path may turn either the
      // "short way" (less than 180°) or the "long way" (more than 180°). long paths can be prevented by negating one
      // end if the dot product, cos Ω, is negative, thus ensuring that −90° ≤ Ω ≤ 90°.
      dot = -dot;
      q1 = -q1;  // TODO: q1.negate() instead would be faster
    }
    const double omega = std::acos(dot);
    if (VS_ISNAN(omega) || std::abs(omega) < 1e-10) {
      // if the orientations are too close
      // fallback to linear interpolation
      const Quaternion<double> linint = q0 + t * (q1 - q0);
      return linint / linint.norm();
    }
    const double som = std::sin(omega);
    const double st0 = std::sin((1 - t) * omega) / som;
    const double st1 = std::sin(t * omega) / som;
    return q0 * st0 + st1 * q1;
  }

  /**
   * Spherical centripetal catmull-rom interpolation (deprecated)
   *
   * This function is deprecated. Use catmullRom() instead.
   *
   * @param q00 quaternion at previous time
   * @param q01 quaternion at time t=time1
   * @param time1 time for q01
   * @param q02 quaternion at time t=time2
   * @param time2 time for q02
   * @param q03 quaternion at next time
   * @param time query time, in [time1;time2]
   */
  static Quaternion catmullRom_deprecated(const Quaternion& q00, Quaternion q01, const T time1, Quaternion q02,
                                          const T time2, Quaternion q03, const T time) {
    const T t0 = 0;
    T dot = q00.dot(q01);
    if (dot < 0.0) {
      dot = -dot;
      q01 = -q01;
    }
    if (dot > 1.0) {
      dot = 1.0;
    }
    const T t1 = t0 + std::sqrt(std::acos(dot));
    dot = q01.dot(q02);
    if (dot < 0.0) {
      dot = -dot;
      q02 = -q02;
    }
    if (dot > 1.0) {
      dot = 1.0;
    }
    const T t2 = t1 + std::sqrt(std::acos(dot));
    dot = q02.dot(q03);
    if (dot < 0.0) {
      dot = -dot;
      q03 = -q03;
    }
    if (dot > 1.0) {
      dot = 1.0;
    }
    const T t3 = t2 + std::sqrt(std::acos(dot));
    const T t = (time - time1) / (time2 - time1) * (t2 - t1) + t1;
    Quaternion q10 = slerp(q00, q01, t1 > t0 ? (t - t0) / (t1 - t0) : 0);
    Quaternion q11 = slerp(q01, q02, t2 > t1 ? (t - t1) / (t2 - t1) : 0);
    Quaternion q12 = slerp(q02, q03, t3 > t2 ? (t - t2) / (t3 - t2) : 0);
    Quaternion q20 = slerp(q10, q11, t2 > t0 ? (t - t0) / (t2 - t0) : 0);
    Quaternion q21 = slerp(q11, q12, t3 > t1 ? (t - t1) / (t3 - t1) : 0);
    return slerp(q20, q21, t2 > t1 ? (t - t1) / (t2 - t1) : 0);
  }

  /**
   * Spherical centripetal catmull-rom interpolation
   * SCHLAG, J.
   * Using geometric constructions to interpolate orientation with quaternions.
   * Graphics GEMS II, Academic Press, 1992, pp. 377-380.
   *
   * @param q00 quaternion at previous time
   * @param q01 quaternion at time t=0
   * @param q02 quaternion at time t=1
   * @param q03 quaternion at next time
   * @param t query time, in [0;1]
   */
  static Quaternion catmullRom(const Quaternion& q00, const Quaternion& q01, const Quaternion& q02,
                               const Quaternion& q03, const T t) {
    Quaternion q10 = slerp(q00, q01, t + 1);
    Quaternion q11 = slerp(q01, q02, t);
    Quaternion q12 = slerp(q02, q03, t - 1);
    Quaternion q20 = slerp(q10, q11, (t + 1) / 2);
    Quaternion q21 = slerp(q11, q12, t / 2);
    return slerp(q20, q21, t);
  }

  /**
   * Returns the real component.
   */
  const T& getQ0() const { return q0; }

  /**
   * Returns the i component.
   */
  const T& getQ1() const { return q1; }

  /**
   * Returns the j component.
   */
  const T& getQ2() const { return q2; }

  /**
   * Returns the k component.
   */
  const T& getQ3() const { return q3; }

 private:
  T q0, q1, q2, q3;
};

/**
 * Scalar multiplication.
 * @param s scale factor
 * @param q quaternion
 */
template <typename T>
Quaternion<T> operator*(T s, const Quaternion<T>& q) {
  return q * s;
}

/**
 * Output for debug.
 * @param os output stream
 * @param q quaternion.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const Quaternion<T>& q) {
  os << q.getQ0() << ',' << q.getQ1() << ',' << q.getQ2() << ',' << q.getQ3();
  return os;
}
}  // namespace VideoStitch

#endif  // QUATERNION_HPP_
