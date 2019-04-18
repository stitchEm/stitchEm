// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <iosfwd>

namespace VideoStitch {

/**
 * @brief A 3D vector class.
 */
template <typename T>
class VS_EXPORT Vector3 {
 public:
  /**
   * Creates a vector with the given components.
   */
  Vector3(T v0, T v1, T v2) {
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
  }

  /**
   * Returns the @a i -th component. @a i must be < 3.
   */
  T operator()(int i) const { return v[i]; }

  /**
   * Returns the squared norm of the vector.
   */
  T normSqr() const { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }

  /**
   * Normalizes the vector
   */
  void normalize() {
    T r = norm();
    if (r > 0) {
      v[0] /= r;
      v[1] /= r;
      v[2] /= r;
    }
  }

  /**
   * Returns the euclidean norm of the vector.
   */
  T norm() const { return sqrt((double)normSqr()); }

  /**
   * Vector subtraction and assignment:
   *  v = v - @a o.
   */
  const Vector3& operator-=(const Vector3& o) {
    v[0] -= o.v[0];
    v[1] -= o.v[1];
    v[2] -= o.v[2];
    return *this;
  }

  const Vector3& operator+=(const Vector3& o) {
    v[0] += o.v[0];
    v[1] += o.v[1];
    v[2] += o.v[2];
    return *this;
  }

  const Vector3& operator/=(const T& o) {
    if (o == 0) {
      v[0] = v[1] = v[2] = 0;
      return *this;
    }
    v[0] /= o;
    v[1] /= o;
    v[2] /= o;
    return *this;
  }

  /**
   * Vector subtraction.
   * @param o right hand side
   */
  Vector3 operator-(const Vector3& o) const { return Vector3(v[0] - o.v[0], v[1] - o.v[1], v[2] - o.v[2]); }

  /**
   * Vector addition.
   * @param o right hand side
   */
  Vector3 operator+(const Vector3& o) const { return Vector3(v[0] + o.v[0], v[1] + o.v[1], v[2] + o.v[2]); }

  /**
   * Division by a scalar.
   * @param o right hand side
   */
  Vector3 operator/(const T& o) const {
    if (o == 0) {
      return Vector3<T>(0, 0, 0);
    }
    return Vector3(v[0] / o, v[1] / o, v[2] / o);
  }

  Vector3 operator*(const T& o) const { return Vector3(v[0] * o, v[1] * o, v[2] * o); }

  /**
   * Prints the vector into @a os.
   */
  void print(std::ostream&) const;

 private:
  template <typename>
  friend class Matrix33;
  T v[3];
};

/**
 * Prints @a v into @a os.
 */
template <typename T>
VS_EXPORT std::ostream& operator<<(std::ostream&, const Vector3<T>&);

/**
 * @brief A 3 x 3 matrix class.
 */
template <typename T>
class Matrix33 {
 public:
  /**
   * Matrix:
   *   m00 m01 m02
   *   m10 m11 m12
   *   m20 m21 m22
   */
  Matrix33(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22) {
    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
  }

  /**
   * Creates an identity matrix.
   */
  Matrix33() {
    m[0][0] = (T)1;
    m[0][1] = (T)0;
    m[0][2] = (T)0;
    m[1][0] = (T)0;
    m[1][1] = (T)1;
    m[1][2] = (T)0;
    m[2][0] = (T)0;
    m[2][1] = (T)0;
    m[2][2] = (T)1;
  }

  /**
   * Creates a rotation matrix of t radians around axix X.
   */
  static Matrix33 rotationX(double t) { return Matrix33(1.0, 0.0, 0.0, 0.0, cos(t), sin(t), 0.0, -sin(t), cos(t)); }

  /**
   * Creates a rotation matrix of t radians around axix Y.
   */
  static Matrix33 rotationY(double t) { return Matrix33(cos(t), 0.0, -sin(t), 0.0, 1.0, 0.0, sin(t), 0.0, cos(t)); }

  /**
   * Creates a rotation matrix of t radians around axix Z.
   */
  static Matrix33 rotationZ(double t) { return Matrix33(cos(t), sin(t), 0.0, -sin(t), cos(t), 0.0, 0.0, 0.0, 1.0); }

  /**
   * Copy ctor.
   */
  Matrix33(const Matrix33& o) { memcpy(this->m, o.m, 9 * sizeof(T)); }

  /**
   * Assignement operator.
   */
  Matrix33& operator=(const Matrix33& o) {
    memcpy(this->m, o.m, 9 * sizeof(T));
    return *this;
  }

  /**
   * Returns element (@a row, @a col) of the matrix. Zero-based.
   */
  T operator()(int row, int col) const { return m[row][col]; }

  /**
   * Matrix subtraction.
   * @param o right hand side
   */
  Matrix33 operator-(const Matrix33& o) const {
    return Matrix33(m[0][0] - o.m[0][0], m[0][1] - o.m[0][1], m[0][2] - o.m[0][2],

                    m[1][0] - o.m[1][0], m[1][1] - o.m[1][1], m[1][2] - o.m[1][2],

                    m[2][0] - o.m[2][0], m[2][1] - o.m[2][1], m[2][2] - o.m[2][2]);
  }

  /**
   * @brief Computes the determinant of the matrix
   */
  T det() const {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  }

  /**
   * Returns the result of matrix product of this and @a o.
   * @note Return by copy, not for intensive apps.
   */
  Matrix33 operator*(const Matrix33& o) const {
    return Matrix33(m[0][0] * o.m[0][0] + m[0][1] * o.m[1][0] + m[0][2] * o.m[2][0],
                    m[0][0] * o.m[0][1] + m[0][1] * o.m[1][1] + m[0][2] * o.m[2][1],
                    m[0][0] * o.m[0][2] + m[0][1] * o.m[1][2] + m[0][2] * o.m[2][2],

                    m[1][0] * o.m[0][0] + m[1][1] * o.m[1][0] + m[1][2] * o.m[2][0],
                    m[1][0] * o.m[0][1] + m[1][1] * o.m[1][1] + m[1][2] * o.m[2][1],
                    m[1][0] * o.m[0][2] + m[1][1] * o.m[1][2] + m[1][2] * o.m[2][2],

                    m[2][0] * o.m[0][0] + m[2][1] * o.m[1][0] + m[2][2] * o.m[2][0],
                    m[2][0] * o.m[0][1] + m[2][1] * o.m[1][1] + m[2][2] * o.m[2][1],
                    m[2][0] * o.m[0][2] + m[2][1] * o.m[1][2] + m[2][2] * o.m[2][2]);
  }

  /**
   * Multiply on the right,
   * *this = *this . @a o
   */
  const Matrix33& operator*=(const Matrix33& o) {
    T m00 = m[0][0] * o.m[0][0] + m[0][1] * o.m[1][0] + m[0][2] * o.m[2][0];
    T m01 = m[0][0] * o.m[0][1] + m[0][1] * o.m[1][1] + m[0][2] * o.m[2][1];
    T m02 = m[0][0] * o.m[0][2] + m[0][1] * o.m[1][2] + m[0][2] * o.m[2][2];

    T m10 = m[1][0] * o.m[0][0] + m[1][1] * o.m[1][0] + m[1][2] * o.m[2][0];
    T m11 = m[1][0] * o.m[0][1] + m[1][1] * o.m[1][1] + m[1][2] * o.m[2][1];
    T m12 = m[1][0] * o.m[0][2] + m[1][1] * o.m[1][2] + m[1][2] * o.m[2][2];

    T m20 = m[2][0] * o.m[0][0] + m[2][1] * o.m[1][0] + m[2][2] * o.m[2][0];
    T m21 = m[2][0] * o.m[0][1] + m[2][1] * o.m[1][1] + m[2][2] * o.m[2][1];
    T m22 = m[2][0] * o.m[0][2] + m[2][1] * o.m[1][2] + m[2][2] * o.m[2][2];

    m[0][0] = m00;
    m[0][1] = m01;
    m[0][2] = m02;
    m[1][0] = m10;
    m[1][1] = m11;
    m[1][2] = m12;
    m[2][0] = m20;
    m[2][1] = m21;
    m[2][2] = m22;
    return *this;
  }

  /**
   * Matrix-vector product.
   * @note: Return by value.
   */
  Vector3<T> operator*(const Vector3<T>& v) {
    return Vector3<T>(m[0][0] * v.v[0] + m[0][1] * v.v[1] + m[0][2] * v.v[2],
                      m[1][0] * v.v[0] + m[1][1] * v.v[1] + m[1][2] * v.v[2],
                      m[2][0] * v.v[0] + m[2][1] * v.v[1] + m[2][2] * v.v[2]);
  }

  /**
   * Prints the matrix to @a s.
   */
  void print(std::ostream&) const;

  /**
   * Compute sth einverse of a matrix. Returns false if singular.
   */
  bool inverse(Matrix33& res) const {
    const T det = m[0][0] * (m[2][2] * m[1][1] - m[2][1] * m[1][2]) -
                  m[1][0] * (m[2][2] * m[0][1] - m[2][1] * m[0][2]) + m[2][0] * (m[1][2] * m[0][1] - m[1][1] * m[0][2]);
    if (fabs(det) < 0.00001) {
      return false;
    }
    const T invDet = T(1.0) / det;
    res.m[0][0] = invDet * (m[2][2] * m[1][1] - m[2][1] * m[1][2]);
    res.m[0][1] = -invDet * (m[2][2] * m[0][1] - m[2][1] * m[0][2]);
    res.m[0][2] = invDet * (m[1][2] * m[0][1] - m[1][1] * m[0][2]);
    res.m[1][0] = -invDet * (m[2][2] * m[1][0] - m[2][0] * m[1][2]);
    res.m[1][1] = invDet * (m[2][2] * m[0][0] - m[2][0] * m[0][2]);
    res.m[1][2] = -invDet * (m[1][2] * m[0][0] - m[1][0] * m[0][2]);
    res.m[2][0] = invDet * (m[2][1] * m[1][0] - m[2][0] * m[1][1]);
    res.m[2][1] = -invDet * (m[2][1] * m[0][0] - m[2][0] * m[0][1]);
    res.m[2][2] = invDet * (m[1][1] * m[0][0] - m[1][0] * m[0][1]);
    return true;
  }

  /**
   * Returns the transpose. Note that this is out of place.
   */
  Matrix33 transpose() const {
    return Matrix33(m[0][0], m[1][0], m[2][0],

                    m[0][1], m[1][1], m[2][1],

                    m[0][2], m[1][2], m[2][2]);
  }

  /**
   * Computes the trace of the matrix.
   */
  T trace() const { return m[0][0] + m[1][1] + m[2][2]; }

  /**
   * Assuming that this is a rotation matrix, returns the (canonical) euler angles.
   * Body 3-1-2 parameterization
   * ftp://sbai2009.ene.unb.br/Projects/GPS-IMU/George/arquivos/Bibliografia/79.pdf
   */
  void toEuler(double& yaw, double& pitch, double& roll) const {
    yaw = atan2(m[2][0], m[2][2]);
    pitch = -asin(m[2][1]);
    roll = atan2(m[0][1], m[1][1]);
  }

  /**
   * Creates a rotation matrix from Euler angles (3-1-2 parameterization), with angles given in radians
   */
  static Matrix33 fromEulerZXY(T Y, T X, T Z) { return rotationZ(Z) * rotationX(X) * rotationY(Y); }

  /**
   * Returns a const pointer to the raw array
   */
  const T* data() const { return &m[0][0]; }

 private:
  T m[3][3];
};

/**
 * Prints @a m to @a s.
 */
template <typename T>
VS_EXPORT std::ostream& operator<<(std::ostream&, const Matrix33<T>&);

template <typename T>
Vector3<T> crossVector(const Vector3<T>& a, const Vector3<T>& b) {
  return Vector3<T>(a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2), a(0) * b(1) - a(1) * b(0));
}

template <typename T>
T dotVector(const Vector3<T>& a, const Vector3<T>& b) {
  return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
}
}  // namespace VideoStitch
