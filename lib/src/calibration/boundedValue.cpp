// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "boundedValue.hpp"

#include <Eigen/Dense>
#include <iostream>

#include <limits>

namespace VideoStitch {
namespace Calibration {

BoundedValue::BoundedValue() : modifier(new std::array<double, 4>) {
  /*By default, bound is [-1; 1]*/
  shift = 0;
  scale = 1;
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > R(modifier->data());
  R.setIdentity();
}

BoundedValue::~BoundedValue() {}

bool BoundedValue::isConstant() const { return (scale <= std::numeric_limits<double>::epsilon()); }

void BoundedValue::setBounds(double min, double max) {
  shift = 0.5 * (min + max);
  scale = 0.5 * (max - min);
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > R(modifier->data());
  R.setIdentity();
}

double *BoundedValue::getMinimizerPtr() const { return modifier->data(); }

void BoundedValue::setMinimizerValues(const double *ptr) {
  (*modifier)[0] = ptr[0];
  (*modifier)[1] = ptr[1];
  (*modifier)[2] = ptr[2];
  (*modifier)[3] = ptr[3];
}

double BoundedValue::getValue() const {
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > R(modifier->data());

  Eigen::Vector2d vec;
  vec(0) = 0;
  vec(1) = 1;

  Eigen::Vector2d updated_vec = (Eigen::Vector2d)(R * vec);

  double val = scale * updated_vec(0) + shift;
  return val;
}

void BoundedValue::getJacobian(Eigen::Matrix<double, 1, 4> &J) const {
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > R(modifier->data());

  Eigen::Vector2d vec;
  vec(0) = 0;
  vec(1) = 1;
  Eigen::Vector2d updated_vec = (Eigen::Vector2d)(R * vec);

  /* scale [1 0; 0 0] * update * R * vec */
  /* scale * R(0, 0) * updated_vec(0) +  scale * R(0, 1) * updated_vec(1) */

  J(0, 0) = scale * updated_vec(0);
  J(0, 1) = 0;
  J(0, 2) = scale * updated_vec(1);
  J(0, 3) = 0;
}

void BoundedValue::setValue(double val) {
  Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor> > R(modifier->data());

  /*Scale == 0 means we have no uncertainty, shift IS the value*/
  if (fabs(scale) < 1e-12) {
    R.setIdentity();
    return;
  }

  double b = (val - shift) / scale;
  double angle = -asin(b);

  assert(std::abs(b) <= 1.);
  R(0, 0) = cos(angle);
  R(0, 1) = -sin(angle);
  R(1, 0) = sin(angle);
  R(1, 1) = cos(angle);
}

void BoundedValue::tieTo(const BoundedValue &other) {
  shift = other.shift;
  scale = other.scale;
  modifier = other.modifier;
}

void BoundedValue::untie() {
  // create a new modifier object, with the content of the old one
  std::shared_ptr<std::array<double, 4> > tmp_copy = modifier;
  modifier.reset(new std::array<double, 4>);
  *modifier = *tmp_copy;
}

}  // namespace Calibration
}  // namespace VideoStitch
