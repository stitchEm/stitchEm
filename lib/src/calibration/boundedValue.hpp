// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef __BOUNDEDVALUE__
#define __BOUNDEDVALUE__

#include <array>
#include <memory>
#include <Eigen/Dense>

namespace VideoStitch {
namespace Calibration {

/**
@brief A bounded value is a value (real scalar) which is forced to lie on a bounded domain.
@details We use a 2D rotation matrix. This matrix is the "optimized" content.
@details This rotation matrix rotates the vector (0; 1).
@details Keeping only the X value, the initial domain is [-1;1].
@details This is then scaled and shifted so the final domain is [min;max]
*/
class BoundedValue {
 public:
  BoundedValue();
  virtual ~BoundedValue();

  /**
  @brief Returns whether the BoundedValue is constant
  */
  bool isConstant() const;

  /**
  @brief Set bounds to be min and max
  @param min the minimal allowed value
  @param max the maximal allowed value
  */
  void setBounds(double min, double max);

  /**
  @brief Estimate the double scalar from the matrix
  @return estimated value
  */
  double getValue() const;

  /**
  @brief Jacobian of value wrt matrix
  @param J the jacobian updated
  */
  void getJacobian(Eigen::Matrix<double, 1, 4> &J) const;

  /**
  @brief Update the rotation matrix so that the estimated value is val
  @param val the updated value
  */
  void setValue(double val);

  /**
  @brief Retrieve the content of the rotation matrix
  @return pointer to rotation matrix
  */
  double *getMinimizerPtr() const;

  /**
  @brief Update the content of the rotation matrix
  @param ptr the rotation matrix content to use
  */
  void setMinimizerValues(const double *ptr);

  /**
  @brief Tie the bounded value to another one (i.e. will share the underlying objects)
  @param other the other bounded value to be tied to
  */
  void tieTo(const BoundedValue &other);

  /**
  @brief Untie the bounded value (i.e. will have its own values independent from other objects)
  */
  void untie();

 private:
  double shift;
  double scale;
  std::shared_ptr<std::array<double, 4> > modifier;
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
