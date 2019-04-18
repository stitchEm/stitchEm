// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"
#include "object.hpp"
#include "quaternion.hpp"
#include "projections.hpp"
#include "geometryDef.hpp"

#include <vector>

#define DECLARE_CURVE_WITHOUT_RESETER(name, value_type)                                   \
  /**                                                                                     \
   * Frees the previous curve and replaces it with a new one. Ownership is taken.         \
   * @param newCurve new curve                                                            \
   */                                                                                     \
  virtual void replace##name(CurveTemplate<value_type>* newCurve);                        \
  /**                                                                                     \
   * Same as above, but returns the old curve, which becomes owned the caller.            \
   * @param newCurve new curve                                                            \
   */                                                                                     \
  virtual CurveTemplate<value_type>* displace##name(CurveTemplate<value_type>* newCurve); \
  /**                                                                                     \
   * Returns the curve.                                                                   \
   */                                                                                     \
  virtual const CurveTemplate<value_type>& get##name() const;

#define DECLARE_CURVE(name, value_type)              \
  DECLARE_CURVE_WITHOUT_RESETER(name, value_type)    \
  /**                                                \
   * Resets the curve to its constant default value. \
   */                                                \
  virtual void reset##name();

namespace VideoStitch {
namespace Core {

/**
 * A simple point.
 */
template <typename ValueType>
class VS_EXPORT PointTemplate {
 public:
  /**
   * Creates a point at coordinates (t, v).
   */
  PointTemplate(int t, const ValueType& v) : t(t), v(v) {}
  /**
   * Tests points for equality.
   */
  bool operator==(const PointTemplate& other) const;
  /**
   * Time coordinate.
   */
  int t;
  /**
   * Value coordinate.
   */
  ValueType v;

  /**
   * Return the defaut value for ValueType.
   */
  static ValueType defaultValue();
};

/**
 * Base spline.
 */
template <typename ValueType>
class VS_EXPORT SplineTemplate {
 public:
  /**
   * Value type (e.g. double).
   */
  typedef ValueType value_type;

  /**
   * Type/degree of a spline.
   */
  enum Type { PointType = 0, LineType = 1, CubicType = 3 };

  /**
   * Creates a spline that's a single point.
   */
  static SplineTemplate* point(int t, const ValueType& v);

  /**
   * Creates a linear/slerp spline that links the spline to the given point.
   * @param t Time value. Must be larger than the current time value (strictly).
   * @param v value.
   * @returns NULL on error.
   */
  SplineTemplate* lineTo(int t, const ValueType& v);

  /**
   * Creates a cubic/spherical spline that links the spline to the given point.
   * @param t Time value. Must be larger than the current time value (strictly).
   * @param v value.
   * @returns NULL on error.
   */
  SplineTemplate* cubicTo(int t, const ValueType& v);

  /**
   * Clones a spline and attach it to spline @a prev.
   */
  SplineTemplate* clone(SplineTemplate* prev) const;

  ~SplineTemplate();

  /**
   * Makes a spline linear/spherical linear. Does nothing if the spline is not cubic/spherical quadratic.
   */
  void makeLinear();

  /**
   * Makes a spline cubic/spherical quadratic. Does nothing if the spline is not linear/spherical linear.
   */
  void makeCubic();

  /**
   * Returns the value for time @a i.
   * @param i time. Must be in range.
   */
  ValueType at(int i) const;

  /**
   * Does not test for prev and next equality.
   */
  bool operator==(const SplineTemplate& other) const;

  /**
   * Returns the type of the spline.
   */
  Type getType() const;

  /**
   * Previous spline. If NULL, control points should be ignored, the spline is a single point.
   */

  SplineTemplate* prev;
  /**
   * Next spline.
   */
  SplineTemplate* next;
  /**
   * End point.
   */
  PointTemplate<ValueType> end;

 private:
  /**
   * Generic spline from the endpoint of a previous spline.
   * @param prev Previous spline. NULL means that the spline is a single point.
   */
  SplineTemplate(SplineTemplate* prev, Type type, int endT, const ValueType& endV);

  /**
   * Degree of the spline.
   */
  Type type;

  template <typename>
  friend class CurveTemplate;
};

/**
 * A cubic spline.
 */
typedef SplineTemplate<double> Spline;

/**
 * A Quaternion Spline.
 */
typedef SplineTemplate<Quaternion<double> > SphericalSpline;

/**
 * @brief An immutable class to represent a time-dependant parameter.
 * The parameters are represented as piecewise splines. The value is extended outside of the domain specified by the
 * splines as a constant value. Two levels of usage: 1 - Simple interface through at(). You don't have to care about how
 * the curves are represented internally. 2 - More powerful interface where you can explicitly manipulate the splines.
 */
template <typename ValueType>
class VS_EXPORT CurveTemplate : public Ptv::Object {
 public:
  /**
   * Creates a constant curve with no splines.
   */
  explicit CurveTemplate(const ValueType& constant);

  /**
   * Creates from splines. Takes ownership of splines.
   */
  explicit CurveTemplate(SplineTemplate<ValueType>* splines);

  /**
   * Clones the Curve.
   */
  CurveTemplate* clone() const;

  /**
   * Parses from PTV.
   */
  static CurveTemplate* create(const Ptv::Value& value);

  ~CurveTemplate();

  /**
   * Comparison operators.
   * @{
   */
  bool operator==(const CurveTemplate& other) const;
  bool operator!=(const CurveTemplate& other) const;
  /**
   * @}
   */

  Ptv::Value* serialize() const;

  // Simplified interface.
  /**
   * Get the value of the parameter at the specified time.
   * Interpolate out-of-bounds values as constant.
   */
  ValueType at(int t) const;

  // Direct interface.
  /**
   * Returns a const linked list of splines.
   * Warning: can be null, in which case the curve is a constant whose value can be retreived with at().
   */
  const SplineTemplate<ValueType>* splines() const;

  /**
   * Returns the last spline, with null successor.
   * Warning: can be null, in which case the curve is a constant whose value can be retreived with at().
   */
  const SplineTemplate<ValueType>* getLastSpline() const;

  /**
   * Returns a linked list of splines.
   * Warning: can be null, in which case the curve is a constant whose value can be retreived with at().
   * You can create the first spline by splitting at the desired position.
   */
  SplineTemplate<ValueType>* splines();

  /**
   * Splits the curve by splitting the spline at the given position in two splines.
   * The overall shape is not changed. If @a t falls on an already existing endpoint, does nothing.
   */
  void splitAt(int t);

  /**
   * If there is an endpoint under @a t, merge the two surrounding splines into one.
   * If @a t does not falls on an already existing endpoint, does nothing.
   */
  void mergeAt(int t);

  /**
   * Sets the const value for the curve. Only useful if the curve has no splines.
   * @param value constant value.
   */
  void setConstantValue(const ValueType& value);

  /**
   * Returns the const value for the curve. Only useful if the curve has no splines.
   */
  const ValueType& getConstantValue() const;

  // Other

  /**
   * Modifies a curve by extending it beyond its bounds using a source curve.
   * The splines from the source curve that fall outside of the current curve's bounds are simply cloned and inserted in
   * the current curve.
   * @param source Source curve.
   */
  void extend(const CurveTemplate* source);

  /**
   * Returns the spline that falls under @a t (i.e. the first spline whose endpoint is >= t)
   * Returns the first spline if @a t is before all splines, and NULL if it is after all splines.
   */
  SplineTemplate<ValueType>* upperSpline(int t) const;

 private:
  SplineTemplate<ValueType>* firstSpline;
  SplineTemplate<ValueType>* lastSpline;
  ValueType constant;
};

template <typename Curve>
Curve* create(const Ptv::Value&);

template <typename Curve>
Ptv::Value* serialize(const Curve*);

/**
 * Reads a spline.
 * @return NULL of failure.
 */
template <typename ValueType>
Core::SplineTemplate<ValueType>* readSpline(Core::SplineTemplate<ValueType>* prev,
                                            std::vector<Ptv::Value*>::const_iterator& it,
                                            std::vector<Ptv::Value*>::const_iterator end);

#if defined(_MSC_VER)
#else
extern template class VS_EXPORT CurveTemplate<double>;
extern template class VS_EXPORT CurveTemplate<Quaternion<double> >;
extern template class VS_EXPORT CurveTemplate<GeometryDefinition>;
#endif

typedef PointTemplate<double> Point;
typedef PointTemplate<Quaternion<double> > QuaternionPoint;
typedef SplineTemplate<double> Spline;
typedef SplineTemplate<Quaternion<double> > QuaternionSpline;
typedef CurveTemplate<double> Curve;
typedef CurveTemplate<Quaternion<double> > QuaternionCurve;
typedef CurveTemplate<GeometryDefinition> GeometryDefinitionCurve;

// conversion code from quaternion to euler angles
void VS_EXPORT toEuler(const QuaternionCurve&, Curve** yaw, Curve** pitch, Curve** roll);

template <typename ValueType>
ValueType defaultValue();

#if defined(_MSC_VER)
#else
extern template class VS_EXPORT PointTemplate<double>;
extern template class VS_EXPORT PointTemplate<Quaternion<double> >;
extern template class VS_EXPORT PointTemplate<GeometryDefinition>;
#endif

}  // namespace Core
}  // namespace VideoStitch
