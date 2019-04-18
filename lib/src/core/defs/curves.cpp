// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "common/angles.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/projections.hpp"
#include "libvideostitch/curves.hpp"
#include "libvideostitch/ptv.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

namespace VideoStitch {
namespace Core {

namespace {
struct double2 {
  double2(double x, double y) : x(x), y(y) {}
  double x;
  double y;
};
}  // namespace

/**
 * Reads a point.
 * @return false of failure.
 */
template <typename ValueType>
bool readPoint(std::vector<Ptv::Value*>::const_iterator& it, std::vector<Ptv::Value*>::const_iterator end,
               PointTemplate<ValueType>& point);

template <>
bool readPoint<double>(std::vector<Ptv::Value*>::const_iterator& it, std::vector<Ptv::Value*>::const_iterator end,
                       Point& point) {
  if (it == end || (*it)->getType() != Ptv::Value::INT) {
    return false;
  }
  point.t = (int)(*it)->asInt();
  ++it;
  if (it == end || !(*it)->isConvertibleTo(Ptv::Value::DOUBLE)) {
    return false;
  }
  point.v = (*it)->asDouble();
  ++it;
  return true;
}

template <>
bool readPoint<Quaternion<double>>(std::vector<Ptv::Value*>::const_iterator& it,
                                   std::vector<Ptv::Value*>::const_iterator end, QuaternionPoint& point) {
  if (it == end || (*it)->getType() != Ptv::Value::INT) {
    return false;
  }
  point.t = (int)(*it)->asInt();
  ++it;
  if (it == end || !(*it)->isConvertibleTo(Ptv::Value::LIST)) {
    return false;
  }
  std::vector<Ptv::Value*> listVals = (*it)->asList();
  if (listVals.size() != 4 || !listVals[0]->isConvertibleTo(Ptv::Value::DOUBLE) ||
      !listVals[1]->isConvertibleTo(Ptv::Value::DOUBLE) || !listVals[2]->isConvertibleTo(Ptv::Value::DOUBLE) ||
      !listVals[3]->isConvertibleTo(Ptv::Value::DOUBLE)) {
    return false;
  }
  point.v = Quaternion<double>(listVals[0]->asDouble(), listVals[1]->asDouble(), listVals[2]->asDouble(),
                               listVals[3]->asDouble());
  ++it;
  return true;
}

template <>
bool readPoint<GeometryDefinition>(std::vector<Ptv::Value*>::const_iterator& it,
                                   std::vector<Ptv::Value*>::const_iterator end,
                                   Core::PointTemplate<GeometryDefinition>& point) {
  if (it == end || (*it)->getType() != Ptv::Value::INT) {
    return false;
  }
  point.t = (int)(*it)->asInt();
  ++it;
  if (it == end || !(*it)->isConvertibleTo(Ptv::Value::OBJECT)) {
    return false;
  }

  GeometryDefinition def;
  if (!def.applyDiff(*(*it), true).ok()) {
    return false;
  }

  point.v = def;
  ++it;
  return true;
}

/**
 * Reads a spline.
 * @return NULL of failure.
 */
template <typename ValueType>
SplineTemplate<ValueType>* readSpline(SplineTemplate<ValueType>* prev, std::vector<Ptv::Value*>::const_iterator& it,
                                      std::vector<Ptv::Value*>::const_iterator end) {
  if (it == end || (*it)->getType() != Ptv::Value::INT) {
    return NULL;
  }
  typedef SplineTemplate<ValueType> SplineType;
  typename SplineType::Type type = static_cast<typename SplineType::Type>((*it)->asInt());
  ++it;
  PointTemplate<ValueType> point(0, PointTemplate<ValueType>::defaultValue());
  if (!readPoint<ValueType>(it, end, point)) {
    return NULL;
  }
  switch (type) {
    case SplineType::PointType:
      if (prev != NULL) {
        return NULL;
      }
      return SplineType::point(point.t, point.v);
    case SplineType::LineType:
      if (prev == NULL) {
        return NULL;
      }
      return prev->lineTo(point.t, point.v);
    case SplineType::CubicType:
      if (prev == NULL) {
        return NULL;
      }
      return prev->cubicTo(point.t, point.v);
  }
  // This can happend becaus eof the static cast.
  return NULL;
}

template <>
Curve* create(const Ptv::Value& value) {
  // "mycurve": 10 is the constant value 10. (where 10 is a scalar)
  if (value.isConvertibleTo(Ptv::Value::DOUBLE)) {
    return new Curve(value.asDouble());
  }
  // The general syntax is a list of int and double values. We do not
  // use more structure to avoid too many allocations.
  // The splines are flattened in this list. The first element of each
  // spline is its (int) type.
  // Then, depending on the type, one to three pairs of (int, double)
  // elements that each specify a point.
  if (value.isConvertibleTo(Ptv::Value::LIST)) {
    const std::vector<Ptv::Value*>& listValues = value.asList();
    Spline* firstSpline = NULL;
    Spline* prevSpline = NULL;
    for (std::vector<Ptv::Value*>::const_iterator it = listValues.begin(); it != listValues.end(); /*nothing*/) {
      Spline* spline = readSpline(prevSpline, it, listValues.end());
      if (!spline) {
        delete prevSpline;
        return NULL;
      }
      if (firstSpline == NULL) {
        firstSpline = spline;
      }
      prevSpline = spline;
    }
    return new Curve(firstSpline);
  }
  return NULL;
}

template <>
QuaternionCurve* create(const Ptv::Value& value) {
  if (value.isConvertibleTo(Ptv::Value::LIST)) {
    const std::vector<Ptv::Value*>& listValues = value.asList();
    // "mycurve": {1.0, 0.0, 0.0, 0.0} is a constant value.
    if (listValues.size() == 4 && listValues[0]->isConvertibleTo(Ptv::Value::DOUBLE) &&
        listValues[1]->isConvertibleTo(Ptv::Value::DOUBLE) && listValues[2]->isConvertibleTo(Ptv::Value::DOUBLE) &&
        listValues[3]->isConvertibleTo(Ptv::Value::DOUBLE)) {
      return new QuaternionCurve(Quaternion<double>(listValues[0]->asDouble(), listValues[1]->asDouble(),
                                                    listValues[2]->asDouble(), listValues[3]->asDouble()));
    }

    // The general syntax is a list of int and double values. We do
    // not use more structure to avoid too many allocations.
    // The splines are flattened in this list. The first element of
    // each spline is its (int) type.
    // Then, depending on the type, one to three pairs of (int,
    // double) elements that each specify a point.
    SphericalSpline* firstSpline = NULL;
    SphericalSpline* prevSpline = NULL;
    for (std::vector<Ptv::Value*>::const_iterator it = listValues.begin(); it != listValues.end(); /*nothing*/) {
      SphericalSpline* spline = readSpline<Quaternion<double>>(prevSpline, it, listValues.end());
      if (!spline) {
        delete prevSpline;
        return NULL;
      }
      if (firstSpline == NULL) {
        firstSpline = spline;
      }
      prevSpline = spline;
    }
    return new QuaternionCurve(firstSpline);
  }
  return NULL;
}

template <>
GeometryDefinitionCurve* create(const Ptv::Value& value) {
  if (value.isConvertibleTo(Ptv::Value::LIST)) {
    const std::vector<Ptv::Value*>& listValues = value.asList();

    SplineTemplate<GeometryDefinition>* firstSpline = nullptr;
    SplineTemplate<GeometryDefinition>* prevSpline = nullptr;

    for (std::vector<Ptv::Value*>::const_iterator it = listValues.begin(); it != listValues.end(); /*nothing*/) {
      SplineTemplate<GeometryDefinition>* spline = readSpline<GeometryDefinition>(prevSpline, it, listValues.end());
      if (!spline) {
        delete prevSpline;
        return nullptr;
      }
      if (firstSpline == nullptr) {
        firstSpline = spline;
      }
      prevSpline = spline;
    }
    return new GeometryDefinitionCurve(firstSpline);
  } else {
    GeometryDefinition geometry;
    const Status ret = geometry.applyDiff(value, true);
    if (ret.ok()) {
      return new GeometryDefinitionCurve(geometry);
    } else {
      return nullptr;
    }
  }
}

template <>
Ptv::Value* serialize(const Curve* that) {
  Ptv::Value* value = Ptv::Value::emptyObject();
  if (!that->splines()) {
    value->asDouble() = that->at(0);
  } else {
    std::vector<Ptv::Value*>& listVals = value->asList();
    for (const Spline* spline = that->splines(); spline != NULL; spline = spline->next) {
      Ptv::Value* value = Ptv::Value::emptyObject();
      value->asInt() = spline->getType();
      listVals.push_back(value);
      value = Ptv::Value::emptyObject();
      value->asInt() = spline->end.t;
      listVals.push_back(value);
      value = Ptv::Value::emptyObject();
      value->asDouble() = spline->end.v;
      listVals.push_back(value);
    }
  }
  return value;
}

void serializeQuaternion(const Quaternion<double>& q, Ptv::Value* value) {
  std::vector<Ptv::Value*>& listVals = value->asList();
  Ptv::Value* v = Ptv::Value::emptyObject();
  v->asDouble() = q.getQ0();
  listVals.push_back(v);
  v = Ptv::Value::emptyObject();
  v->asDouble() = q.getQ1();
  listVals.push_back(v);
  v = Ptv::Value::emptyObject();
  v->asDouble() = q.getQ2();
  listVals.push_back(v);
  v = Ptv::Value::emptyObject();
  v->asDouble() = q.getQ3();
  listVals.push_back(v);
}

template <>
Ptv::Value* serialize(const QuaternionCurve* that) {
  Ptv::Value* value = Ptv::Value::emptyObject();
  if (!that->splines()) {
    Quaternion<double> q = that->at(0);
    serializeQuaternion(q, value);
  } else {
    std::vector<Ptv::Value*>& listVals = value->asList();
    for (const SphericalSpline* spline = that->splines(); spline != NULL; spline = spline->next) {
      Ptv::Value* val = Ptv::Value::emptyObject();
      val->asInt() = spline->getType();
      listVals.push_back(val);
      val = Ptv::Value::emptyObject();
      val->asInt() = spline->end.t;
      listVals.push_back(val);
      val = Ptv::Value::emptyObject();
      serializeQuaternion(spline->end.v, val);
      listVals.push_back(val);
    }
  }
  return value;
}

template <>
Ptv::Value* serialize(const GeometryDefinitionCurve* that) {
  Ptv::Value* value = Ptv::Value::emptyObject();

  if (!that->splines()) {
    GeometryDefinition q = that->at(0);
    q.serialize(*value);
  } else {
    std::vector<Ptv::Value*>& listVals = value->asList();

    for (const SplineTemplate<GeometryDefinition>* spline = that->splines(); spline != NULL; spline = spline->next) {
      Ptv::Value* val = Ptv::Value::emptyObject();
      val->asInt() = spline->getType();
      listVals.push_back(val);
      val = Ptv::Value::emptyObject();
      val->asInt() = spline->end.t;
      listVals.push_back(val);
      val = Ptv::Value::emptyObject();
      GeometryDefinition q = spline->end.v;
      q.serialize(*val);
      listVals.push_back(val);
    }
  }
  return value;
}

template <typename ValueType>
bool valueEquals(const ValueType& v1, const ValueType& v2) {
  return v1 == v2;
}
template <>
bool valueEquals(const double& v1, const double& v2) {
  return fabs(v1 - v2) < 0.000001;
}

template <typename ValueType>
SplineTemplate<ValueType>* SplineTemplate<ValueType>::point(int t, const ValueType& v) {
  return new SplineTemplate<ValueType>(NULL, PointType, t, v);
}

template <typename ValueType>
SplineTemplate<ValueType>* SplineTemplate<ValueType>::lineTo(int t, const ValueType& v) {
  if (t <= end.t) {
    return NULL;
  }
  SplineTemplate<ValueType>* newSpline = new SplineTemplate<ValueType>(this, LineType, t, v);
  next = newSpline;
  return newSpline;
}

template <typename ValueType>
SplineTemplate<ValueType>* SplineTemplate<ValueType>::cubicTo(int t, const ValueType& v) {
  if (t <= end.t) {
    return NULL;
  }
  SplineTemplate<ValueType>* newSpline = new SplineTemplate<ValueType>(this, CubicType, t, v);
  next = newSpline;
  return newSpline;
}

template <typename ValueType>
SplineTemplate<ValueType>* SplineTemplate<ValueType>::clone(SplineTemplate<ValueType>* prev) const {
  return new SplineTemplate<ValueType>(prev, type, end.t, end.v);
}

template <typename ValueType>
void SplineTemplate<ValueType>::makeLinear() {
  switch (getType()) {
    case PointType:
    case LineType:
      return;
    case CubicType:
      type = LineType;
  }
}

template <typename ValueType>
typename SplineTemplate<ValueType>::Type SplineTemplate<ValueType>::getType() const {
  return type;
}

template <typename ValueType>
SplineTemplate<ValueType>::~SplineTemplate() {
  // NOTE: Not a recursive stack deleter to avoid stack overflows.
  // Find the last spline.
  SplineTemplate* lastSpline = this;
  while (lastSpline->next) {
    lastSpline = lastSpline->next;
  }
  // Delete it recursively, affectively making the one before it the last spline.
  while (lastSpline != this) {
    SplineTemplate* nextLastSpline = lastSpline->prev;
    assert(lastSpline->next == NULL);  // By definition (loop invariant).
    delete lastSpline;                 // Does not recurse since next == NULL.
    nextLastSpline->next = NULL;       // Make it the last spline
    lastSpline = nextLastSpline;
  }
  assert(next == NULL);  // Loop invariant.
}

template <typename ValueType>
bool SplineTemplate<ValueType>::operator==(const SplineTemplate<ValueType>& other) const {
  return type == other.type && end == other.end;
}

template <typename ValueType>
SplineTemplate<ValueType>::SplineTemplate(SplineTemplate<ValueType>* prev, Type type, int endT, const ValueType& endV)
    : prev(prev), next(NULL), end(endT, endV), type(type) {
  if (prev) {
    prev->next = this;
  }
}

template <typename ValueType>
bool PointTemplate<ValueType>::operator==(const PointTemplate<ValueType>& other) const {
  return t == other.t && valueEquals<ValueType>(v, other.v);
}

template class PointTemplate<double>;
template class PointTemplate<Quaternion<double>>;
template class PointTemplate<GeometryDefinition>;

// ------------- Centripetal Catmull-Rom splines ------------------------

// http://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline

// P. J. Barry and R. N. Goldman.
// A recursive evaluation algorithm for a class of catmull-rom splines.
// SIGGRAPH Computer Graphics, 22(4):199{204, 1988.

template <>
void SplineTemplate<double>::makeCubic() {
  switch (getType()) {
    case PointType:
    case CubicType:
      return;
    case LineType:
      if (end.t - prev->end.t >= 2.0) {
        type = CubicType;
      }
  }
}

namespace {
double lerp(double v0, double v1, double t) { return v0 + t * (v1 - v0); }

double catmullRom(double v00, double t0, double v01, double t1, double v02, double t2, double v03, double t3,
                  double t) {
  double v10 = lerp(v00, v01, (t - t0) / (t1 - t0));
  double v11 = lerp(v01, v02, (t - t1) / (t2 - t1));
  double v12 = lerp(v02, v03, (t - t2) / (t3 - t2));
  double v20 = lerp(v10, v11, (t - t0) / (t2 - t0));
  double v21 = lerp(v11, v12, (t - t1) / (t3 - t1));
  return lerp(v20, v21, (t - t1) / (t2 - t1));
}
}  // namespace

template <>
double SplineTemplate<double>::at(int t) const {
  switch (type) {
    case PointType:
      return end.v;
    case LineType:
      if (t <= prev->end.t) {
        return prev->end.v;
      } else if (t >= end.t) {
        return end.v;
      } else {
        return lerp(prev->end.v, end.v, (t - prev->end.t) / (double)(end.t - prev->end.t));
      }
    case CubicType:
      int prevT, nextT;
      double prevV, nextV;
      if (prev->prev != NULL) {
        prevT = prev->prev->end.t;
        prevV = prev->prev->end.v;
      } else {
        // tie it off close to the known values for smooth ease-in
        prevT = prev->end.t - 1;
        prevV = prev->end.v;
      }
      if (next != NULL) {
        nextT = next->end.t;
        nextV = next->end.v;
      } else {
        // tie it off close to the known values for smooth ease-out
        nextT = end.t + 1;
        nextV = end.v;
      }
      return catmullRom(prevV, prevT, prev->end.v, prev->end.t, end.v, end.t, nextV, nextT, t);
  }
  return 0.0;
}

template class SplineTemplate<double>;

// ------------------------ Spherical splines -------------------------

template <>
void SplineTemplate<Quaternion<double>>::makeCubic() {
  // Create the control points through spherical
  // linear interpolation at 1/3 and 2/3
  switch (getType()) {
    case PointType:
    case CubicType:
      return;
    case LineType:
      type = CubicType;
  }
}

template <>
Quaternion<double> SplineTemplate<Quaternion<double>>::at(int t) const {
  switch (type) {
    case PointType:
      return end.v;
    case LineType:
      return Quaternion<double>::slerp(prev->end.v, end.v, (t - prev->end.t) / (double)(end.t - prev->end.t));
    case CubicType:
      const Quaternion<double> prevV = prev->prev ? prev->prev->end.v : prev->end.v;
      const Quaternion<double> nextV = next ? next->end.v : end.v;
      return Quaternion<double>::catmullRom_deprecated(prevV, prev->end.v, prev->end.t, end.v, end.t, nextV, t);
  }
  assert(false);
  return Quaternion<double>();
}

template class SplineTemplate<Quaternion<double>>;

// ------------------------ GeometryDefinition splines -------------------------

template <>
void SplineTemplate<GeometryDefinition>::makeCubic() {}

template <>
GeometryDefinition SplineTemplate<GeometryDefinition>::at(int t) const {
  switch (type) {
    case PointType:
      return end.v;
    case CubicType:
    case LineType:
      if (t <= prev->end.t) {
        return prev->end.v;
      } else if (t >= end.t) {
        return end.v;
      } else {
#define INTERPOLATE_MEMBER(NAME) \
  ret.set##NAME(lerp(prev->end.v.get##NAME(), end.v.get##NAME(), (t - prev->end.t) / (double)(end.t - prev->end.t)))

        GeometryDefinition ret;

        INTERPOLATE_MEMBER(CenterX);
        INTERPOLATE_MEMBER(CenterY);
        if (prev->end.v.hasRadialDistortion() || end.v.hasRadialDistortion()) {
          INTERPOLATE_MEMBER(DistortA);
          INTERPOLATE_MEMBER(DistortB);
          INTERPOLATE_MEMBER(DistortC);
        }
        if (prev->end.v.hasNonRadialDistortion() || end.v.hasNonRadialDistortion()) {
          INTERPOLATE_MEMBER(DistortP1);
          INTERPOLATE_MEMBER(DistortP2);
          INTERPOLATE_MEMBER(DistortS1);
          INTERPOLATE_MEMBER(DistortS2);
          INTERPOLATE_MEMBER(DistortS3);
          INTERPOLATE_MEMBER(DistortS4);
          INTERPOLATE_MEMBER(DistortTau1);
          INTERPOLATE_MEMBER(DistortTau2);
        }
        INTERPOLATE_MEMBER(HorizontalFocal);

        if (prev->end.v.hasVerticalFocal() || end.v.hasVerticalFocal()) {
          INTERPOLATE_MEMBER(VerticalFocal);
        }

        INTERPOLATE_MEMBER(Yaw);
        INTERPOLATE_MEMBER(Pitch);
        INTERPOLATE_MEMBER(Roll);

        if (prev->end.v.hasTranslation() || end.v.hasTranslation()) {
          INTERPOLATE_MEMBER(TranslationX);
          INTERPOLATE_MEMBER(TranslationY);
          INTERPOLATE_MEMBER(TranslationZ);
        }

#undef INTERPOLATE_MEMBER

        return ret;
      }
  }
  assert(false);
  return GeometryDefinition();
}

template class SplineTemplate<GeometryDefinition>;

// ------------------------ Curve implementation ----------------------

template <typename ValueType>
CurveTemplate<ValueType>::CurveTemplate(const ValueType& constant)
    : firstSpline(NULL), lastSpline(NULL), constant(constant) {}

template <typename ValueType>
CurveTemplate<ValueType>::CurveTemplate(SplineTemplate<ValueType>* splines)
    : firstSpline(splines), lastSpline(NULL), constant(PointTemplate<ValueType>::defaultValue()) {
  for (SplineTemplate<ValueType>* spline = splines; spline != NULL; spline = spline->next) {
    if (spline->next == NULL) {
      lastSpline = spline;
    }
  }
}

// Omit from static analysis
// clang-3.7 memory leak possibly false positive
#ifndef __clang_analyzer__

template <typename ValueType>
CurveTemplate<ValueType>* CurveTemplate<ValueType>::clone() const {
  if (!firstSpline) {
    return new CurveTemplate(constant);
  }
  SplineTemplate<ValueType>* splinesCopy = firstSpline->clone(NULL);
  SplineTemplate<ValueType>* curSplineCopy = splinesCopy;
  for (const SplineTemplate<ValueType>* curSpline = firstSpline->next; curSpline != NULL; curSpline = curSpline->next) {
    curSplineCopy = curSpline->clone(curSplineCopy);
  }

  CurveTemplate<ValueType>* ret = new CurveTemplate(splinesCopy);
  ret->setConstantValue(getConstantValue());
  return ret;
}

#endif  // __clang_analyzer__

template <typename ValueType>
CurveTemplate<ValueType>::~CurveTemplate() {
  delete firstSpline;
}

template <typename ValueType>
bool CurveTemplate<ValueType>::operator==(const CurveTemplate& other) const {
  const SplineTemplate<ValueType>* curSpline = splines();
  const SplineTemplate<ValueType>* curOtherSpline = other.splines();
  if (curSpline == NULL) {
    return curOtherSpline == NULL && valueEquals<ValueType>(constant, other.constant);
  }
  while (curSpline != NULL && curOtherSpline != NULL) {
    if (!(*curSpline == *curOtherSpline)) {
      return false;
    }
    curSpline = curSpline->next;
    curOtherSpline = curOtherSpline->next;
  }
  return curSpline == NULL && curOtherSpline == NULL;
}

template <typename ValueType>
bool CurveTemplate<ValueType>::operator!=(const CurveTemplate& other) const {
  return !(*this == other);
}

template <typename ValueType>
SplineTemplate<ValueType>* CurveTemplate<ValueType>::upperSpline(int t) const {
  if (firstSpline == NULL) {
    return NULL;
  }
  if (t <= firstSpline->end.t) {
    return firstSpline;
  }
  // Find the correct spline.
  for (SplineTemplate<ValueType>* spline = firstSpline; spline != NULL; spline = spline->next) {
    if (t <= spline->end.t) {
      return spline;
    }
  }
  return NULL;
}

template <typename ValueType>
ValueType CurveTemplate<ValueType>::at(int t) const {
  if (firstSpline == NULL) {
    return constant;
  }
  const SplineTemplate<ValueType>* spline = upperSpline(t);
  if (spline) {
    return spline->at(t);
  } else {
    return lastSpline->end.v;
  }
}

template <typename ValueType>
const SplineTemplate<ValueType>* CurveTemplate<ValueType>::splines() const {
  return firstSpline;
}

template <typename ValueType>
const SplineTemplate<ValueType>* CurveTemplate<ValueType>::getLastSpline() const {
  return lastSpline;
}

template <typename ValueType>
SplineTemplate<ValueType>* CurveTemplate<ValueType>::splines() {
  return firstSpline;
}

template <typename ValueType>
void CurveTemplate<ValueType>::splitAt(int t) {
  if (firstSpline == NULL) {
    firstSpline = SplineTemplate<ValueType>::point(t, constant);
    lastSpline = firstSpline;
    return;
  }
  SplineTemplate<ValueType>* spline = upperSpline(t);
  if (spline == NULL) {
    // After all splines.
    lastSpline = lastSpline->lineTo(t, lastSpline->end.v);
    return;
  }
  if (t == spline->end.t) {
    // No need to split, already a point.
    return;
  }
  SplineTemplate<ValueType>* prevSpline = spline->prev;
  SplineTemplate<ValueType>* newSpline = NULL;
  if (prevSpline == NULL) {
    // Before all spines.
    newSpline = SplineTemplate<ValueType>::point(t, spline->end.v);
    newSpline->next = spline;
    spline->prev = spline;
    spline->type = SplineTemplate<ValueType>::LineType;
    firstSpline = newSpline;
  } else {
    switch (spline->getType()) {
      case SplineTemplate<ValueType>::PointType:
        assert(false);  // Handled above.
        return;
      case SplineTemplate<ValueType>::LineType:
        newSpline = prevSpline->lineTo(t, spline->at(t));
        break;
      case SplineTemplate<ValueType>::CubicType: {
        newSpline = prevSpline->cubicTo(t, spline->at(t));
        break;
      }
    }
  }
  newSpline->next = spline;
  spline->prev = newSpline;
}

template <typename ValueType>
void CurveTemplate<ValueType>::mergeAt(int t) {
  SplineTemplate<ValueType>* spline = upperSpline(t);
  if (spline == NULL) {
    // After all splines, or no splines.
    return;
  }
  if (t != spline->end.t) {
    // Not an endpoint.
    return;
  }
  SplineTemplate<ValueType>* prevSpline = spline->prev;
  if (!prevSpline) {
    // First spline,
    if (spline->next == NULL) {
      // First and only spline, make constant.
      constant = spline->end.v;
      delete spline;
      firstSpline = NULL;
      lastSpline = NULL;
    } else {
      firstSpline = spline->next;
      spline->next = NULL;
      delete spline;
      firstSpline->prev = NULL;
      firstSpline->type = SplineTemplate<ValueType>::PointType;
    }
    return;
  } else {
    prevSpline->next = spline->next;
    if (spline->next != NULL) {
      spline->next->prev = prevSpline;
    }
    spline->next = NULL;
    delete spline;
    // If spline is the last frame, we need to update the last frame.
    if (spline == lastSpline) {
      lastSpline = prevSpline;
    }
  }
}

template <typename ValueType>
void CurveTemplate<ValueType>::extend(const CurveTemplate* source) {
  if (source->firstSpline == NULL) {
    return;  // Nothing to do.
  }
  if (firstSpline == NULL) {
    // Constant, simply copy all splines.
    firstSpline = source->firstSpline->clone(NULL);
    lastSpline = firstSpline;
    for (const SplineTemplate<ValueType>* curSourceSpline = source->firstSpline->next; curSourceSpline;
         curSourceSpline = curSourceSpline->next) {
      lastSpline = curSourceSpline->clone(lastSpline);
    }
    return;
  }
  // 1 - Handle source completely after us:
  // S:                   x----x----x
  // this: x-----x----x
  //       x-----x----x---x----x----x
  if (source->firstSpline->end.t > lastSpline->end.t) {
    // Cubic to the first source point.
    lastSpline = lastSpline->cubicTo(source->firstSpline->end.t, source->firstSpline->end.v);
    // And simply append the rest.
    for (const SplineTemplate<ValueType>* curSourceSpline = source->firstSpline->next; curSourceSpline;
         curSourceSpline = curSourceSpline->next) {
      lastSpline = curSourceSpline->clone(lastSpline);
    }
    return;
  }

  // 2 - Handle source completely before us:
  // S:    x-----x----x
  // this:                x----x----x
  //       x-----x----x---x----x----x
  if (source->lastSpline->end.t < firstSpline->end.t) {
    SplineTemplate<ValueType>* oldFirstSpline = firstSpline;
    SplineTemplate<ValueType>* oldLastSpline = lastSpline;
    firstSpline = source->firstSpline->clone(NULL);
    lastSpline = firstSpline;
    for (const SplineTemplate<ValueType>* curSourceSpline = source->firstSpline->next; curSourceSpline;
         curSourceSpline = curSourceSpline->next) {
      lastSpline = curSourceSpline->clone(lastSpline);
    }
    // Cubic to the old first point.
    lastSpline = lastSpline->cubicTo(oldFirstSpline->end.t, oldFirstSpline->end.v);
    if (oldFirstSpline->next) {
      oldFirstSpline->next->prev = lastSpline;
      lastSpline->next = oldFirstSpline->next;
      lastSpline = oldLastSpline;
      oldFirstSpline->next = NULL;
    }
    delete oldFirstSpline;
    return;
  }

  // 3 - This is a noop:
  // S:                   x----x----x
  // this: x-----x----x-----x------x----x
  if (source->lastSpline->end.t < lastSpline->end.t && source->firstSpline->end.t > firstSpline->end.t) {
    return;
  }

  // 4 - Also handles case 3, but less efficiently, so we keep 3.
  // S:                   x----x----x
  // this: x-----x----x-----x
  //       x-----x----x-----x--x----x
  if (source->firstSpline->end.t >= firstSpline->end.t) {
    // Discard all sources curves before lastSpline:
    const SplineTemplate<ValueType>* curSourceSpline = NULL;
    for (curSourceSpline = source->firstSpline->next; curSourceSpline && curSourceSpline->end.t < lastSpline->end.t;
         curSourceSpline = curSourceSpline->next) {
    }
    if (curSourceSpline == NULL) {
      return;
    }
    // Join cubicly:
    lastSpline = lastSpline->cubicTo(curSourceSpline->end.t, curSourceSpline->end.v);
    // And append the rest.
    for (curSourceSpline = curSourceSpline->next; curSourceSpline; curSourceSpline = curSourceSpline->next) {
      lastSpline = curSourceSpline->clone(lastSpline);
    }
    return;
  }

  // 5 -
  // S:    x-----x----x-----x
  // this:                x----x----x
  //       x-----x----x---x----x----x
  {
    SplineTemplate<ValueType>* oldFirstSpline = firstSpline;
    SplineTemplate<ValueType>* oldLastSpline = lastSpline;
    // Copy source splines until oldFirstSpline.
    firstSpline = source->firstSpline->clone(NULL);
    lastSpline = firstSpline;
    const SplineTemplate<ValueType>* curSourceSpline = NULL;
    for (curSourceSpline = source->firstSpline->next; curSourceSpline && curSourceSpline->end.t < oldFirstSpline->end.t;
         curSourceSpline = curSourceSpline->next) {
      lastSpline = curSourceSpline->clone(lastSpline);
    }
    // Join cubicly:
    lastSpline = lastSpline->cubicTo(oldFirstSpline->end.t, oldFirstSpline->end.v);
    if (oldFirstSpline->next) {
      oldFirstSpline->next->prev = lastSpline;
      lastSpline->next = oldFirstSpline->next;
      lastSpline = oldLastSpline;
      oldFirstSpline->next = NULL;
    }
    delete oldFirstSpline;
    // Extend to the end.

    // 6 -
    // S:    x-----x----x-----x------x----x
    // this:                x----x----x
    //       x-----x----x---x----x----x---x
    if (source->lastSpline->end.t > lastSpline->end.t) {
      // Discard all source splines before lastSpline.
      for (curSourceSpline = source->firstSpline->next; curSourceSpline && curSourceSpline->end.t <= lastSpline->end.t;
           curSourceSpline = curSourceSpline->next) {
      }
      assert(curSourceSpline);
      // Join cubicly:
      lastSpline = lastSpline->cubicTo(curSourceSpline->end.t, curSourceSpline->end.v);
      // And append the rest.
      for (curSourceSpline = curSourceSpline->next; curSourceSpline; curSourceSpline = curSourceSpline->next) {
        lastSpline = curSourceSpline->clone(lastSpline);
      }
    }
  }
}

template <typename ValueType>
std::string debug(CurveTemplate<ValueType>& curve) {
  std::stringstream graph;
  std::stringstream diag;
  const SplineTemplate<ValueType>* prev = NULL;
  int i = 0;
  for (const SplineTemplate<ValueType>* spline = curve.firstSpline; spline; spline = spline->next) {
    if (spline->prev != prev) {
      diag << "Link broken at " << i << std::endl;
    }
    graph << spline->type << "-> " << spline->end.t << std::endl;
    prev = spline;
    ++i;
  }
  return graph.str() + diag.str();
}

template <typename ValueType>
void CurveTemplate<ValueType>::setConstantValue(const ValueType& value) {
  constant = value;
}

template <typename ValueType>
const ValueType& CurveTemplate<ValueType>::getConstantValue() const {
  return constant;
}

template class CurveTemplate<double>;
template class CurveTemplate<Quaternion<double>>;
template class CurveTemplate<GeometryDefinition>;

#define CONTINUOUS(angle)                                                          \
  {                                                                                \
    if (fabs(angle + 2 * M_PI - prev##angle) < fabs(angle - prev##angle)) {        \
      mod##angle += 2 * M_PI;                                                      \
    } else if (fabs(angle - 2 * M_PI - prev##angle) < fabs(angle - prev##angle)) { \
      mod##angle -= 2 * M_PI;                                                      \
    }                                                                              \
    prev##angle = angle;                                                           \
    angle += mod##angle;                                                           \
    angle = radToDeg(angle);                                                       \
  }

// conversion code from quaternion to euler angles
// since it's mostly use for display, don't stay inside
// the range [-pi,pi]
void toEuler(const QuaternionCurve& qc, Curve** yc, Curve** pc, Curve** rc) {
  // handle constant case.
  if (qc.splines() == NULL) {
    double yaw, pitch, roll;
    qc.at(0).toEuler(yaw, pitch, roll);
    *yc = new Curve(yaw);
    *pc = new Curve(pitch);
    *rc = new Curve(roll);
    return;
  }

  // Non-constant case.
  Spline *yfirst = NULL, *pfirst = NULL, *rfirst = NULL;
  Spline *yspline = NULL, *pspline = NULL, *rspline = NULL;
  double prevyaw = 0.0, prevpitch = 0.0, prevroll = 0.0;
  double modyaw = 0.0, modpitch = 0.0, modroll = 0.0;
  for (const SphericalSpline* spline = qc.splines(); spline; spline = spline->next) {
    const Quaternion<double>& q = spline->end.v;
    double yaw, pitch, roll;
    q.toEuler(yaw, pitch, roll);
    CONTINUOUS(yaw)
    CONTINUOUS(pitch)
    CONTINUOUS(roll)
    switch (spline->getType()) {
      case SphericalSpline::PointType:
        assert(!yspline && !pspline && !rspline);
        assert(!yfirst && !pfirst && !rfirst);
        yspline = Spline::point(spline->end.t, yaw);
        pspline = Spline::point(spline->end.t, pitch);
        rspline = Spline::point(spline->end.t, roll);
        break;
      case SphericalSpline::LineType:
        assert(yspline && pspline && rspline);
        yspline = yspline->lineTo(spline->end.t, yaw);
        pspline = pspline->lineTo(spline->end.t, pitch);
        rspline = rspline->lineTo(spline->end.t, roll);
        break;
      case SphericalSpline::CubicType:
        assert(yspline && pspline && rspline);
        yspline = yspline->cubicTo(spline->end.t, yaw);
        pspline = pspline->cubicTo(spline->end.t, pitch);
        rspline = rspline->cubicTo(spline->end.t, roll);
        break;
    }
    if (yfirst == NULL) {
      yfirst = yspline;
      pfirst = pspline;
      rfirst = rspline;
    }
  }
  assert(yfirst && pfirst && rfirst);
  *yc = new Curve(yfirst);
  *pc = new Curve(pfirst);
  *rc = new Curve(rfirst);
}

template <>
double defaultValue() {
  return 0.0;
}
template <>
Quaternion<double> defaultValue() {
  return Quaternion<double>();
}
template <>
GeometryDefinition defaultValue() {
  return GeometryDefinition();
}

template <typename ValueType>
ValueType PointTemplate<ValueType>::defaultValue() {
  return Core::defaultValue<ValueType>();
}

template double PointTemplate<double>::defaultValue();
template Quaternion<double> PointTemplate<Quaternion<double>>::defaultValue();
template GeometryDefinition PointTemplate<GeometryDefinition>::defaultValue();

template <typename ValueType>
CurveTemplate<ValueType>* CurveTemplate<ValueType>::create(const Ptv::Value& value) {
  return Core::create<CurveTemplate<ValueType>>(value);
}

template Curve* Curve::create(const Ptv::Value&);
template QuaternionCurve* QuaternionCurve::create(const Ptv::Value&);
template GeometryDefinitionCurve* GeometryDefinitionCurve::create(const Ptv::Value&);

template <typename ValueType>
Ptv::Value* CurveTemplate<ValueType>::serialize() const {
  return Core::serialize(this);
}
template Ptv::Value* Curve::serialize() const;
template Ptv::Value* QuaternionCurve::serialize() const;
template Ptv::Value* GeometryDefinitionCurve::serialize() const;

#if defined(_MSC_VER)
template class VS_EXPORT Quaternion<double>;

template class VS_EXPORT CurveTemplate<double>;
template class VS_EXPORT CurveTemplate<Quaternion<double>>;
template class VS_EXPORT CurveTemplate<GeometryDefinition>;

template class VS_EXPORT PointTemplate<double>;
template class VS_EXPORT PointTemplate<Quaternion<double>>;
template class VS_EXPORT PointTemplate<GeometryDefinition>;
#endif
}  // namespace Core
}  // namespace VideoStitch
