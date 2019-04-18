// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "object.hpp"
#include "status.hpp"

#include <list>
#include <cmath>

namespace VideoStitch {
namespace Core {

/**
 * A class that represents control points between inputs which are used to align images relative to one other.
 * The coordinates of (alignment) control points are given relatively to the top-left of the image from where they were
 * extracted
 */
class ControlPoint {
 public:
  ControlPoint()
      : error(0.0),
        score(1.0),
        index0(0),
        index1(0),
        x0(0.0),
        y0(0.0),
        x1(0.0),
        y1(0.0),
        rx0(0.),
        ry0(0.),
        rx1(0.),
        ry1(0.),
        frameNumber(-1),
        artificial(false) {}

  ControlPoint(int index0, int index1, double x0, double y0, double x1, double y1, int frameNumber, double error,
               double score, bool artificial = false)
      : error(error),
        score(score),
        index0(index0),
        index1(index1),
        x0(x0),
        y0(y0),
        x1(x1),
        y1(y1),
        rx0(0.),
        ry0(0.),
        rx1(0.),
        ry1(0.),
        frameNumber(frameNumber),
        artificial(artificial) {}

  bool operator==(const ControlPoint& other) const {
    const double eps = 0.001;
    return (index0 == other.index0 && index1 == other.index1 && fabs(x0 - other.x0) < eps &&
            fabs(y0 - other.y0) < eps && fabs(x1 - other.x1) < eps && fabs(y1 - other.y1) < eps &&
            artificial == other.artificial) ||
           (index0 == other.index1 && index1 == other.index0 && fabs(x0 - other.x1) < eps &&
            fabs(y0 - other.y1) < eps && fabs(x1 - other.x0) < eps && fabs(y1 - other.y0) < eps &&
            artificial == other.artificial);
  }

  /**
  Error will be the model<->measure distance
  */
  double error;

  /**
  Score is the matching score, the ratio of distances between the first and second match (uniqueness)
  */
  double score;

  /* Node index of the first image */
  videoreaderid_t index0;

  /* Node index of the second image */
  videoreaderid_t index1;

  /* Coordinates in first image */
  double x0;
  double y0;

  /* Coordinates in second image */
  double x1;
  double y1;

  /* Reprojected coordinates of (x0,y0) in second image */
  double rx0;
  double ry0;

  /* Reprojected coordinates of (x1,y1) in first image */
  double rx1;
  double ry1;

  /* Frame number (defined over time) or -1 if not known*/
  int frameNumber;

  /* Boolean indicating if ControlPoint is artificial or not */
  bool artificial;
};

class ControlPointComparator {
 public:
  bool operator()(const ControlPoint& lhs, const ControlPoint& rhs) { return lhs.score < rhs.score; }
};

typedef std::list<ControlPoint> ControlPointList;

/**
 * @brief The Control Point List definition
 */
class VS_EXPORT ControlPointListDefinition : public Ptv::Object {
 public:
  /**
   * Build from a Ptv::Value.
   * @param value Input value.
   * @return The parsed ControlPointListDefinition.
   */
  static Potential<ControlPointListDefinition> create(const Ptv::Value& value);

  /**
   * Clones a ControlPointListDefinition java-style.
   * @return A similar ControlPointListDefinition. Ownership is given to the caller.
   */
  virtual ControlPointListDefinition* clone() const;

  virtual Ptv::Value* serialize() const;

  /**
   * Comparison operator.
   */
  virtual bool operator==(const ControlPointListDefinition& other) const;

  /**
   * Validates that the ControlPointListDefinition makes sense.
   * @param os The sink for error messages.
   * @return false in case of failure.
   */
  virtual bool validate(std::ostream& os, const videoreaderid_t numVideoInputs) const;

  virtual ~ControlPointListDefinition();

  /**
   * Return the calibration control point list
   */
  virtual const ControlPointList& getCalibrationControlPointList() const;

  /**
   * Sets the calibration control point list
   */
  virtual void setCalibrationControlPointList(const ControlPointList& list);

 protected:
  ControlPointListDefinition();

 private:
  /**
   * Disabled, use clone()
   */
  ControlPointListDefinition(const ControlPointListDefinition&) = delete;

  /**
   * Disabled, use clone()
   */
  ControlPointListDefinition& operator=(const ControlPointListDefinition&) = delete;

  /**
   * Parse from the given ptv. Values not specified are not overridden.
   * @param value Input value.
   */
  Status applyDiff(const Ptv::Value& value);

 private:
  class Pimpl;
  Pimpl* const pimpl;
};

}  // namespace Core
}  // namespace VideoStitch
