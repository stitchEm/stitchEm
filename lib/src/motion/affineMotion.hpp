// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gme.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/matrix.hpp"

#include <vector>
#include <random>

namespace VideoStitch {

namespace Core {
class InputDefinition;
class PanoDefinition;
}  // namespace Core

namespace Motion {

/**
 * A class that detects the motion of a single video.
 * The motion model is affine?
 */
class AffineMotionModelEstimation {
 public:
  typedef ImageSpace::MotionVector MotionVector;
  typedef ImageSpace::MotionVectorField MotionVectorField;
  typedef ImageSpace::MotionVectorFieldTimeSeries MotionVectorFieldTimeSeries;
  typedef std::map<int64_t, std::pair<bool, Matrix33<double> > > MotionModel;

  AffineMotionModelEstimation() {}
  virtual ~AffineMotionModelEstimation() {}

  static void motionModel(const MotionVectorFieldTimeSeries&, MotionModel&, const Core::InputDefinition&);

  static Status motionModel(const ImageSpace::MotionVectorField& field, Matrix33<double>& affine,
                            const Core::InputDefinition&);

 private:
  AffineMotionModelEstimation(const AffineMotionModelEstimation&);
};
}  // namespace Motion
}  // namespace VideoStitch
