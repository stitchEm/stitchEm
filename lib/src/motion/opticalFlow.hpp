// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "motion/gme.hpp"
#include "common/queue.hpp"

#include "libvideostitch/inputDef.hpp"

namespace VideoStitch {
namespace Motion {

struct OpticalFlow {
  OpticalFlow() : input(0), frame(0), inputDef(NULL) {}

  OpticalFlow(int in, int fr, Motion::ImageSpace::MotionVectorField fi, const Core::InputDefinition* id)
      : input(in), frame(fr), field(fi), inputDef(id) {}

  float computeMedianMagnitude2() const;

  std::string toString() const;

  /**
   * @brief Applies an upscaling factor to the motion vectors (both source and destination)
   *
   * If the images were downscaled before the computation of the optical flow,
   * use this method to get the correspondences coordinates back to the reference of
   * the original image.
   */
  void applyFactor(float factor);

  /**
   * @brief Randomly select nbSamplesToKeep motion vectors
   * @param nbSamplesToKeep : number of motion vectors to keep.
   *                          If this value is larger than field.size(), then all
   *                          motion vectors are kept
   */
  void sampleMotionVectors(std::size_t nbSamplesToKeep);

  void filterSmallMotions(double epsilon = 1e-6);

  int input;
  int frame;
  Motion::ImageSpace::MotionVectorField field;
  const Core::InputDefinition* inputDef;
};

struct OpticalFlowCompare {
  bool operator()(const OpticalFlow& lhs, const OpticalFlow& rhs) const { return lhs.frame > rhs.frame; }
};

typedef PriorityQueue<OpticalFlow, OpticalFlowCompare> OpticalFlowQueue;

}  // namespace Motion
}  // namespace VideoStitch
