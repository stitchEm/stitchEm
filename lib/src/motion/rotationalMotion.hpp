// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gme.hpp"

#include "libvideostitch/quaternion.hpp"
#include "libvideostitch/status.hpp"

#include <map>
#include <vector>
#include <random>

namespace VideoStitch {

namespace Core {
class InputDefinition;
class PanoDefinition;
}  // namespace Core

namespace Motion {

/**
 * @brief Detects the global rotation of a panoramic video
 * The motion model is rotational and computed from samples from
 * any inputs, reprojected on the sphere.
 *
 * It uses the method "Least-Square Rigid Motion Using SVD" : https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
 * restricted to a pure rotational motion: translation is assumed to be 0 (all the points lie on the unit sphere)
 *
 */
class RotationRansac {
 public:
  /**
   * @brief RotationRansac
   * @param field: the vector of quaternion pairs (from -> to).
   * @param inlierThreshold: allowed angular discrepancy between a point and its reprojection (1.5° is a reasonable
   * value)
   * @param numIters: number of iterations of the ransac loop
   * @param minConsensusSamples: minimum number of inliers for the ransac to be considered successful
   * @param gen: random number generator
   */
  RotationRansac(const SphericalSpace::MotionVectorField& field, double inlierThreshold, int numIters,
                 std::size_t minConsensusSamples, std::default_random_engine& gen)
      : inlierThreshold(inlierThreshold),
        minSamplesForFit(2),
        numIters(numIters),
        minConsensusSamples(minConsensusSamples),
        field(field),
        gen(gen) {}

  ~RotationRansac() {}

  /**
   * @brief Computes ransac on the motion vector field
   * @param qRot: output quaternion which represents the best quaternion according to the set of inliers
   * @return true if the estimation is successful, false otherwise
   */
  bool ransac(Quaternion<double>& qRot);

 private:
  /**
   * @brief find the best rotation given the list of samples of bitSet
   * @return true if everything went OK, false is an error has occurred
   */
  bool fit(Quaternion<double>& qRot, const std::vector<bool>& bitSet) const;

  bool isConsensualSample(Quaternion<double>& qRot, SphericalSpace::MotionVector mv) const;

  /**
   * @brief Populate bitSet randomly with the right number of bits
   * @param numBitsSets: exact number of bits to be set to 1
   * @param bitSet: vector filled by this function. Must be resized prior to the call to this function
   *
   * @return code: true if everything was OK, false otherwise
   *
   */
  bool populateRandom(size_t numBitsSets, std::vector<bool>& bitSet);

  const double inlierThreshold;
  const std::size_t minSamplesForFit;
  const int numIters;
  const std::size_t minConsensusSamples;
  const SphericalSpace::MotionVectorField& field;
  std::default_random_engine& gen;
};

/**
 * A class that detects the global motion of a panoramic video.
 * The motion model is rotational and computed from samples from
 * any inputs, reprojected on the sphere.
 */
class RotationalMotionModelEstimation {
 public:
  typedef std::map<int64_t, Quaternion<double> > MotionModel;

  explicit RotationalMotionModelEstimation(const Core::PanoDefinition& panorama);
  virtual ~RotationalMotionModelEstimation() {}

  Status motionModel(std::vector<std::pair<ImageSpace::MotionVectorFieldTimeSeries, const Core::InputDefinition*> >&,
                     MotionModel&) const;
  Status motionModel(std::vector<std::pair<ImageSpace::MotionVectorField, const Core::InputDefinition*> >& in, int time,
                     Quaternion<double>&) const;
  Status motionModel(const SphericalSpace::MotionVectorFieldTimeSeries&, MotionModel&) const;
  Status motionModel(const SphericalSpace::MotionVectorField&, Quaternion<double>&) const;

 private:
  void transform(const ImageSpace::MotionVectorField&, const Core::InputDefinition&, int time,
                 SphericalSpace::MotionVectorField&) const;

  const Core::PanoDefinition& panorama;

  RotationalMotionModelEstimation();
  RotationalMotionModelEstimation(const RotationalMotionModelEstimation&);
};
}  // namespace Motion
}  // namespace VideoStitch
