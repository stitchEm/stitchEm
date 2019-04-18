// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rotationalMotion.hpp"

//#define RANSAC_VERBOSE
#include "util/ransac.hpp"
#include "core/geoTransform.hpp"

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/logging.hpp"

#include <opencv2/core/core.hpp>

namespace VideoStitch {
namespace Motion {

bool RotationRansac::ransac(Quaternion<double>& qRot) {
  if (field.empty()) {
    Logger::get(Logger::Error) << "RotationRansac: the motion field is empty. Ransac has failed" << std::endl;
    return false;
  }

  std::size_t bestNumConsensual = 0;

  // Check that the input motionField is valid
  for (std::size_t i = 0; i < field.size(); ++i) {
    if ((std::abs(field[i].from.getQ0()) > 1e-6) || (std::abs(field[i].to.getQ0()) > 1e-6) ||
        (std::abs(field[i].from.norm() - 1.) > 1e-6) || (std::abs(field[i].to.norm() - 1.) > 1e-6)) {
      std::ostringstream oss;
      oss << "RotationRansac::ransac(): Invalid quaternion pair: ";
      oss << field[i].from << " -> " << field[i].to;
      Logger::get(Logger::Error) << oss.str() << std::endl;
      return false;
    }
  }

  if (minSamplesForFit > field.size()) {
    std::ostringstream oss;
    oss << "RotationRansac::ransac(): minSamplesForFit (" << minSamplesForFit << ") ";
    oss << "is larger than the number of samples (" << field.size() << ")." << std::endl;
    oss << "RotationRansac::ransac() has failed";
    Logger::get(Logger::Error) << oss.str() << std::endl;
    return false;
  }

  std::vector<bool> bitSet(field.size());
  std::vector<bool> bestBitSet;

  for (int iter = 0; iter < numIters; ++iter) {
    if (!populateRandom(minSamplesForFit, bitSet)) {
      return false;
    }
    Quaternion<double> currentRot;
    if (!fit(currentRot, bitSet)) {
      return false;
    }

    std::size_t numConsensual = 0;
    for (size_t i = 0; i < field.size(); ++i) {
      if (bitSet[i] || isConsensualSample(currentRot, field[i])) {
        ++numConsensual;
        bitSet[i] = true;
      } else {
        bitSet[i] = false;
      }
    }

    if (numConsensual > minConsensusSamples && numConsensual > bestNumConsensual) {
      bestBitSet = bitSet;
      bestNumConsensual = numConsensual;
    }
  }

  if (bestNumConsensual == 0) {
    return false;
  }

  if (bestBitSet.size() != field.size()) {
    Logger::get(Logger::Error) << "bestBitSet size (" << bestBitSet.size() << ") != field size (" << field.size() << ")"
                               << std::endl;
    return false;
  }
  if (!fit(qRot, bestBitSet)) {
    Logger::get(Logger::Error) << "RotationRansac::ransac(): rotation computed on the best set of inliers has failed"
                               << std::endl;
    return false;
  }
#ifndef NDEBUG
  Logger::get(Logger::Debug) << "RotationRansac::OK: nb inliers: " << bestNumConsensual << " / " << field.size()
                             << std::endl;
#endif

  return true;
}

bool RotationRansac::fit(Quaternion<double>& qRot, const std::vector<bool>& bitSet) const {
  if (bitSet.size() != field.size()) {
    std::ostringstream oss;
    oss << "RotationRansac::fit(): bitSet size (" << bitSet.size() << ") != field size (" << field.size() << ")";
    Logger::get(Logger::Error) << oss.str() << std::endl;
    return false;
  }
  int count = 0;
  for (std::size_t i = 0; i < field.size(); ++i) {
    if (bitSet[i]) {
      count++;
    }
  }

  cv::Mat A(count, 3, CV_64F);
  cv::Mat B(count, 3, CV_64F);

  int currentRow = 0;
  for (std::size_t i = 0; i < field.size(); ++i) {
    if (bitSet[i]) {
      A.at<double>(currentRow, 0) = field[i].from.getQ1();
      A.at<double>(currentRow, 1) = field[i].from.getQ2();
      A.at<double>(currentRow, 2) = field[i].from.getQ3();

      B.at<double>(currentRow, 0) = field[i].to.getQ1();
      B.at<double>(currentRow, 1) = field[i].to.getQ2();
      B.at<double>(currentRow, 2) = field[i].to.getQ3();

      currentRow++;
    }
  }

  cv::SVD svd(A.t() * B);
  cv::Mat Rcv = svd.vt.t() * svd.u.t();

  double det = cv::determinant(Rcv);
  if (det < 0) {
    // Achtung! We found a reflection, not a rotation
    svd.vt.at<double>(2, 0) *= -1.;
    svd.vt.at<double>(2, 1) *= -1.;
    svd.vt.at<double>(2, 2) *= -1.;
    Rcv = svd.vt.t() * svd.u.t();
  }

  Matrix33<double> R(Rcv.at<double>(0, 0), Rcv.at<double>(0, 1), Rcv.at<double>(0, 2), Rcv.at<double>(1, 0),
                     Rcv.at<double>(1, 1), Rcv.at<double>(1, 2), Rcv.at<double>(2, 0), Rcv.at<double>(2, 1),
                     Rcv.at<double>(2, 2));
  if (std::abs(R.det() - 1.) > 1e-6) {
    std::ostringstream oss;
    oss << "RotationRansac::ransac(): R is not rotation matrix" << std::endl;
    oss << R << std::endl;
    Logger::get(Logger::Error) << oss.str();
    return false;
  }

  qRot = Quaternion<double>::fromRotationMatrix(R);
  if (std::abs(qRot.norm() - 1.0) > 1e-6) {
    Logger::get(Logger::Error) << "RotationRansac::ransac(): quaternion is not a proper rotation: " << qRot
                               << std::endl;
    return false;
  }

  return true;
}

bool RotationRansac::isConsensualSample(Quaternion<double>& qRot, SphericalSpace::MotionVector mv) const {
  Quaternion<double> dest = qRot * mv.from * qRot.conjugate();

  double angle = acos(dest.dot(mv.to));
  if (angle < inlierThreshold) {
    return true;
  }
  return false;
}

bool RotationRansac::populateRandom(size_t numBitsSets, std::vector<bool>& bitSet) {
  if (numBitsSets > bitSet.size()) {
    std::ostringstream oss;
    oss << "RotationRansac::populateRandom() : numBitsSets (" << numBitsSets << ") > bitSet.size(";
    oss << bitSet.size() << ")";
    Logger::get(Logger::Error) << oss.str() << std::endl;
    return false;
  }
  for (size_t i = 0; i < numBitsSets; ++i) {
    bitSet[i] = true;
  }
  for (size_t i = numBitsSets; i < bitSet.size(); ++i) {
    bitSet[i] = false;
  }
  std::shuffle(bitSet.begin(), bitSet.end(), gen);
  return true;
}

namespace {

/**
 * Non-linear optimization problem for camera rotation estimation.
 * See http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2001/Schmidt01-UQF.pdf
 */
class RotationEstimationProblem : public Util::SolverProblem {
 public:
  RotationEstimationProblem(const SphericalSpace::MotionVectorField& field, const Quaternion<double>& h0)
      : field(field), h0(h0) {
    // compute an orthogonal basis for the tangential hyperplane
    const cv::Matx<double, 4, 3> bp(-h0.getQ1() / h0.getQ0(), -h0.getQ2() / h0.getQ0(), -h0.getQ3() / h0.getQ0(), 1, 0,
                                    0, 0, 1, 0, 0, 0, 1);
    cv::SVD svd(bp);
    B = svd.u;
  }

  int numParams() const { return 3; }

  int getNumInputSamples() const { return (int)field.size(); }

  int getNumValuesPerSample() const { return 1; }

  int getNumAdditionalValues() const { return 0; }

  /* reprojection error */
  void eval(const double* params, int m_dat, double* fvec, const char* fFilter, int /*iterNum*/, bool*) const {
    const Quaternion<double> hz = paramToQuaternion(params);

    for (size_t i = 0; i < (size_t)m_dat; ++i) {
      if (!fFilter || fFilter[i]) {
        Quaternion<double> reproj = hz * field[i].from * hz.conjugate();
        fvec[i] = (field[i].to - reproj).norm();
      } else {
        // not part of the random sample
        fvec[i] = 0.0;
      }
    }
  }

  template <typename Array>
  Quaternion<double> paramToQuaternion(const Array& v) const {
    // convert the 3-vector v to a 4-vector v4
    // v4 = B.v
    const cv::Matx<double, 4, 1> vec = B * cv::Vec3d(v[0], v[1], v[2]);
    Quaternion<double> v4(vec(0, 0), vec(1, 0), vec(2, 0), vec(3, 0));
    // normalize v4
    double theta = v4.norm();
    if (theta != 0) {
      v4 /= theta;
    }
    // compute the resulting quaternion hZ from v4
    // hZ = sin(θ) v4 + cos(θ) h0 where θ is the norm of v4
    return sin(theta) * v4 + cos(theta) * h0;
  }

  const SphericalSpace::MotionVectorField& field;
  cv::Matx<double, 4, 3> B;
  Quaternion<double> h0;
};

/**
 * Solver for the optimization problem consisting of fitting a rotational
 * motion model to the spherical motion vectors field, using least squares
 * reprojection error as the quantity to minimize.
 * The Ransac paradigm removes outliers due to for example objects in
 * movement.
 */
class RotationEstimationRansacSolver : public Util::RansacSolver<Util::LmminSolver<Util::SolverProblem>> {
 public:
  // Ransac: For the problem to be at least constrained, we need: params.size() elements.
  RotationEstimationRansacSolver(const Util::SolverProblem& problem, int numIters, int minConsensusSamples,
                                 double sphereDist, bool debug = false)
      : Util::RansacSolver<Util::LmminSolver<Util::SolverProblem>>(problem, 3, numIters, minConsensusSamples, nullptr,
                                                                   debug),
        inlierThreshold(sphereDist * 0.026185) {
    getControl() = lm_control_double;
  }

 private:
  bool isConsensualSample(double* values) const {
    // Maximum re-projection error value to classify as inlier.
    // This corresponds to a 1.5° error
    return values[0] < inlierThreshold;
  }

  const double inlierThreshold;
};
}  // namespace

RotationalMotionModelEstimation::RotationalMotionModelEstimation(const Core::PanoDefinition& pano) : panorama(pano) {}

/**
 * Transform the image space motion vectors to perspective space
 * motion vectors before modelling the rotation.
 */
Status RotationalMotionModelEstimation::motionModel(
    std::vector<std::pair<ImageSpace::MotionVectorFieldTimeSeries, const Core::InputDefinition*>>& in,
    MotionModel& model) const {
  // map the points correspondences in image-space to their coordinates
  // in spherical-space
  SphericalSpace::MotionVectorFieldTimeSeries sphericalFieldTimeSeries;
  for (size_t i = 0; i < in.size(); ++i) {
    const ImageSpace::MotionVectorFieldTimeSeries& ts = in[i].first;
    const Core::InputDefinition& im = *in[i].second;
    for (auto mvfield = ts.begin(); mvfield != ts.end(); ++mvfield) {
      SphericalSpace::MotionVectorField sphericalField;
      transform(mvfield->second, im, (int)mvfield->first, sphericalField);
      sphericalFieldTimeSeries[mvfield->first] = sphericalField;
    }
  }

  return motionModel(sphericalFieldTimeSeries, model);
}

Status RotationalMotionModelEstimation::motionModel(
    std::vector<std::pair<ImageSpace::MotionVectorField, const Core::InputDefinition*>>& in, int time,
    Quaternion<double>& h) const {
  SphericalSpace::MotionVectorField sphericalField;
  for (unsigned i = 0; i < in.size(); ++i) {
    transform(in[i].first, *in[i].second, time, sphericalField);
  }

  return motionModel(sphericalField, h);
}

Status RotationalMotionModelEstimation::motionModel(const SphericalSpace::MotionVectorFieldTimeSeries& timeSeries,
                                                    MotionModel& model) const {
  for (auto mvfield = timeSeries.begin(); mvfield != timeSeries.end(); ++mvfield) {
    Quaternion<double> h;
    if (!motionModel(mvfield->second, h).ok()) {
      Logger::get(Logger::Warning) << "Could not estimate rotation for frame " << mvfield->first << ", skipping."
                                   << std::endl;
      continue;
    }
    model[mvfield->first] = h;
    // operating point is the value of the last time the algorithm was run successfully
  }
  return Status::OK();
}

Status RotationalMotionModelEstimation::motionModel(const SphericalSpace::MotionVectorField& field,
                                                    Quaternion<double>& rotation) const {
  if (field.empty()) {
    rotation = Quaternion<double>(0.0, 0.0, 0.0, 0.0);
    return {Origin::StabilizationAlgorithm, ErrType::AlgorithmFailure,
            "Could not estimate rotational motion model. Encountered empty motion vector field."};
  }
  double sphereDist = field[0].from.norm();
  if (std::abs(sphereDist - 1.) > 1e-6) {
    return {Origin::StabilizationAlgorithm, ErrType::AlgorithmFailure, "Points do not lie on the unit sphere"};
  }

  std::default_random_engine gen(42);
  // We require 50% of the motion vectors to be consensuals.
  Motion::RotationRansac rotationRansac(field, 0.026185, 100, field.size() / 2, gen);
  Quaternion<double> qRot;
  bool ransacResult = rotationRansac.ransac(qRot);

  if (ransacResult) {
    rotation = qRot;
  } else {
    rotation = Quaternion<double>(0.0, 0.0, 0.0, 0.0);
    return {Origin::StabilizationAlgorithm, ErrType::AlgorithmFailure,
            "Could not estimate rotational motion model. RANSAC failed."};
  }
  return Status::OK();
}

void RotationalMotionModelEstimation::transform(const ImageSpace::MotionVectorField& field,
                                                const Core::InputDefinition& im, int time,
                                                SphericalSpace::MotionVectorField& sphericalField) const {
  std::unique_ptr<Core::TransformStack::GeoTransform> transform(
      Core::TransformStack::GeoTransform::create(panorama, im));
  Core::TopLeftCoords2 center((float)im.getWidth() / 2.0f, (float)im.getHeight() / 2.0f);
  for (ImageSpace::MotionVectorField::const_iterator vec = field.begin(); vec != field.end(); ++vec) {
    const Core::SphericalCoords3 src = transform->mapInputToScaledCameraSphereInRigBase(
        im, Core::CenterCoords2(Core::TopLeftCoords2((float)vec->from.x, (float)vec->from.y), center), time);
    const Core::SphericalCoords3 dst = transform->mapInputToScaledCameraSphereInRigBase(
        im, Core::CenterCoords2(Core::TopLeftCoords2((float)vec->to.x, (float)vec->to.y), center), time);
    // our "spherical coordinates" expressed as imaginary-only quaternions
    Quaternion<double> sQuat(0, src.x, src.y, src.z);
    Quaternion<double> dQuat(0, dst.x, dst.y, dst.z);
    sphericalField.push_back(SphericalSpace::MotionVector(sQuat / sQuat.norm(), dQuat / dQuat.norm()));
  }
}

}  // namespace Motion
}  // namespace VideoStitch
