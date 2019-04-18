// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef ROTATION_ESTIMATION_HPP_
#define ROTATION_ESTIMATION_HPP_

#include "util/lmfit/lmmin.hpp"

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

namespace VideoStitch {
namespace Calibration {

typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> SpherePointMatch;
typedef std::vector<SpherePointMatch> MatchList;

class RotationEstimationProblem : public Util::SolverProblem {
 public:
  explicit RotationEstimationProblem(const MatchList& matchList) : matchList(matchList) {}

  virtual void eval(const double* params, int m_dat, double* fvec, const char* fFilter, int /*iterationNumber*/,
                    bool* /* requestBreak */) const {
    Eigen::Matrix3d R;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        R(i, j) = params[i * 3 + j];
      }
    }

    for (auto i = 0; i < m_dat; ++i) {
      if (!fFilter || fFilter[i]) {
        Eigen::Vector3d rotatedPoint = (Eigen::Vector3d)(R * matchList[i].first);
        double y = (rotatedPoint.cross(matchList[i].second)).squaredNorm();
        double x = rotatedPoint.dot(matchList[i].second);
        // angular distance on the sphere
        double error = atan2(y, x);
        fvec[i] = error;
      } else {
        fvec[i] = 0.0;
      }
    }
  }

  virtual int numParams() const { return 9; }

  virtual int getNumInputSamples() const { return (int)matchList.size(); }

  virtual int getNumValuesPerSample() const { return 1; }

  virtual int getNumAdditionalValues() const { return 0; }

  MatchList getMatchList() const { return matchList; }

 private:
  MatchList matchList;
};

class RotationEstimationSolver : public Util::Solver<RotationEstimationProblem> {
 public:
  RotationEstimationSolver(const RotationEstimationProblem& problem, const char* const sampleFilter,
                           bool /*debug = false*/, bool /*useFloatPrecision = false*/)
      : Solver(problem), sampleFilter(sampleFilter) {}

  virtual bool run(std::vector<double>& params) {
    MatchList matchList = problem.getMatchList();
    const auto numberMatches = matchList.size();

    auto numberSamples = 0;
    // First count the number of samples that will be used
    for (uint32_t i = 0; i < numberMatches; ++i) {
      if (!sampleFilter || sampleFilter[i]) {
        numberSamples++;
      }
    }

    if (numberSamples < 3) {
      return false;
    }

    /*
     * Estimate the optimal rotation using the selected samples
     *
     * Least-Squares Rigid Motion Using SVD, Olga Sorkine
     * http://igl.ethz.ch/projects/ARAP/svd_rot.pdf
     */

    /*Build the covariance matrix*/
    Eigen::Matrix3d M;
    M.fill(0);
    for (uint32_t i = 0; i < numberMatches; ++i) {
      if (!sampleFilter || sampleFilter[i]) {
        Eigen::Vector3d first = matchList[i].first;
        Eigen::Vector3d second = matchList[i].second;

        for (int y = 0; y < 3; y++) {
          for (int x = 0; x < 3; x++) {
            M(y, x) += first(y) * second(x);
          }
        }
      }
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    /*Make sure we create a matrix with unit determinant*/
    double determinant = (U * V.transpose()).determinant();
    Eigen::Matrix3d D;
    D.setIdentity();
    D(2, 2) = 1.0 / determinant;
    Eigen::Matrix3d R = U * D * V.transpose();
    rotationToParam(params, R);

    return true;
  }

 private:
  void rotationToParam(std::vector<double>& params, const Eigen::Matrix3d& R) const {
    for (auto i = 0; i < 3; ++i) {
      for (auto j = 0; j < 3; ++j) {
        params[i + 3 * j] = R(i, j);
      }
    }
  }

  const char* const sampleFilter;
};

}  // namespace Calibration
}  // namespace VideoStitch

#endif
