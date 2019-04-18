// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "affineMotion.hpp"

#include "libvideostitch/inputDef.hpp"

#include <opencv2/core/core.hpp>

#include <algorithm>
#include <random>

//#define RANSAC_VERBOSE

namespace VideoStitch {
namespace Motion {

namespace {
/**
 * A RANSAC affine model fitter.
 * We require the motion vectors to be relative to the center of the image.
 */
class RansacAffine {
 public:
  RansacAffine(const ImageSpace::MotionVectorField& field, double inlierThreshold, int minSamplesForFit, int numIters,
               int minConsensusSamples, std::default_random_engine& gen)
      : inlierThreshold(inlierThreshold),
        minSamplesForFit(minSamplesForFit),
        numIters(numIters),
        minConsensusSamples(minConsensusSamples),
        field(field),
        bitSet(field.size()),
        gen(gen) {}

  virtual ~RansacAffine() {}

  bool ransac(std::vector<double>& params) {
    int bestNumConsensual = 0;
    std::vector<double> curModel(params.size());
    // Inliners and consensus sets. 0 means not selected.
    std::vector<double> residuals(field.size());
    for (int iter = 0; iter < numIters; ++iter) {
#ifdef RANSAC_VERBOSE
      std::cout << "iter " << iter << ":" << std::endl;
#endif
      curModel = params;
      // Select random subset of size minSamplesForFit. bitSet = maybeInlinersSet.
      populateRandom(minSamplesForFit);
      // Fit model on subset.
      fitAffine(curModel);
#ifdef RANSAC_VERBOSE
      for (size_t k = 0; k < params.size(); ++k) {
        std::cout << "  " << curModel[k] << std::endl;
      }
#endif
      // Get the residuals. bitSet = consensusSet.
      int numConsensual = 0;
      for (size_t i = 0; i < field.size(); ++i) {
        if (bitSet[i] > 0 || isConsensualSample(curModel, field[i])) {
          ++numConsensual;
          bitSet[i] = 1;
        } else {
          bitSet[i] = 0;
        }
      }
#ifdef RANSAC_VERBOSE
      std::cout << "numConsensual: " << numConsensual << "/" << field.size() << " " << minConsensusSamples << " "
                << bestNumConsensual << std::endl;
#endif
      if (numConsensual > minConsensusSamples && numConsensual > bestNumConsensual) {
#ifdef RANSAC_VERBOSE
        std::cout << "new best : " << numConsensual << std::endl;
        std::cout << " model:" << std::endl;
        for (size_t k = 0; k < params.size(); ++k) {
          std::cout << "  " << curModel[k] << std::endl;
        }
#endif
        fitAffine(curModel);
#ifdef RANSAC_VERBOSE
        std::cout << " model2:" << std::endl;
        for (size_t k = 0; k < params.size(); ++k) {
          std::cout << "  " << curModel[k] << std::endl;
        }
#endif
        params = curModel;
        bestNumConsensual = numConsensual;
      }
    }
    if (bestNumConsensual == 0) {
      return false;
    }
    return true;
  }

  void fitAffine(std::vector<double>& model) const {
    int count = 0;
    for (size_t i = 0; i < field.size(); ++i) {
      if (bitSet[i] > 0) {
        count++;
      }
    }

    // linear least-squares : solve normal equation
    cv::Mat P(2 * count, 6, CV_64F);
    cv::Mat Q(2 * count, 1, CV_64F);
    int j = 0;
    for (size_t i = 0; i < field.size(); ++i) {
      if (bitSet[i] > 0) {
        ImageSpace::MotionVector mv = field[i];
        int indexRow = 2 * j;
        P.at<double>(indexRow, 0) = mv.from.x;
        P.at<double>(indexRow, 1) = mv.from.y;
        P.at<double>(indexRow, 2) = 1;
        P.at<double>(indexRow, 3) = 0;
        P.at<double>(indexRow, 4) = 0;
        P.at<double>(indexRow, 5) = 0;
        P.at<double>(indexRow + 1, 0) = 0;
        P.at<double>(indexRow + 1, 1) = 0;
        P.at<double>(indexRow + 1, 2) = 0;
        P.at<double>(indexRow + 1, 3) = mv.from.x;
        P.at<double>(indexRow + 1, 4) = mv.from.y;
        P.at<double>(indexRow + 1, 5) = 1;
        Q.at<double>(indexRow, 0) = mv.to.x;
        Q.at<double>(indexRow + 1, 0) = mv.to.y;
        j++;
      }
    }
    cv::Mat A = (P.t() * P).inv() * P.t() * Q;

    model[0] = A.at<double>(0, 0);
    model[1] = A.at<double>(1, 0);
    model[2] = A.at<double>(2, 0);
    model[3] = A.at<double>(3, 0);
    model[4] = A.at<double>(4, 0);
    model[5] = A.at<double>(5, 0);
  }

  bool isConsensualSample(std::vector<double>& model, ImageSpace::MotionVector mv) const {
    // Maximum re-projection error value to classify as inlier.
    const double reproj_x = (double)mv.to.x - (double)mv.from.x * model[0] - (double)mv.from.y * model[1] - model[2];
    const double reproj_y = (double)mv.to.y - (double)mv.from.x * model[3] - (double)mv.from.y * model[4] - model[5];
    return sqrt(reproj_x * reproj_x + reproj_y * reproj_y) < inlierThreshold;
  }

  void populateRandom(size_t numBitsSets) {
    for (size_t i = 0; i < numBitsSets; ++i) {
      bitSet[i] = 1;
    }
    for (size_t i = numBitsSets; i < bitSet.size(); ++i) {
      bitSet[i] = 0;
    }
    std::shuffle(bitSet.begin(), bitSet.end(), gen);
  }

 private:
  const double inlierThreshold;
  const int minSamplesForFit;
  const int numIters;
  const int minConsensusSamples;
  const ImageSpace::MotionVectorField& field;
  std::vector<char> bitSet;
  std::default_random_engine& gen;
};
}  // namespace

void AffineMotionModelEstimation::motionModel(const MotionVectorFieldTimeSeries& timeSeries, MotionModel& model,
                                              const Core::InputDefinition& im) {
  for (auto mvfield = timeSeries.begin(); mvfield != timeSeries.end(); ++mvfield) {
    Matrix33<double> a;
    motionModel(mvfield->second, a, im);
    model[mvfield->first] = std::make_pair(true, a);
  }
}

Status AffineMotionModelEstimation::motionModel(const ImageSpace::MotionVectorField& field, Matrix33<double>& affine,
                                                const Core::InputDefinition& im) {
  // use centered coordinates
  ImageSpace::MotionVectorField centeredField;
  for (size_t i = 0; i < field.size(); i++) {
    float2 first, second;
    first.x = field[i].from.x - (float)im.getWidth() / 2.0f;
    first.y = field[i].from.y - (float)im.getHeight() / 2.0f;
    second.x = field[i].to.x - (float)im.getWidth() / 2.0f;
    second.y = field[i].to.y - (float)im.getHeight() / 2.0f;
    centeredField.push_back(ImageSpace::MotionVector(first, second));
  }

  std::default_random_engine gen(42);
  // We require 40% of the motion vectors to be consensuals.
  RansacAffine solver(centeredField, 10.0, 3, 100, (int)((double)field.size() * 0.4), gen);

  std::vector<double> params{1, 0, 0, 0, 1, 0};
  if (field.size() < 3) {
    return {Origin::StabilizationAlgorithm, ErrType::AlgorithmFailure,
            "Could not estimate affine model, pushing last model"};
  }
  if (!solver.ransac(params)) {
    // ok, just fit anything...
    solver.populateRandom(centeredField.size());
    solver.fitAffine(params);
    affine = Matrix33<double>(params[0], params[1], params[2], params[3], params[4], params[5], 0, 0, 1);
    return {Origin::StabilizationAlgorithm, ErrType::AlgorithmFailure,
            "Could not estimate affine model, approximating frame"};
  }
  affine = Matrix33<double>(params[0], params[1], params[2], params[3], params[4], params[5], 0, 0, 1);
  return Status::OK();
}

}  // namespace Motion
}  // namespace VideoStitch
