// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef RANSACPROBLEM_HPP_
#define RANSACPROBLEM_HPP_

#include <algorithm>
#include <iostream>
#include "lmfit/lmmin.hpp"
#include <random>

// #define RANSAC_VERBOSE

namespace VideoStitch {
namespace Util {

/**
 * A RANSAC layer over any Solver
 * NOT thread-safe.
 */
template <typename Solver_t>
class RansacSolver : public Solver<typename Solver_t::Problem_t> {
 public:
  /**
   * Creates a RANSAC model.
   * @param minSamplesForFit Minimum number of samples required to fit a model.
   */
  RansacSolver(const typename Solver_t::Problem_t& problem, int minSamplesForFit, int numIters, int minConsensusSamples,
               std::default_random_engine* gen, bool debug = false, bool useFloatPrecision = false)
      : Solver<typename Solver_t::Problem_t>(problem),
        minSamplesForFit(minSamplesForFit),
        numIters(numIters),
        minConsensusSamples(minConsensusSamples),
        bitSet(problem.getNumInputSamples()),
        solver(problem, bitSet.data(), debug, useFloatPrecision),
        gen(gen) {}

  virtual ~RansacSolver() {}

  lm_control_struct& getControl() { return solver.getControl(); }

  bool run(std::vector<double>& params) {
    std::vector<char> inlierIndices;
    std::vector<double> outputResiduals;
    return run(params, inlierIndices, outputResiduals);
  }

  bool run(std::vector<double>& params, std::vector<char>& inlierIndices, std::vector<double>& outputResiduals) {
    if ((int)bitSet.size() < minSamplesForFit) {
      return false;
    }
    int bestNumConsensual = 0;
    std::vector<double> curModel(params.size());
    // Inliers and consensus sets. 0 means not selected.
    std::vector<double> residuals(solver.getProblem().getNumOutputValues());
    for (int iter = 0; iter < numIters; ++iter) {
#ifdef RANSAC_VERBOSE
      std::cout << "iter " << iter << ":" << std::endl;
#endif
      curModel = params;
      // Select random subset of size minSamplesForFit. bitSet = maybeInlinersSet.
      populateRandom(bitSet, minSamplesForFit);
      // Fit model on subset.
      if (!solver.run(curModel)) {
        continue;
      }
#ifdef RANSAC_VERBOSE
      for (size_t k = 0; k < params.size(); ++k) {
        std::cout << "  " << curModel[k] << std::endl;
      }
#endif
      // And get the residuals.
      bool requestBreakNotPossible = false;
      solver.getProblem().eval(curModel.data(), (int)residuals.size(), residuals.data(), NULL, 0,
                               &requestBreakNotPossible);
      // Get the residuals. bitSet = consensusSet.

      int numConsensual = 0;
      for (int i = 0; i < solver.getProblem().getNumInputSamples(); ++i) {
        if (isConsensualSample(residuals.data() + i * solver.getProblem().getNumValuesPerSample())) {
          ++numConsensual;
          bitSet[i] = 1;
        } else {
          bitSet[i] = 0;
        }
      }
#ifdef RANSAC_VERBOSE
      std::cout << "numConsensual: " << numConsensual << "/" << solver.getProblem().getNumInputSamples() << " "
                << minConsensusSamples << " " << bestNumConsensual << std::endl;
#endif

      // Check if found rotation matrix fits the presets bounds
      if (!validate(curModel.data())) {
#ifdef RANSAC_VERBOSE
        std::cout << "estimated rotation out of the presets" << std::endl;
#endif
        continue;
      }

      if (numConsensual > minConsensusSamples && numConsensual > bestNumConsensual) {
#ifdef RANSAC_VERBOSE
        std::cout << "new best : " << numConsensual << std::endl;
        std::cout << " model:" << std::endl;
        for (size_t k = 0; k < params.size(); ++k) {
          std::cout << "  " << curModel[k] << std::endl;
        }
#endif
        if (!solver.run(curModel)) {
          continue;
        }
#ifdef RANSAC_VERBOSE
        std::cout << " model2:" << std::endl;
        for (size_t k = 0; k < params.size(); ++k) {
          std::cout << "  " << curModel[k] << std::endl;
        }
#endif
        params = curModel;
        bestNumConsensual = numConsensual;
        inlierIndices = std::vector<char>(bitSet);
        outputResiduals = std::vector<double>(residuals);

        if (numConsensual ==
            solver.getProblem()
                .getNumInputSamples()) {  // if all samples are inliers then stop looking for a better consensus
          return true;
        }
      }
    }
    if (bestNumConsensual == 0) {
      return false;
    }
    return true;
  }

 private:
  virtual bool validate(double* /*values*/) const { return true; }

  /**
   * Implements the criterion for a consensual samples.
   * @param values The getValuesPerSample() values for this sample.
   */
  virtual bool isConsensualSample(double* values) const = 0;

  /**
   * Implements the random selection. The default is purely random selection, but some algorithms may have further
   * constraints.
   * @param vector to populate.
   * @param numBitsSets Minimum number of samples to select.
   */
  virtual void populateRandom(std::vector<char>& v, size_t numBitsSets) const {
    for (size_t i = 0; i < numBitsSets; ++i) {
      v[i] = 1;
    }
    for (size_t i = numBitsSets; i < v.size(); ++i) {
      v[i] = 0;
    }
    std::shuffle(v.begin(), v.end(), *gen);
  }

  const int minSamplesForFit;
  const int numIters;
  const int minConsensusSamples;
  std::vector<char> bitSet;
  Solver_t solver;
  std::default_random_engine* gen;
};
}  // namespace Util
}  // namespace VideoStitch

#endif
