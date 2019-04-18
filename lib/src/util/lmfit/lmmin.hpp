// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LMMIN_HPP_
#define LMMIN_HPP_

#include "lmmin.h"
#include <cassert>
#include <cmath>
#include <vector>

namespace VideoStitch {
namespace Util {

/**
 * A problem specification for BaseSolver (see below).
 * NOT thread-safe.
 */
class SolverProblem {
 public:
  virtual ~SolverProblem() {}

  /**
   * To be implemented by subclasses.
   * @param params parameter values.
   * @param m_dat Dimension of fvec == number of samples.
   * @param fvec Ouput vector. Shall start with the getNumInputSamples() * getValuesPerSample() samples outputs.
   * @param fFilter If set, this is a sample filter. Only the samples where fFilter is non-zero should be used.
   * @param requestBreak set true to request a break
   */
  virtual void eval(const double* params, int m_dat, double* fvec, const char* fFilter, int iterationNumber,
                    bool* requestBreak) const = 0;

  virtual int numParams() const = 0;

  /**
   * To be implemented by subclasses. Returns the number of input samples.
   */
  virtual int getNumInputSamples() const = 0;

  /**
   * To be implemented by subclasses. Returns the number of output values per input sample.
   */
  virtual int getNumValuesPerSample() const = 0;

  /**
   * To be implemented by subclasses. Returns the number of additional values ().
   */
  virtual int getNumAdditionalValues() const = 0;

  /**
   * Returns the number of output values (size of fvec in eval()).
   */
  int getNumOutputValues() const { return getNumInputSamples() * getNumValuesPerSample() + getNumAdditionalValues(); }

 protected:
  SolverProblem() {}

 private:
};

/**
 * Abstract solver.
 */
template <typename Problem>
class Solver {
 public:
  typedef Problem Problem_t;

  explicit Solver(const Problem& problem) : problem(problem) {}
  virtual ~Solver() {}

  /**
   * Solve the problem. On input, @a params shall contain the initial guess of the solution. On output,it will contains
   * the solution.
   * @param params parameter values.
   * @return false on failure.
   */
  virtual bool run(std::vector<double>& params) = 0;

  const Problem& getProblem() const { return problem; }

 protected:
  const Problem& problem;
};

/**
 * A solver.
 * A very simple, lightweight layer on top of lmmmin to use it easily in c++ code.
 */
template <typename Problem>
class LmminSolver : public Solver<Problem> {
 public:
  /**
   * A solver for the given problem.
   * @param problem The problem to solve. Must outlive us.
   * @param fFilter If set, this is a sample filter. Only the samples where fFilter is non-zero should be used.
   * @param debug Whether to print debug info.
   * @param useFloatPrecision Whether to use float precision (default: double precision)
   */
  LmminSolver(const Problem& problem, const char* fFilter, bool debug = false, bool useFloatPrecision = false)
      : Solver<Problem>(problem), status({0.0, 0, 0}), sampleFilter(fFilter) {
    if (useFloatPrecision) {
      control = lm_control_float;
    } else {
      control = lm_control_double;
    }
    control.printflags = debug ? 7 : 0;  // monitor status (+1) and parameters (+2)
  }

  bool run(std::vector<double>& params) {
    lmmin((int)params.size(), params.data(), this->problem.getNumOutputValues(), (const void*)this, &evalStatic,
          &control, &status, lm_printout_std);
    // std::cout << lm_infmsg[status.info] << std::endl;
    return status.info < 4;
  }

  bool run(std::vector<double>& params, const int maxNumIter) {
    control.maxcall = maxNumIter;
    return run(params);
  }

  lm_control_struct& getControl() { return control; }

 private:
  static void evalStatic(const double* params, int m_dat, const void* data, double* fvec, int* info) {
    const LmminSolver* that = reinterpret_cast<const LmminSolver*>(data);
    // If we have a NaN here, something is wrong.
    for (int i = 0; i < that->problem.numParams(); i++) {
      if (VS_ISNAN(params[i])) {
        assert(false);
        *info = -1;  // request break
        return;
      }
    }
    bool requestBreak = false;
    that->problem.eval(params, m_dat, fvec, that->sampleFilter, that->status.nfev, &requestBreak);
    if (requestBreak) {
      *info = -1;
    }
  }

  // Algorithm parameters.
  lm_control_struct control;

  // Current status
  lm_status_struct status;

  // Temporary.
  const char* const sampleFilter;
};

}  // namespace Util
}  // namespace VideoStitch

#endif
