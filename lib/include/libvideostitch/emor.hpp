// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef EMOR_HPP_
#define EMOR_HPP_

#include "config.hpp"

#include <array>

namespace VideoStitch {
namespace Core {

#if (_MSC_VER && _MSC_VER < 1900)
#define VS_CONSTEXPR
#else
#define VS_CONSTEXPR constexpr
#endif

/**
 * A base response curve.
 */
class VS_EXPORT ResponseCurve {
 public:
  /**
   * The size of the lookup table for one direction.
   */
  static VS_CONSTEXPR int lutSize() { return 1024; }

  /**
   * The total size of the lookup table (both directions).
   */
  static VS_CONSTEXPR int totalLutSize() {
    // additional value for fast interpolation of the last value
    return 2 * lutSize() + 1;
  }

  virtual ~ResponseCurve();

  /**
   * Returns the response curve.
   * The first 1024 values are the direct response, the next 1024 values are the inverse response.
   * Values are in [0,1].
   */
  const float* getResponseCurve() const;

  /**
   * Returns the inverse response curve.
   * Pointer to the 1024 values of the inverse response (second half of getResponseCurve).
   * Values are in [0,1].
   */
  const float* getInverseResponseCurve() const;

  /**
   * Takes the reciprocal of the curve.
   */
  void invert();

  /**
   * How much did makeMonotonous() have to correct on creation?
   */
  int getMonotonyError() const;

 protected:
  ResponseCurve();

  /**
   * Fill in the elements LutSize..(2 * LutSize - 1) with the inverse response.
   */
  void computeInverseResponse();

  /**
   * The response curve.
   */
  float* responseCurve;

 private:
  ResponseCurve(const ResponseCurve&);

  /**
   * Makes sure that the forward response is monotonous.
   */
  void makeMonotonous();

  /**
   * Before making the curve monotonous, calculate how far off it was
   */
  void calculateMonotonyError();

  int monotonyError;
};

class ValueResponseCurve : public ResponseCurve {
 public:
  explicit ValueResponseCurve(const std::array<uint16_t, 256>& values);
};

/**
 * A response curve that uses the first 6 eigenvectors of the EMoR model.
 */
class VS_EXPORT EmorResponseCurve : public ResponseCurve {
 public:
  /**
   * Builds an response curve with the specified EMoR coefficients (coordinates in the EMoR basis).
   * @param emor1: Coefficient for eigenvector 1.
   * @param emor2: Coefficient for eigenvector 2.
   * @param emor3: Coefficient for eigenvector 3.
   * @param emor4: Coefficient for eigenvector 4.
   * @param emor5: Coefficient for eigenvector 5.
   */
  EmorResponseCurve(double emor1, double emor2, double emor3, double emor4, double emor5);
};

/**
 * A response curve that uses the first 6 eigenvectors of the inverse EMoR model.
 */
class VS_EXPORT InvEmorResponseCurve : public ResponseCurve {
 public:
  /**
   * Builds an response curve with the specified EMoR coefficients (coordinates in the inverse EMoR basis).
   * @param emor1: Coefficient for eigenvector 1.
   * @param emor2: Coefficient for eigenvector 2.
   * @param emor3: Coefficient for eigenvector 3.
   * @param emor4: Coefficient for eigenvector 4.
   * @param emor5: Coefficient for eigenvector 5.
   */
  InvEmorResponseCurve(double emor1, double emor2, double emor3, double emor4, double emor5);
};
}  // namespace Core
}  // namespace VideoStitch

#undef VS_CONSTEXPR

#endif
