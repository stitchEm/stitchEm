// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <complex>
#include <vector>
#include <array>

/* Define DEBUG_MATLAB in order to save the results of each    *
 * step of the B-format conversion algorithm in a file.        *
 * This is mostly intended to verify the results against the   *
 * MATLAB version.                                             */
#if defined(DEBUG_MATLAB)
#include <fstream>
#endif

namespace VideoStitch {
namespace Orah4i {

int get4iToBBlockSize();

class Orah4iToB {
 public:
  ~Orah4iToB();
  static Orah4iToB* create();
  bool process(double* in, double* out);

 private:
  struct BqCoeff {
    double a[3];
    double b[2];
  };
  class Biquad {
   public:
    explicit Biquad(BqCoeff c) : c_(c), d_({{0., 0.}}) {}
    ~Biquad() {}
    void step(double* sample) {
      double s = *sample;
      *sample = (c_.a[0] * s) + d_[0];
      d_[0] = (c_.a[1] * s) - (c_.b[0] * (*sample)) + d_[1];
      d_[1] = (c_.a[2] * s) - (c_.b[1] * (*sample));
    }

   private:
    BqCoeff c_;
    std::array<double, 2> d_;
  };

  double* previousBuffer_;
  double* workBuffer_;
  double* windowCoeffs_;
  double* specSm_[4];
  std::complex<double>* specCardHfOut_[4];
  double* fftBuffer_;
  std::complex<double>* specIn_[4];
  int* ipWorkArea_;
  double* wWorkArea_;

  std::vector<std::vector<Biquad>> bq_;

  Orah4iToB();

  bool OToBSetup_();
  void OToBMakeWindow_();
  void OToBApplyWindow_();
  void OToBApplyFft_();
  void OToBMicrophoneCorrection_(double* in);
  void OToBConvertToBFormat_();
  void OToBApplyIfft_();

#if defined(DEBUG_MATLAB)
  std::ofstream ofs_;
  int64_t cntBlk_;
  void dump_(double* data, const std::string& dataName, int nbElements, int nChannels = 4);
  void dumpComplexPlanar_(std::complex<double>* data[4], const std::string& dataName, int nbElements,
                          bool isSpec = false);
  void dumpPlanar_(double* data[4], const std::string& dataName, int nbElements);
#endif
};

}  // namespace Orah4i
}  // namespace VideoStitch
