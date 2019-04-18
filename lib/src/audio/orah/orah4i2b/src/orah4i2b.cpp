// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "orah4i2b.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>

#if defined(DEBUG_MATLAB)
#include <limits>
#include <iostream>
#include <fstream>
typedef std::numeric_limits<double> dbl;
#endif

// Ooura FFT (external)
extern "C" {
extern void rdft(int n, int isgn, double* a, int* ip, double* w);
}

#if defined(DEBUG_MATLAB)
const std::string kMatlabDumpFile = "dumpFile.m";
#endif

namespace VideoStitch {
namespace Orah4i {

/* The STFT size must be specified here since the matrix coefficients *
 * in orah4i2b_mtx_44.hpp and orah4i2b_mtx_48.hpp are for a fixed     *
 * block size.                                                        */
#define STFT_SIZE 1024
static_assert((STFT_SIZE & (~STFT_SIZE + 1)) == STFT_SIZE, "[orah4i2b] STFT block size must be 2^n");
static const int kHalfBufferSize{STFT_SIZE * 2};
static const int kFullBufferSize{STFT_SIZE * 4};
static const int kSpecSize{(STFT_SIZE / 2) + 1};
static const int kSpecBufferSize{kSpecSize * 8};
static const double kScaleFactor{2.0 / (double)STFT_SIZE};

// Sample rate must be defined. Default to 44.1 kHz
#if !defined(SAMPLE_RATE)
#warning "[orah4i2b] Sample rate not specified. Using default (44.1kHz)"
#define SAMPLE_RATE 44
#else
#if (SAMPLE_RATE != 44)
#error "Unknown sample rate. Must be 44"
#endif
#endif

// Conversion algorithm parameters
#define ALPHA_CAL 0.116099773242630
#define ALPHA_CAL_INV (1.0 - 0.116099773242630)
#define J_BAND 20
#define J_DELTA 3
static const double xoverd[3] = {1.0, 0.5, 0.0};
static const double xoveru[3] = {0.0, 0.5, 1.0};
#define LF_SPEC_SIZE (J_BAND + J_DELTA)
#include "orah4i2b_mtx_44.hpp"

// Menomics
enum { POS_MIC_1 = 0, POS_MIC_2 = 1, POS_MIC_3 = 2, POS_MIC_4 = 3 };
enum { POS_AMB_W = 0, POS_AMB_X = 1, POS_AMB_Y = 2, POS_AMB_Z = 3 };
enum { DO_IFFT = -1, DO_FFT = 1 };

Orah4iToB::Orah4iToB()
#if defined(DEBUG_MATLAB)
    : ofs_(kMatlabDumpFile, std::ofstream::app),
      cntBlk_(0)
#endif
{

  // Fill v with biquad filters
  std::vector<Biquad> v;
#define ADD_EQ_COEFF(a__) \
  { v.push_back(Biquad(a__)); }
#include "orah4i2b_eq.impl"

  // We need a copy of v for each microphone
  bq_.resize(4, v);

  previousBuffer_ = nullptr;
  workBuffer_ = nullptr;
  windowCoeffs_ = nullptr;
  fftBuffer_ = nullptr;
  ipWorkArea_ = nullptr;
  wWorkArea_ = nullptr;

#if defined(DEBUG_MATLAB)
  ofs_ << "invs = [] ;\n";
  ofs_ << "invseq = [] ;\n";
#endif
}

#define LET_GO(X)       \
  if ((X) != nullptr) { \
    delete[](X);        \
    (X) = nullptr;      \
  }
Orah4iToB::~Orah4iToB() {
  LET_GO(windowCoeffs_);
  LET_GO(previousBuffer_);
  LET_GO(workBuffer_);
  LET_GO(fftBuffer_);
  LET_GO(ipWorkArea_);
  LET_GO(wWorkArea_);
  for (int i = 0; i < 4; i++) {
    LET_GO(specSm_[i]);
    LET_GO(specCardHfOut_[i]);
  }
}
#undef LET_GO

Orah4iToB* Orah4iToB::create() {
  Orah4iToB* p = new Orah4iToB();
  if (!p->OToBSetup_()) {
    delete p;
    return nullptr;
  }
  return p;
}

#define CHECK_ASSIGN(X, Y) \
  if (X == nullptr) {      \
    assert(0 && (Y));      \
    return false;          \
  }
bool Orah4iToB::OToBSetup_() {
  OToBMakeWindow_();
  previousBuffer_ = new double[kHalfBufferSize]{0};
  workBuffer_ = new double[kFullBufferSize]{0};
  fftBuffer_ = new double[kSpecBufferSize]{0};
  size_t bufLen = size_t(2.0 + std::ceil(std::sqrt((double)STFT_SIZE / 2.0)));
  ipWorkArea_ = new int[bufLen]{0};
  bufLen = size_t((STFT_SIZE / 2));
  wWorkArea_ = new double[bufLen]{0};
  for (int i = 0; i < 4; i++) {
    specSm_[i] = new double[LF_SPEC_SIZE]{0};
    specCardHfOut_[i] = new std::complex<double>[(size_t)(kSpecSize - J_BAND)];
  }
  return true;
}
#undef CHECK_ASSIGN

void Orah4iToB::OToBMakeWindow_(void) {
  int i;
  assert(STFT_SIZE > 0 && "[orah4i2b] stft_size must be set before calling make_window()");
  windowCoeffs_ = new double[STFT_SIZE]{0};
  for (i = 0; i < STFT_SIZE; i++) {
    double val = (M_PI * (double)i) / (double)STFT_SIZE;
    windowCoeffs_[i] = sin(val);
  }
}

void Orah4iToB::OToBApplyWindow_() {
  assert(windowCoeffs_ != nullptr && "[orah4i2b] Window coefficients may have not been set");
  for (int i = 0, j = 0; i < STFT_SIZE; i++, j += 4) {
    workBuffer_[j + 0] *= windowCoeffs_[i];
    workBuffer_[j + 1] *= windowCoeffs_[i];
    workBuffer_[j + 2] *= windowCoeffs_[i];
    workBuffer_[j + 3] *= windowCoeffs_[i];
  }
}

#define STRIPE(X, Y) fftBuffer_[i + ((Y) * (STFT_SIZE + 2))] = workBuffer_[j + (X)]
#define FFT(X) rdft(STFT_SIZE, DO_FFT, &fftBuffer_[(X) * (STFT_SIZE + 2)], ipWorkArea_, wWorkArea_)
#define MV_NYQ(X)                                                                            \
  {                                                                                          \
    fftBuffer_[(((X) + 1) * (STFT_SIZE + 2)) - 2] = fftBuffer_[((X) * (STFT_SIZE + 2)) + 1]; \
    fftBuffer_[((X) * (STFT_SIZE + 2)) + 1] = 0;                                             \
  }
void Orah4iToB::OToBApplyFft_(void) {
  int i, j;

  for (i = 0, j = 0; i < STFT_SIZE; i++, j += 4) {
    STRIPE(POS_MIC_1, 0);
    STRIPE(POS_MIC_2, 1);
    STRIPE(POS_MIC_3, 2);
    STRIPE(POS_MIC_4, 3);
  }

  FFT(0);
  FFT(1);
  FFT(2);
  FFT(3);

  /* Since Ooura FFT returns the real part of N/2+1 (Nyquist) in  *
   * the real part of a[0], we need to move it to a[N/2+1], for   *
   * further processing.                                          */
  MV_NYQ(0);
  MV_NYQ(1);
  MV_NYQ(2);
  MV_NYQ(3);

  /* Assign beginning of each channel to a complex double *
   * pointer for easier access during processing.         */
  for (i = 0; i < 4; i++) {
    specIn_[i] = (std::complex<double>*)&fftBuffer_[i * (STFT_SIZE + 2)];
  }
}
#undef MV_NYQ
#undef FFT
#undef STRIPE

#define MV_NYQ(X)                                                                            \
  {                                                                                          \
    fftBuffer_[((X) * (STFT_SIZE + 2)) + 1] = fftBuffer_[(((X) + 1) * (STFT_SIZE + 2)) - 2]; \
    fftBuffer_[(((X) + 1) * (STFT_SIZE + 2)) - 2] = 0;                                       \
  }
#define IFFT(X) rdft(STFT_SIZE, DO_IFFT, &fftBuffer_[(X) * (STFT_SIZE + 2)], ipWorkArea_, wWorkArea_)
#define UNSTRIPE(X) workBuffer_[j + (X)] = fftBuffer_[i + ((X) * (STFT_SIZE + 2))] * kScaleFactor
void Orah4iToB::OToBApplyIfft_(void) {
  /* Since Ooura expects a[N/2+1] to be in the imaginary part of  *
   * a[0], we need to move it there before IFFT.                  */
  MV_NYQ(POS_AMB_W);
  MV_NYQ(POS_AMB_X);
  MV_NYQ(POS_AMB_Y);
  MV_NYQ(POS_AMB_Z);
  IFFT(POS_AMB_W);
  IFFT(POS_AMB_X);
  IFFT(POS_AMB_Y);
  IFFT(POS_AMB_Z);

  /* As per Ooura documentation, we need  *
   * to scale IFFT output:                *
   * Inverse of                           *
   *      rdft(n, 1, a, ip, w);           *
   *  is                                  *
   *      rdft(n, -1, a, ip, w);          *
   *      for (j = 0; j <= n - 1; j++) {  *
   *          a[j] *= 2.0 / n;            *
   *      }                               */
  for (int i = 0, j = 0; i < STFT_SIZE; i++, j += 4) {
    UNSTRIPE(POS_AMB_W) * M_SQRT2;
    UNSTRIPE(POS_AMB_X);
    UNSTRIPE(POS_AMB_Y);
    UNSTRIPE(POS_AMB_Z);
  }
}

#undef UNSTRIPE
#undef IFFT
#undef MV_NYQ

void Orah4iToB::OToBMicrophoneCorrection_(double* in) {
  // Interleaved samples
  for (int i = 0; i < get4iToBBlockSize(); ++i) {
    for (size_t cell = 0; cell < bq_[0].size(); ++cell) {
      // Equalize each microphone capsule
      bq_[0][cell].step(in);
      bq_[1][cell].step(in + 1);
      bq_[2][cell].step(in + 2);
      bq_[3][cell].step(in + 3);
    }
    in += 4;
  }
}

void Orah4iToB::OToBConvertToBFormat_(void) {
#if defined(DEBUG_MATLAB)
  dumpComplexPlanar_(specIn_, "spec_in_vs", get4iToBBlockSize());
#endif

  /* LF power spectrum estimation
   *
   *  MATLAB:
   *  spec_cal = spec_in;
   *  for n = 1:nmic,
   *      spec_sm(n,idx_LF) = alpha_cal.*abs(spec_in(n,idx_LF)).^2 + (1-alpha_cal).* spec_sm(n,idx_LF);
   *  end
   */

#define SPEC_SM(X)                 \
  val = std::abs(specIn_[(X)][i]); \
  val *= val;                      \
  specSm_[(X)][i] = (ALPHA_CAL * val) + (ALPHA_CAL_INV * specSm_[(X)][i])
  for (int i = 0; i < LF_SPEC_SIZE; i++) {
    double val;
    SPEC_SM(0);
    SPEC_SM(1);
    SPEC_SM(2);
    SPEC_SM(3);
  }
#undef SPEC_SM

#if defined(DEBUG_MATLAB)
  dumpPlanar_(specSm_, "spec_sm_vs", LF_SPEC_SIZE);
#endif

  /* LF power matching
   *
   *  MATLAB:
   *  spec_mean = mean(spec_sm(:,idx_LF), 1);
   *  for n = 1:nmic,
   *      H = sqrt((spec_mean+1e-10) ./ (spec_sm(n,idx_LF)+1e-10));
   *      spec_cal(n,idx_LF) = spec_in(n,idx_LF).*H;
   *  end
   */
#define SM(X) (specSm_[(X)][i])
  double specMean[LF_SPEC_SIZE];
  for (int i = 0; i < LF_SPEC_SIZE; i++) {
    specMean[i] = ((SM(0) + SM(1) + SM(2) + SM(3)) / 4.0) + 1e-10;
  }
#undef SM

#if defined(DEBUG_MATLAB)
  dump_(specMean, "spec_mean_vs", LF_SPEC_SIZE, 1);
#endif

  std::complex<double> specCal[4][LF_SPEC_SIZE];
  for (int i = 0; i < LF_SPEC_SIZE; i++) {
    specCal[0][i] = specIn_[0][i] * std::sqrt(specMean[i] / (specSm_[0][i] + 1e-10));
    specCal[1][i] = specIn_[1][i] * std::sqrt(specMean[i] / (specSm_[1][i] + 1e-10));
    specCal[2][i] = specIn_[2][i] * std::sqrt(specMean[i] / (specSm_[2][i] + 1e-10));
    specCal[3][i] = specIn_[3][i] * std::sqrt(specMean[i] / (specSm_[3][i] + 1e-10));
  }

#if defined(DEBUG_MATLAB)
  std::complex<double>* specCalDebug[4];
  for (int c = 0; c < 4; c++) {
    specCalDebug[c] = new std::complex<double>[LF_SPEC_SIZE];
    for (int i = 0; i < LF_SPEC_SIZE; i++) {
      specCalDebug[c][i] = specCal[c][i];
    }
  }
  dumpComplexPlanar_(specCalDebug, "spec_cal_vs", LF_SPEC_SIZE);
  for (int c = 0; c < 4; c++) {
    delete[] specCalDebug[c];
  }
#endif
  /* LF virtual cardioid
   *
   *  MATLAB:
   *  for n = 1:npair,
   *      spec_card_LF(n,idx_LF) = cmp_card(n,idx_LF) .* ...
   *                               ( pm_card(n,idx_LF) .* spec_cal(mic_idx(n,1),idx_LF) - ...
   *                                 spec_cal(mic_idx(n,2),idx_LF) );
   *  end
   */
  std::complex<double> specCardLf[4][LF_SPEC_SIZE];
  for (int i = 0; i < LF_SPEC_SIZE; i++) {
    specCardLf[0][i] = cmp_card[i] * (pm_card[i] * specCal[0][i] - specCal[2][i]);
    specCardLf[1][i] = cmp_card[i] * (pm_card[i] * specCal[2][i] - specCal[1][i]);
    specCardLf[2][i] = cmp_card[i] * (pm_card[i] * specCal[1][i] - specCal[3][i]);
    specCardLf[3][i] = cmp_card[i] * (pm_card[i] * specCal[3][i] - specCal[0][i]);
  }

#if defined(DEBUG_MATLAB)
  std::complex<double>* specCardLfDebug[4];
  for (int c = 0; c < 4; c++) {
    specCardLfDebug[c] = new std::complex<double>[LF_SPEC_SIZE];
    for (int i = 0; i < LF_SPEC_SIZE; i++) {
      specCardLfDebug[c][i] = specCardLf[c][i];
    }
  }
  dumpComplexPlanar_(specCardLfDebug, "spec_card_lf_vs", LF_SPEC_SIZE);
  for (int c = 0; c < 4; c++) {
    delete[] specCardLfDebug[c];
  }
#endif

  /* Apply LF transform matrix
   *
   *  MATLAB:
   *  spec_card_LF(:,idx_LF) = MA2B_LF * spec_card_LF(:,idx_LF);
   */
#define M_VAL(X, Y) (ma2b_LF[(X)][(Y)] * specCardLf[(X)][i])
  std::complex<double> specCardLfOut[4][LF_SPEC_SIZE];
  for (int i = 0; i < LF_SPEC_SIZE; i++) {
    specCardLfOut[0][i] = M_VAL(0, 0) + M_VAL(1, 0) + M_VAL(2, 0) + M_VAL(3, 0);
    specCardLfOut[1][i] = M_VAL(0, 1) + M_VAL(1, 1) + M_VAL(2, 1) + M_VAL(3, 1);
    specCardLfOut[2][i] = M_VAL(0, 2) + M_VAL(1, 2) + M_VAL(2, 2) + M_VAL(3, 2);
    specCardLfOut[3][i] = M_VAL(0, 3) + M_VAL(1, 3) + M_VAL(2, 3) + M_VAL(3, 3);
  }
#undef M_VAL

#if defined(DEBUG_MATLAB)
  std::complex<double>* specCardLfOutDebug[4];
  for (int c = 0; c < 4; c++) {
    specCardLfOutDebug[c] = new std::complex<double>[LF_SPEC_SIZE];
    for (int i = 0; i < LF_SPEC_SIZE; i++) {
      specCardLfOutDebug[c][i] = specCardLfOut[c][i];
    }
  }
  dumpComplexPlanar_(specCardLfOutDebug, "lf_out_vs", LF_SPEC_SIZE, true);
  for (int c = 0; c < 4; c++) {
    delete[] specCardLfOutDebug[c];
  }
#endif

  /* Apply HF transform matrix
   *
   *  MATLAB:
   *  spec_card_HF(:,idx_HF) = MA2B_HF * spec_card_HF(:,idx_HF);
   */
#define M_VAL(X, Y, Z) (ma2b_HF[(X)][(Y)] * specIn_[Z][i])
  for (int i = J_BAND, j = 0; i < kSpecSize; i++, j++) {
    specCardHfOut_[0][j] = M_VAL(0, 0, 0) + M_VAL(1, 0, 2) + M_VAL(2, 0, 1) + M_VAL(3, 0, 3);
    specCardHfOut_[1][j] = M_VAL(0, 1, 0) + M_VAL(1, 1, 2) + M_VAL(2, 1, 1) + M_VAL(3, 1, 3);
    specCardHfOut_[2][j] = M_VAL(0, 2, 0) + M_VAL(1, 2, 2) + M_VAL(2, 2, 1) + M_VAL(3, 2, 3);
    specCardHfOut_[3][j] = M_VAL(0, 3, 0) + M_VAL(1, 3, 2) + M_VAL(2, 3, 1) + M_VAL(3, 3, 3);
  }
#undef M_VAL

#if defined(DEBUG_MATLAB)
  dumpComplexPlanar_(specCardHfOut_, "hf_out_vs", kSpecSize - J_BAND, true);
#endif
  /* Cross-fade low and high
   *
   *  MATLAB:
   *  for n = 1:npair,
   *      spec_B(n,1:jband-1) = spec_card_LF(n,1:jband-1);
   *      spec_B(n,jband:jband+jdelta-1) = Xover_dec.*spec_card_LF(n,jband:jband+jdelta-1) + ...
   *                                       Xover_inc.*spec_card_HF(n,jband:jband+jdelta-1);
   *      spec_B(n,jband+jdelta:specsize) = spec_card_HF(n,jband+jdelta:specsize);
   *  end
   */
  for (int i = 0; i < J_BAND; i++) {
    specIn_[0][i] = specCardLfOut[0][i];
    specIn_[1][i] = specCardLfOut[1][i];
    specIn_[2][i] = specCardLfOut[2][i];
    specIn_[3][i] = specCardLfOut[3][i];
  }
#define SPEC_XO(X) specIn_[(X)][i] = (xoverd[j] * specCardLfOut[(X)][i]) + (xoveru[j] * specCardHfOut_[(X)][j])
  for (int i = J_BAND, j = 0; i < LF_SPEC_SIZE; i++, j++) {
    SPEC_XO(0);
    SPEC_XO(1);
    SPEC_XO(2);
    SPEC_XO(3);
  }
#undef SPEC_XO
  for (int i = LF_SPEC_SIZE, j = LF_SPEC_SIZE - J_BAND; i < kSpecSize; i++, j++) {
    specIn_[0][i] = specCardHfOut_[0][j];
    specIn_[1][i] = specCardHfOut_[1][j];
    specIn_[2][i] = specCardHfOut_[2][j];
    specIn_[3][i] = specCardHfOut_[3][j];
  }

#if defined(DEBUG_MATLAB)
  dumpComplexPlanar_(specIn_, "spec_out_vs", kSpecSize, true);
#endif
}

bool Orah4iToB::process(double* in, double* out) {
  if (in == nullptr || out == nullptr) {
    assert(false && "[orah4i2b] Valid input and output buffers required");
    return false;
  }

#if defined(DEBUG_MATLAB)
  dump_(in, "invs", get4iToBBlockSize());
#endif

  // TODO: add Microphone correction after debugging
  OToBMicrophoneCorrection_(in);  // Directly in-place on input block

#if defined(DEBUG_MATLAB)
  dump_(in, "invseq", get4iToBBlockSize());
#endif

  std::memcpy(&workBuffer_[kHalfBufferSize], in, (size_t)kHalfBufferSize * sizeof(double));
  OToBApplyWindow_();
#if defined(DEBUG_MATLAB)
  dump_(workBuffer_, "in_win_vs", get4iToBBlockSize());
#endif
  OToBApplyFft_();
  OToBConvertToBFormat_();
  OToBApplyIfft_();
  OToBApplyWindow_();
  for (int i = 0; i < kHalfBufferSize; i++) {
    out[i] = workBuffer_[i] + previousBuffer_[i];
  }
#if defined(DEBUG_MATLAB)
  dump_(out, "outvs", get4iToBBlockSize());
#endif
  std::memcpy(previousBuffer_, &workBuffer_[kHalfBufferSize], (size_t)kHalfBufferSize * sizeof(double));
  std::memcpy(workBuffer_, in, (size_t)kHalfBufferSize * sizeof(double));

#if defined(DEBUG_MATLAB)
  cntBlk_++;
#endif

  return true;
}

int get4iToBBlockSize() { return STFT_SIZE / 2; }

#if defined(DEBUG_MATLAB)
/// dump_ data of 4 channels interleaved into a matlab mtx
/// works also to dump a single channel
void Orah4iToB::dump_(double* data, const std::string& dataName, int nbElements, int nChannels) {
  ofs_.precision(dbl::max_digits10);
  for (int i = 0; i < nbElements; i++) {
    for (int c = 0; c < nChannels; c++) {
      ofs_ << dataName << "(" << c + 1 << ", " << cntBlk_ * get4iToBBlockSize() + i + 1
           << ") = " << data[nChannels * i + c] << ";\n";
    }
  }
}

void Orah4iToB::dumpComplexPlanar_(std::complex<double>* data[4], const std::string& dataName, int nbElements,
                                   bool isSpec) {
  ofs_.precision(dbl::max_digits10);
  for (int i = 0; i < nbElements; i++) {
    for (int c = 0; c < 4; c++) {
      int idxMat;
      if (isSpec) {
        idxMat = cntBlk_ * kSpecSize + i + 1;
      } else {
        idxMat = cntBlk_ * get4iToBBlockSize() + i + 1;
      }
      ofs_ << dataName << "(" << c + 1 << ", " << idxMat << ") = " << std::real(data[c][i]) << "+"
           << std::imag(data[c][i]) << "j;\n";
    }
  }
}

void Orah4iToB::dumpPlanar_(double* data[4], const std::string& dataName, int nbElements) {
  ofs_.precision(dbl::max_digits10);
  for (int i = 0; i < nbElements; i++) {
    for (int c = 0; c < 4; c++) {
      ofs_ << dataName << "(" << c + 1 << ", " << cntBlk_ * get4iToBBlockSize() + i + 1 << ") = " << data[c][i]
           << ";\n";
    }
  }
}
#endif

}  // namespace Orah4i
}  // namespace VideoStitch
