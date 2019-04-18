// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// A utility for calculating the time offset between two audio streams

#include "delayCalculator.hpp"

#include "util/fft.h"

#include "libvideostitch/logging.hpp"

#include <complex>
#include <algorithm>

namespace VideoStitch {
namespace Audio {

static const double kSearchTime = 10.0;

DelayCalculator::DelayCalculator() : result_(0) { numSamples_ = (size_t)(kSearchTime * getDefaultSamplingRate()); }

DelayCalculator::~DelayCalculator() {}

/// \fn size_t DelayCalculator::getOffset() const
/// \brief Return the result of the offset calculation
///
/// It is meaningless to call this function until `DelayCalculator::fill()`
/// has returned `true`.
size_t DelayCalculator::getOffset() const { return result_; }

namespace {
inline void extractOneChannel(const AudioBlock& in, std::vector<double>* out) {
  out->assign(in[0].begin(), in[0].end());
}
}  // namespace

/// \fn bool DelayCalculator::fill(const AudioBlock& early, const AudioBlock& late)
/// \brief Add samples to the search buffers until `kSearchTime` is achieved, then
///        calculate the offset.
/// \param early The real-time audio
/// \param late The delayed audio
/// \return `false` if `kSearchTime` of audio is not in the buffer yet. `true` when
///         the buffers are full and the offset has been calculated.
bool DelayCalculator::fill(const AudioBlock& early, const AudioBlock& late) {
  std::vector<double> monoEarly;
  std::vector<double> monoLate;
  extractOneChannel(early, &monoEarly);
  extractOneChannel(late, &monoLate);
  early_.insert(early_.end(), monoEarly.begin(), monoEarly.end());
  late_.insert(late_.end(), monoLate.begin(), monoLate.end());
  if (numSamples_ > 0) {
    numSamples_ -= monoEarly.size();
  }
  if (numSamples_ == 0) {
    calculate_();
    return true;
  }
  return false;
}

/// \fn void DelayCalculator::reset()
/// \brief Empty buffers and reset sample count to initial value
void DelayCalculator::reset() {
  early_.resize(0);
  late_.resize(0);
  numSamples_ = (size_t)(kSearchTime * getDefaultSamplingRate());
  result_ = 0;
}

void DelayCalculator::calculate_() {
  // Buffer sizes -----------
  size_t len = early_.size();
  if (len != late_.size()) {
    Logger::get(Logger::Warning) << "Different buffer sizes for automatic delay compensation" << std::endl;
    len = std::min(len, late_.size());
  }
  size_t fftSize = 2;
  while ((len * 2) > fftSize) {
    fftSize *= 2;
  }
  const size_t realBufferSize = len * sizeof(double);
  const size_t cpxBufferSize = 2 * fftSize * sizeof(double);  // 2 x because re and im parts

  // Buffers ----------------
  std::complex<double>* cpxStore = new std::complex<double>[fftSize];
  std::complex<double>* cpxData = new std::complex<double>[fftSize];
  double* store = reinterpret_cast<double*>(cpxStore);  // C++ standard says we can do this
  double* data = reinterpret_cast<double*>(cpxData);

  memset(data, 0, cpxBufferSize);
  memcpy(data, early_.data(), realBufferSize);
  rdft((int)fftSize, 1, data);
  memcpy(store, data, cpxBufferSize);

  memset(data, 0, cpxBufferSize);
  memcpy(data, late_.data(), realBufferSize);
  rdft((int)fftSize, 1, data);

  // Don't use DC or Nyquist values (Ooura stores them in the first slot)
  cpxData[0] = 0;
  cpxStore[0] = 0;

  for (size_t k = 1; k < (fftSize / 2); k++) {
    cpxData[k] = cpxStore[k] * conj(cpxData[k]);
  }
  rdft((int)fftSize, -1, data);

  double max = 0;
  size_t idx = 0;
  for (size_t k = 1; k < fftSize; k++) {
    if (data[k] > max) {
      max = data[k];
      idx = k;
    }
  }
  if (idx > (fftSize / 2)) {
    result_ = idx - fftSize;
  } else {
    result_ = idx;
  }

  Logger::get(Logger::Info) << "[Auto Delay Compensation] Found an offset of " << idx << std::endl;

  delete[] cpxStore;
  delete[] cpxData;
}

}  // end namespace Audio
}  // end namespace VideoStitch
