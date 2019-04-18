// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "histogramView.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>

namespace VideoStitch {
namespace Image {

CpuHistogramView::CpuHistogramView(const uint32_t* storage) : storage(storage), normFactor(-1), average(-1.0) {}

double CpuHistogramView::sqrDistanceL2(const CpuHistogramView& other) const {
  // Cache normalization factors.
  getNormFactor();
  other.getNormFactor();
  double res = 0.0;
  for (int i = 0; i < 256; ++i) {
    const double l = getNormalizedSampleInternal(i) - other.getNormalizedSampleInternal(i);
    res += l * l;
  }
  return res;
}

double CpuHistogramView::sqrDistanceChi2(const CpuHistogramView& other) const {
  // Cache normalization factors.
  getNormFactor();
  other.getNormFactor();
  double res = 0.0;
  for (int i = 0; i < 256; ++i) {
    if (count(i) > 0 || other.count(i) > 0) {
      const double p = getNormalizedSampleInternal(i);
      const double q = other.getNormalizedSampleInternal(i);
      res += ((p - q) * (p - q)) / (p + q);
    }
  }
  return res;
}

void boxConvolve(double* dst, const double* src, const int width) {
  const double div = 2 * width + 1;
  double accum = src[0] * width;
  for (int i = 0; i < width; ++i) {
    accum += src[i];
  }
  for (int i = 0; i < width; ++i) {
    accum += src[i + width];
    dst[i] = accum / div;
    accum -= src[0];
  }
  for (int i = width; i < 256 - width; ++i) {
    accum += src[i + width];
    dst[i] = accum / div;
    accum -= src[i - width];
  }
  for (int i = 256 - width; i < 256; ++i) {
    accum += src[255];
    dst[i] = accum / div;
    accum -= src[i - width];
  }
}

double CpuHistogramView::sqrDistanceQF(const CpuHistogramView& other, int width) const {
  // Cache normalization factors.
  getNormFactor();
  other.getNormFactor();
  // (P-Q)^t A (P-Q), where A is a positive definite matrix
  double pqm[256];
  for (int i = 0; i < 256; ++i) {
    pqm[i] = getNormalizedSampleInternal(i) - other.getNormalizedSampleInternal(i);
  }
  // Here, we make it a bit more efficient by making the additional assumption A == B^t B, so that:
  // (P-Q)^t A (P-Q) == (P-Q)^t B^t B (P-Q) = (B (P-Q))^t B (P-Q)
  // In addition, we assume that B has translation symmetry along the diagonal, so that B (P - Q) is a simple
  // convolution.

  // Convolution.
  double tmp[256];
  boxConvolve(tmp, pqm, width);
  boxConvolve(pqm, tmp, width);

  // Dot product.
  double res = 0.0;
  for (int i = 0; i < 256; ++i) {
    res += pqm[i] * pqm[i];
  }
  return res;
}

double CpuHistogramView::getNormalizedSample(int i) const {
  getNormFactor();
  return getNormalizedSampleInternal(i);
}

double CpuHistogramView::getNormalizedSampleInternal(int i) const {
  assert(normFactor >= 0);
  assert(0 <= i && i < 256);
  return normFactor ? storage[i] / (double)normFactor : 0.0;
}

int64_t CpuHistogramView::getNormFactor() const {
  if (normFactor < 0) {
    normFactor = 0;
    for (int i = 0; i < 256; ++i) {
      normFactor += storage[i];
    }
  }
  return normFactor;
}

double CpuHistogramView::getAverage() const {
  if (average < 0.0) {
    average = 0.0;
    for (int i = 0; i < 256; ++i) {
      average += (double)i * storage[i];
    }
    average /= (double)getNormFactor();
  }
  return average;
}

std::string CpuHistogramView::debugString() const {
  std::stringstream ss;
  ss.fill(' ');
  for (int i = 0; i < 256; ++i) {
    ss << std::setw(3) << i << ": " << std::setw(15) << storage[i] << std::endl;
  }
  ss << "total " << getNormFactor() << std::endl;
  return ss.str();
}

RGBCpuHistogramView::RGBCpuHistogramView(const uint32_t* storage)
    : red(storage), green(storage + 256), blue(storage + 512) {}

}  // namespace Image
}  // namespace VideoStitch
