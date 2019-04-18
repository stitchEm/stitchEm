// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef HISTOGRAM_VIEW_HPP_
#define HISTOGRAM_VIEW_HPP_

#include <cassert>
#include <stdint.h>
#include <string>

namespace VideoStitch {
namespace Image {
/**
 * A class that provides a thin wrapper to manipulate histograms.
 */
class CpuHistogramView {
 public:
  /**
   * Creates a view of @a storage as a histogram.
   * @param storage Buffer for storing the values. Must be able to hold 256 elements. Not owned.
   */
  explicit CpuHistogramView(const uint32_t* storage);

  /**
   * Returns the squared L2 distance.
   * @param other target
   */
  double sqrDistanceL2(const CpuHistogramView& other) const;

  /**
   * Returns the squared Chi-2 distance. The factor 1/2 is left out.
   * @param other target
   */
  double sqrDistanceChi2(const CpuHistogramView& other) const;

  /**
   * Returns the Quadratic Form distance, using a double gaussian band matrix.
   * @param other target
   * @param width width of the diagonal band.
   */
  double sqrDistanceQF(const CpuHistogramView& other, int width) const;

  /**
   * Returns the average value. Cached.
   */
  double getAverage() const;

  /**
   * Returns the normalization factor. Cached.
   */
  int64_t getNormFactor() const;

  /**
   * Returns the count for bucket @a i.
   * @param i bucket index. Must be in [0;255]
   */
  uint32_t count(int i) const {
    assert(0 <= i && i < 256);
    return storage[i];
  }

  /**
   * Returns the normalized value for a given bucket.
   * @param i bucket index. Must be in [0;255]
   */
  double getNormalizedSample(int i) const;

  /**
   * Returns a string for debug.
   */
  std::string debugString() const;

 private:
  /**
   * Return the normalized value for a given bucket. The norm factor must be in cache and will not be recomputed.
   * @param i bucket index. Must be in [0;255]
   */
  double getNormalizedSampleInternal(int i) const;

  const uint32_t* storage;
  mutable int64_t normFactor;
  mutable double average;
};

/**
 * Histogram distance functor interface.
 */
class HistoDistance {
 public:
  virtual ~HistoDistance() {}
  virtual double operator()(const Image::CpuHistogramView& histoK, const Image::CpuHistogramView& histoL) const = 0;
};

/**
 * L2 histogram distance functor.
 */
class L2HistoDistance : public HistoDistance {
 public:
  double operator()(const Image::CpuHistogramView& histoK, const Image::CpuHistogramView& histoL) const {
    return histoK.sqrDistanceChi2(histoL);
  }
};

/**
 * Chi2 histogram distance functor.
 */
class Chi2HistoDistance : public HistoDistance {
 public:
  double operator()(const Image::CpuHistogramView& histoK, const Image::CpuHistogramView& histoL) const {
    return histoK.sqrDistanceChi2(histoL);
  }
};

/**
 * QF histogram distance functor.
 */
class QFHistoDistance : public HistoDistance {
 public:
  explicit QFHistoDistance(int width) : width(width) {}
  double operator()(const Image::CpuHistogramView& histoK, const Image::CpuHistogramView& histoL) const {
    return histoK.sqrDistanceQF(histoL, width);
  }

 private:
  const int width;
};

/**
 * A simple holder for 3 histograms in a consecutive buffer.
 */
class RGBCpuHistogramView {
 public:
  /**
   * Creates a view of @a storage as an RGB histogram.
   * @param storage Bbuffer for storing the values. Must be able to hold 3 * 256 elements. Planar. Not owned.
   */
  explicit RGBCpuHistogramView(const uint32_t* storage);

  CpuHistogramView red;
  CpuHistogramView green;
  CpuHistogramView blue;
};

/**
 * For tests. Box filters @a v with the given width.
 * @param dst destination. Must be of size 256.
 * @param src Data to filter. Must be of size 256.
 * @param width bandwidth.
 */
void boxConvolve(double* dst, const double* src, const int width);
}  // namespace Image
}  // namespace VideoStitch
#endif
