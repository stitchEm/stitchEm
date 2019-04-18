// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <assert.h>
#include <iostream>
#include <vector>

#include "libvideostitch/inputDef.hpp"
#include "core/controllerInputFrames.hpp"
#include "core/geoTransform.hpp"

#include <random>

namespace VideoStitch {
namespace Util {

/**
 * A class that represents a point in an input.
 */
class Point {
 public:
  /**
   * Creates a valid point.
   */
  explicit Point(int videoInputId, Core::TopLeftCoords2 coords)
      : videoInputId_(videoInputId), coords_(coords), color_(make_float3(-1.0f, 0.0f, 0.0f)) {}

  /**
   * Value accessor.
   */
  const float3& color() const {
    assert(hasColor());
    return color_;
  }

  /**
   * Value setter.
   */
  void setColor(const float3& color) { color_ = color; }

  /**
   * Sets value to 'no color'.
   */
  void setNoColor() { color_.x = -1.0f; }

  /**
   * Sets value to 'no color'.
   */
  bool hasColor() const { return color_.x >= 0.0f; }

  /**
   * Coords accessor.
   */
  Core::TopLeftCoords2 coords() const { return coords_; }

  videoreaderid_t videoInputId() const { return videoInputId_; }

 private:
  // Video input id.
  const videoreaderid_t videoInputId_;
  // Coordinates in input space.
  const Core::TopLeftCoords2 coords_;
  // R,G,B samples in the neighbourhood.
  float3 color_;
};

/**
 * Represents a pair of points in video input k and l that have the same image.
 */
class PointPair {
 public:
  /**
   * Creates a point set. Takes ownership.
   */
  PointPair(Point* p_k, Point* p_l, Core::SphericalCoords3 sphericCoords)
      : sphericCoords(sphericCoords), p_k(p_k), p_l(p_l), chosen(false) {}

  PointPair(const PointPair& that)
      : sphericCoords(that.sphericCoords), p_k(new Point(*that.p_k)), p_l(new Point(*that.p_l)), chosen(that.chosen) {}

  ~PointPair() {
    delete p_k;
    delete p_l;
  }

  /**
   * Coordinates in spheric space.
   */
  const Core::SphericalCoords3 sphericCoords;

  /**
   * First point
   */
  Point* const p_k;

  /**
   * Second point.
   */
  Point* const p_l;

  /**
   * Choose this PointPair for the next calibration
   */
  void choose() { chosen = true; }

  /**
   * Reset all point pairs before chosing for current round
   */
  void resetChoice() { chosen = false; }

  /**
   * Should be used for calibration
   */
  bool shouldBeUsed() const {
    if (!chosen) {
      return false;
    }
    if (!p_k->hasColor()) {
      return false;
    }
    if (!p_l->hasColor()) {
      return false;
    }
    return true;
  }

 private:
  bool chosen;
};

class PointPairAtTime {
 public:
  PointPairAtTime(PointPair* pointPair, size_t time) : pointPair_(*pointPair), time_(time) {}

  const PointPair& pointPair() const { return pointPair_; }

  size_t time() const { return time_; }

 private:
  PointPair pointPair_;
  size_t time_;
};

class PointSampler {
 public:
  PointSampler(const Core::PanoDefinition& pano, int maxSampledPoints, int minPointsPerInput, int neighbourhoodSize);

  /*
   * Note: Point inputIDs are video input IDs
   */
  const std::vector<PointPair*>& getPointPairs() const;

  int getMinPointsInOneOutput() const { return minPointsInOneOutput; }

  int getNumConnectedComponents() const { return numConnectedComponents; }

  ~PointSampler();

 private:
  mutable std::default_random_engine generator;
  std::vector<std::shared_ptr<Core::TransformStack::GeoTransform>> transforms;
  std::vector<PointPair*> pointPairs;

  int minPointsInOneOutput;
  int numConnectedComponents;
  videoreaderid_t numFloatingInputs;

  bool isFullyMasked(const unsigned char* maskPixelData, const int inputWidth, const int inputHeight, const int p_x,
                     const int p_y, const int neighbourhoodSize);
};

class RadialPointSampler : public PointSampler {
 public:
  RadialPointSampler(const Core::PanoDefinition& pano, int maxSampledPoints, int minPointsPerInput,
                     int neighbourhoodSize, int numberOfRadialBins);

  ~RadialPointSampler(){};

  // for each input (with videoreaderid_t ID)
  // for each radius bin (with int ID)
  // have a list of belonging point pairs
  const std::map<videoreaderid_t, std::map<int, std::vector<PointPair*>>>& getPointPairsByRadius() const;

 private:
  std::map<videoreaderid_t, std::map<int, std::vector<PointPair*>>> pointVectors;
};

}  // namespace Util
}  // namespace VideoStitch
