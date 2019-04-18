// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "opticalFlow.hpp"
#include "gme.hpp"

#include <algorithm>
#include <sstream>

namespace VideoStitch {
namespace Motion {

template <class iterT>
iterT random_unique(iterT begin, iterT end, size_t num_random) {
  size_t left = std::distance(begin, end);
  while (num_random--) {
    iterT r = begin;
    std::advance(r, rand() % left);
    std::swap(*begin, *r);
    ++begin;
    --left;
  }
  return begin;
}

struct float2MagnitudeCompareDescendingOrder {
  bool operator()(const Motion::ImageSpace::MotionVector& lhs, const Motion::ImageSpace::MotionVector& rhs) const {
    return lhs.magnitude2() > rhs.magnitude2();
  }
};

float OpticalFlow::computeMedianMagnitude2() const {
  if (field.empty()) {
    return 0.0f;
  }

  Motion::ImageSpace::MotionVectorField fieldCopy(field);
  float2MagnitudeCompareDescendingOrder cmp;
  nth_element(fieldCopy.begin(), fieldCopy.begin() + fieldCopy.size() / 2, fieldCopy.end(), cmp);
  return fieldCopy[fieldCopy.size() / 2].magnitude2();
}

std::string OpticalFlow::toString() const {
  std::ostringstream oss;
  oss << input << ":" << frame << ":";
  oss << computeMedianMagnitude2() << ":";

  std::vector<float> magnitudes2;
  for (auto motionVector : field) {
    magnitudes2.push_back(motionVector.magnitude2());
  }
  std::sort(magnitudes2.begin(), magnitudes2.end());
  for (auto mag : magnitudes2) {
    oss << mag << ",";
  }
  return oss.str();
}

void OpticalFlow::applyFactor(float factor) {
  for (std::size_t i = 0; i < field.size(); ++i) {
    field[i].from.x *= factor;
    field[i].from.y *= factor;
    field[i].to.x *= factor;
    field[i].to.y *= factor;
  }
}

void OpticalFlow::sampleMotionVectors(std::size_t nbSamplesToKeep) {
  if (nbSamplesToKeep >= field.size()) {
    return;
  }
  random_unique(field.begin(), field.end(), nbSamplesToKeep);
  field.resize(nbSamplesToKeep);
}

void OpticalFlow::filterSmallMotions(double epsilon) {
  field.erase(std::remove_if(field.begin(), field.end(),
                             [&](const Motion::ImageSpace::MotionVector& x) { return x.magnitude2() < epsilon; }),
              field.end());
}

}  // end namespace Motion

}  // end namespace VideoStitch
