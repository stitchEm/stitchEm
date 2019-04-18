// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"

namespace VideoStitch {
namespace Core {

/**
 * Depth estimation definition
 */
class VS_EXPORT DepthDefinition {
 public:
  /**
   * Build with the mandatory fields. The others take default values.
   */
  DepthDefinition();

  virtual ~DepthDefinition();

  /**
   * @brief serialize
   * @param value A ptv value object to fill
   */
  // void serialize(Ptv::Value& value) const;

  /**
   * Comparison operator.
   */
  // bool operator==(const DepthDefinition& other) const;

  /**
   * @brief getNumPyramidLevels
   * @return number of pyramid levels for multi-scale computation
   */
  int getNumPyramidLevels() const;

  /**
   * @brief setNumPyramidLevels
   * @param Number of levels
   */
  void setNumPyramidLevels(int numLevels);

  /**
   * @brief isMultiScale
   * @return whether multi-scale depth processing is enabled
   */
  bool isMultiScale() const;

 private:
  int numPyramidLevels;
};

}  // namespace Core
}  // namespace VideoStitch
