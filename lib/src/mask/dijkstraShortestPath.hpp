// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "core/rect.hpp"
#include "gpu/vectorTypes.hpp"
#include "libvideostitch/status.hpp"
#include "mergerMaskConstant.hpp"

#include <vector>

namespace VideoStitch {
namespace MergerMask {
/**
 * @brief Find the shortest path of a 4-connected matrix using Dijkstra algorithm
 */
class DijkstraShortestPath {
 public:
  /**
   * @brief Find the shortest path from source to target, given a cost buffer
   */
  Status find(const int wrapWidth, const Core::Rect& rect, const std::vector<float>& costsBuffer, const int2& source,
              const int2& target, std::vector<unsigned char>& directions, const bool wrapPath = true);
};

}  // namespace MergerMask
}  // namespace VideoStitch
