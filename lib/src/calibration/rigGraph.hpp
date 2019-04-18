// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common/graph.hpp"

#include "libvideostitch/panoDef.hpp"

#include <Eigen/Dense>

namespace VideoStitch {
namespace Calibration {

typedef std::vector<unsigned int> ConnectedComponent;

class RigGraph : public Graph<unsigned int, Eigen::Matrix3d> {
 public:
  typedef std::vector<WeightedEdge> EdgeList;

  RigGraph();
  RigGraph(const size_t numNodes, EdgeList& e);

  bool isConnected() const;
  std::vector<ConnectedComponent> getConnectedComponents() const;
  EdgeList bfsTraversal(const EdgeList& graph, const unsigned int base) const;

 private:
  const size_t numNodes;
};

}  // namespace Calibration
}  // namespace VideoStitch
