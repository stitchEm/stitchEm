// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rigGraph.hpp"

#include "libvideostitch/logging.hpp"

#include <set>
#include <queue>

namespace VideoStitch {
namespace Calibration {

RigGraph::RigGraph() : Graph<unsigned int, Eigen::Matrix3d>(), numNodes(0) {}

RigGraph::RigGraph(const size_t numNodes, EdgeList& e) : Graph<unsigned int, Eigen::Matrix3d>(e), numNodes(numNodes) {}

RigGraph::EdgeList RigGraph::bfsTraversal(const EdgeList& subGraph, const unsigned int base) const {
  EdgeList traversedEdges;
  std::vector<unsigned int> nodesNotVisited;
  nodesNotVisited.push_back(base);  // start exploration from base
  for (unsigned int i = 0; i < (unsigned int)numNodes; ++i) {
    if (i != base) {
      nodesNotVisited.push_back(i);
    }
  }
  while (!nodesNotVisited.empty()) {  // while all nodes haven't been visited create components
    std::queue<unsigned int> nextNodesToVisit;
    nextNodesToVisit.push(nodesNotVisited.front());
    nodesNotVisited.erase(nodesNotVisited.begin());
    while (!nextNodesToVisit.empty()) {  // while the current component hasn't been exhausted, explore
      unsigned int currentNode = nextNodesToVisit.front();
      nextNodesToVisit.pop();
      // for all non visited neighbors of currentNode
      for (EdgeList::const_iterator e = subGraph.begin(); e != subGraph.end(); ++e) {
        if (e->getFirst() != currentNode && e->getSecond() != currentNode) {
          continue;
        }

        const bool forward = (e->getFirst() == currentNode);  // is edge traversed forward or backwards?
        const unsigned int neighbor = (forward) ? e->getSecond() : e->getFirst();
        std::vector<unsigned int>::iterator neighborInListIterator =
            std::find(nodesNotVisited.begin(), nodesNotVisited.end(), neighbor);
        if (neighborInListIterator == nodesNotVisited.end()) {
          continue;
        }

        nextNodesToVisit.push(neighbor);
        nodesNotVisited.erase(neighborInListIterator);  // mark node as visited
        if (forward) {
          traversedEdges.push_back(WeightedEdge(e->getWeight(), currentNode, neighbor, e->getPayload()));
        } else {
          Eigen::Matrix3d inversePayload;
          inversePayload = e->getPayload().inverse();
          traversedEdges.push_back(WeightedEdge(e->getWeight(), currentNode, neighbor, inversePayload));
        }
      }
    }
  }
  return traversedEdges;
}

std::vector<ConnectedComponent> RigGraph::getConnectedComponents() const {
  const unsigned int base = (unsigned int)0;
  const EdgeList traversedEdges = bfsTraversal(edges, base);
  std::vector<unsigned int> idxComponents;
  for (size_t i = 0; i < numNodes; ++i) {
    idxComponents.push_back((unsigned int)i);
  }
  for (EdgeList::const_iterator e = traversedEdges.begin(); e != traversedEdges.end(); ++e) {
    idxComponents[e->getSecond()] = idxComponents[e->getFirst()];
  }
  // Create a list of connected components from the array of connected components indices, e.g.  create {{0, 2}, {1, 3,
  // 4}, {5}} from [0, 1, 0, 1, 1, 2]
  std::vector<ConnectedComponent> connectedComponents;
  std::vector<bool> markedNodes;
  for (size_t i = 0; i < numNodes; ++i) {
    markedNodes.push_back(false);
  }
  for (size_t i = 0; i < numNodes; ++i) {
    if (!markedNodes[i]) {
      ConnectedComponent currentComponent;
      for (size_t j = i; j < numNodes; ++j) {
        if (idxComponents[j] == idxComponents[i]) {
          currentComponent.push_back((unsigned int)j);
          markedNodes[j] = true;
        }
      }
      connectedComponents.push_back(currentComponent);
    }
  }
  return connectedComponents;
}

bool RigGraph::isConnected() const {
  const std::vector<ConnectedComponent> connectedComponents = getConnectedComponents();
  return (connectedComponents.size() == 1);
}

}  // namespace Calibration
}  // namespace VideoStitch
