// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "geoProps.hpp"

#include <cassert>
#include <cstdlib>

namespace VideoStitch {
namespace Core {

/**
 * Represents an edge in the graph of inputs.
 */
class Edge {
 public:
  /**
   * @param overlap The caller retains ownership.
   */
  Edge(readerid_t fromId, readerid_t toId) : from(fromId), to(toId), opposite_(NULL) {}

  const readerid_t from;
  const readerid_t to;
  OverlapImpl overlap;

  void setOpposite(Edge* opposite) {
    opposite_ = opposite;
    opposite_->opposite_ = this;
  }
  Edge* opposite() const { return opposite_; }

 private:
  Edge* opposite_;  // Not owned.
};

/**
 * Represents a node in the graph of inputs.
 */
class Node {
 public:
  explicit Node(readerid_t id) : id(id) {}

  /**
   * Adds an edge to the node.
   * @param edge The edge to add. The caller retains ownership.
   */
  void addEdge(Edge* edge) {
    assert(edge->from == id);
    assert(edge->to != id);
    edges.push_back(edge);
  }

  /**
   * Return the outgoing edge to @a toId, or NULL.
   * @param toId Id of the edge to find.
   */
  Edge* findEdgeTo(readerid_t toId) const {
    for (size_t i = 0; i < edges.size(); ++i) {
      if (edges[i]->to == toId) {
        return edges[i];
      }
    }
    return NULL;
  }

  /**
   * The node's input id.
   */
  const readerid_t id;

 private:
  std::vector<Edge*> edges;  // Not owned.
};

GeometricPropsImpl::~GeometricPropsImpl() {
  for (size_t i = 0; i < edges.size(); ++i) {
    delete edges[i];
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    delete nodes[i];
  }
}

Node* GeometricPropsImpl::createNodeIfNeeded(readerid_t id) {
  assert(id >= 0);
  if (id >= (int)nodes.size()) {
    nodes.resize(id);
  }
  if (nodes[id] == NULL) {
    nodes[id] = new Node(id);
  }
  return nodes[id];
}

Edge* GeometricPropsImpl::createEdgeIfNeeded(readerid_t firstInput, readerid_t secondInput) {
  return createEdgeIfNeeded(createNodeIfNeeded(firstInput), createNodeIfNeeded(secondInput));
}

Edge* GeometricPropsImpl::createEdgeIfNeeded(Node* firstNode, Node* secondNode) {
  Edge* edge = firstNode->findEdgeTo(secondNode->id);
  if (!edge) {
    edge = new Edge(firstNode->id, secondNode->id);
    edge->setOpposite(new Edge(secondNode->id, firstNode->id));

    edges.push_back(edge);
    edges.push_back(edge->opposite());
    firstNode->addEdge(edge);
    secondNode->addEdge(edge->opposite());
  }
  return edge;
}

void GeometricPropsImpl::setOverlap(readerid_t firstInput, readerid_t secondInput, int numPixels) {
  Edge* edge = createEdgeIfNeeded(firstInput, secondInput);
  edge->overlap.setNumPixels(numPixels);
  edge->opposite()->overlap.setNumPixels(numPixels);
}

const Overlap* GeometricPropsImpl::getOverlap(readerid_t firstInput, readerid_t secondInput) const {
  if (firstInput < 0 || firstInput >= (int)nodes.size()) {
    return NULL;
  }
  if (secondInput < 0 || secondInput >= (int)nodes.size()) {
    return NULL;
  }
  const Edge* edge = nodes[firstInput]->findEdgeTo(secondInput);
  if (edge) {
    return &edge->overlap;
  }
  return NULL;
}
}  // namespace Core
}  // namespace VideoStitch
