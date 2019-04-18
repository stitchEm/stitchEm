// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"

#include <vector>

namespace VideoStitch {
namespace Core {

/**
 * Represents an overlap between two inputs. Public API.
 */
class VS_EXPORT Overlap {
 public:
  virtual ~Overlap() {}

  /**
   * Returns the number of pixels in the overlap.
   */
  virtual int getNumPixels() const = 0;
};

/**
 * @brief A class that stores the geometric properties of a panorama setup. Public API.
 *
 * In particular, it stores the graph of overlaps. Nodes are images, and edges are overlaps.
 *
 */
class VS_EXPORT GeometricProps {
 public:
  virtual ~GeometricProps() {}

  /**
   * Returns the overlap between two inputs, or NULL if the inputs don't overlap.
   * @param firstInput ID of the first input.
   * @param secondInput ID of the second input.
   */
  virtual const Overlap* getOverlap(videoreaderid_t firstInput, videoreaderid_t secondInput) const = 0;
};

class Node;
class Edge;

/**
 * @brief Overlap implementation.
 */
class OverlapImpl : public Overlap {
 public:
  OverlapImpl() : numPixels(0) {}

  virtual int getNumPixels() const { return numPixels; }

  void setNumPixels(int n) { numPixels = n; }

 private:
  int numPixels;
};

/**
 * @brief GeometricProps implementation.
 */
class GeometricPropsImpl : public GeometricProps {
 public:
  virtual ~GeometricPropsImpl();

  /**
   * Returns the overlap between two inputs.
   * @param firstInput First input.
   * @param secondInput Second input.
   */
  virtual const Overlap* getOverlap(videoreaderid_t firstInput, videoreaderid_t secondInput) const;

  /**
   * Sets the overlap between two inputs.
   * @param firstInput First input.
   * @param secondInput Second input.
   * @param numPixels Area of the overlap.
   */
  void setOverlap(videoreaderid_t firstInput, videoreaderid_t secondInput, int numPixels);

 private:
  /**
   * Returns a node and creates it if needed.
   */
  Node* createNodeIfNeeded(videoreaderid_t id);

  /**
   * Returns the edge between two inputs, and creates it if needed.
   * @param firstInput First input.
   * @param secondInput Second input.
   * @Return THe edge. Ownership is retained.
   */
  Edge* createEdgeIfNeeded(videoreaderid_t firstInput, videoreaderid_t secondInput);
  Edge* createEdgeIfNeeded(Node* firstNode, Node* secondNode);

  std::vector<Node*> nodes;  // All nodes are owned here.
  std::vector<Edge*> edges;  // All edges are owned here.
};
}  // namespace Core
}  // namespace VideoStitch
