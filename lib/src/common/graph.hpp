// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include <algorithm>
#include <map>
#include <vector>
#include <iostream>

#include <unordered_map>

namespace VideoStitch {

// Kruskal minimum spanning tree algorithm (almost ;) )
template <typename Vertex, typename Edge>
class Graph {
 public:
  class WeightedEdge {
   public:
    WeightedEdge() : weight(0.0), first(Vertex()), second(Vertex()), payload(Edge()) {}
    WeightedEdge(double w, Vertex f, Vertex s, Edge p) : weight(w), first(f), second(s), payload(p) {}

    bool operator<(const WeightedEdge& other) const {
      if (weight != other.weight) {
        return weight < other.weight;
      }
      if (first != other.first) {
        return first < other.first;
      }
      return second < other.second;
    }

    bool operator==(const WeightedEdge& other) const {
      return (weight == other.weight) && (first == other.first) && (second == other.second) &&
             (payload == other.payload);
    }

    double getWeight() const { return weight; }
    void setWeight(double w) { weight = w; }
    Vertex getFirst() const { return first; }
    Vertex getSecond() const { return second; }
    Edge getPayload() const { return payload; }

   protected:
    double weight;
    Vertex first, second;
    Edge payload;
  };

  typedef Graph<Vertex, Edge> This;

  Graph() {}
  explicit Graph(std::vector<WeightedEdge>& e) : edges(e) {
    std::sort(edges.begin(), edges.end());
    for (typename std::vector<This::WeightedEdge>::const_iterator e = edges.begin(); e != edges.end(); ++e) {
      parent[e->getFirst()] = e->getFirst();
      parent[e->getSecond()] = e->getSecond();
    }
  }

  bool operator<(const This& other) const {
    if (edges.empty() && other.edges.empty()) {
      return false;
    }
    if (edges.size() != other.edges.size()) {
      return edges.size() < other.edges.size();
    }
    for (std::size_t indexEdge = 0; indexEdge < edges.size() - 1; ++indexEdge) {
      if (edges[indexEdge] == other.edges[indexEdge]) {
        continue;
      }
      return edges[indexEdge] < other.edges[indexEdge];
    }
    return edges.back() < other.edges.back();
  }

  /**
   * Returns a minimum spanning tree
   */
  std::vector<WeightedEdge> mst() const {
    std::vector<WeightedEdge> mstree;
    for (typename std::vector<This::WeightedEdge>::const_iterator e = edges.begin(); e != edges.end(); ++e) {
      Vertex pu = find(e->getFirst());
      Vertex pv = find(e->getSecond());
      if (pu == pv) {
        continue;
      }
      mstree.push_back(*e);
      parent[pu] = parent[pv];
    }
    return mstree;
  }

  std::vector<WeightedEdge> getEdges() const { return edges; }

  void toDot(std::ostream& ostr) {
    ostr << "digraph G {" << std::endl;
    for (const WeightedEdge& currentEdge : edges) {
      ostr << "\t" << currentEdge.getFirst() << " -> " << currentEdge.getSecond();
      ostr << " [label=\"" << currentEdge.getPayload() << ":" << currentEdge.getWeight() << "\"];" << std::endl;
    }
    ostr << "}" << std::endl;
  }

  double sumCostsAllWeigths() const {
    double sum = 0.;
    typename std::vector<This::WeightedEdge>::const_iterator it;
    for (it = edges.begin(); it != edges.end(); ++it) {
      sum += it->getWeight();
    }
    return sum;
  }

  /** @brief Compute the average cost of incident edges for each vertex
   */
  std::unordered_map<Vertex, double> averageCostsOfIncidentEdges() const {
    std::unordered_map<Vertex, double> costsVertices;
    std::unordered_map<Vertex, int> counterVertices;

    typename std::unordered_map<Vertex, double>::iterator itMapCosts;

    typename std::vector<This::WeightedEdge>::const_iterator itEdges;
    for (itEdges = edges.begin(); itEdges != edges.end(); ++itEdges) {
      const Vertex& firstV = itEdges->getFirst();
      const Vertex& secondV = itEdges->getSecond();
      costsVertices[firstV] += itEdges->getWeight();
      counterVertices[firstV] += 1;
      costsVertices[secondV] += itEdges->getWeight();
      counterVertices[secondV] += 1;
    }

    // compute the average
    for (itMapCosts = costsVertices.begin(); itMapCosts != costsVertices.end(); ++itMapCosts) {
      itMapCosts->second /= counterVertices[itMapCosts->first];
    }
    return costsVertices;
  }

 protected:
  Vertex find(Vertex x) const {
    if (x != parent[x]) {
      parent[x] = find(parent[x]);
    }
    return parent[x];
  }

  std::vector<WeightedEdge> edges;
  mutable std::unordered_map<Vertex, Vertex> parent;
};
}  // namespace VideoStitch

#endif  // GRAPH_HPP_
