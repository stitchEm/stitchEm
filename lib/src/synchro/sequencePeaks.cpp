// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sequencePeaks.hpp"

#include "libvideostitch/logging.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <set>

namespace VideoStitch {
namespace Synchro {

bool sortByIncreasingOffset(const offset_t &lhs, const offset_t &rhs) { return lhs.second < rhs.second; }

Status initGaussianKernel1D(const float sigma, const std::size_t kernelSize, std::vector<float> &convKernel) {
  if ((kernelSize % 2) != 1) {
    convKernel.clear();
    return {Origin::SynchronizationAlgorithm, ErrType::UnsupportedAction,
            "The size of the gaussian kernel is expected to be odd. Got value: " + std::to_string(kernelSize)};
  }
  convKernel.resize(kernelSize);
  float sum = 0.f;
  for (std::size_t currentIndex = 0; currentIndex < kernelSize; ++currentIndex) {
    int d = static_cast<int>(kernelSize / 2) - static_cast<int>(currentIndex);
    float val = static_cast<float>(exp(-static_cast<float>(d * d) / (2.f * sigma * sigma)));
    convKernel[currentIndex] = val;
    sum += val;
  }
  for (std::size_t currentIndex = 0; currentIndex < kernelSize; ++currentIndex) {
    convKernel[currentIndex] /= sum;
  }
  return Status::OK();
}

Status smoothUsingKernel(const std::vector<offset_t> &correlations, const std::vector<float> &convKernel,
                         std::vector<offset_t> &smoothedCorrelations) {
#ifndef NDEBUG
  // check that the correlations are in sorted order by increasing offset value
  for (std::size_t currentIndex = 1; currentIndex < correlations.size(); ++currentIndex) {
    int previousOffset = correlations[currentIndex - 1].second;
    int currentOffset = correlations[currentIndex].second;
    if (previousOffset >= currentOffset) {
      smoothedCorrelations.clear();
      return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError,
              "Correlations are expected to be sorted by increasing offsets"};
    }
  }
#endif

  smoothedCorrelations.resize(correlations.size() - convKernel.size() + 1);
  std::size_t kernelRadius = convKernel.size() / 2;
  for (std::size_t currentStartIndex = kernelRadius; currentStartIndex < correlations.size() - kernelRadius;
       ++currentStartIndex) {
    double newVal = 0.f;
    for (std::size_t indexOffset = 0; indexOffset < convKernel.size(); ++indexOffset) {
      double currentVal = correlations[currentStartIndex - kernelRadius + indexOffset].first;
      newVal += currentVal * static_cast<double>(convKernel[indexOffset]);
    }
    smoothedCorrelations[currentStartIndex - kernelRadius] =
        std::make_pair(newVal, correlations[currentStartIndex].second);
  }
  return Status::OK();
}

Status findMinPeaks(const std::vector<offset_t> &correlations, const std::size_t nb, float delta, float hardThreshold,
                    std::vector<offset_t> &peaks) {
  if (correlations.empty()) {
    return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Missing correlation values"};
  }

#ifndef NDEBUG
  // check that the correlations are in sorted order by increasing offset value
  for (std::size_t currentIndex = 1; currentIndex < correlations.size(); ++currentIndex) {
    int previousOffset = correlations[currentIndex - 1].second;
    int currentOffset = correlations[currentIndex].second;
    if (previousOffset >= currentOffset) {
      peaks.clear();
      return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError,
              "Correlations are expected to be sorted by increasing offsets"};
    }
  }
#endif

  std::size_t nbElem = std::min(nb, correlations.size());
  peaks.clear();
  peaks.reserve(nbElem);
  double mn = std::numeric_limits<double>::max();
  double mx = std::numeric_limits<double>::lowest();
  std::size_t mxpos = 0;
  bool lookForMax = true;
  for (std::size_t currentIndex = 0; currentIndex < correlations.size(); ++currentIndex) {
    double currentVal = 1. - correlations[currentIndex].first;
    if (currentVal > mx) {
      mx = currentVal;
      mxpos = currentIndex;
    }
    if (currentVal < mn) {
      mn = currentVal;
    }

    if (lookForMax) {
      if (currentVal < (mx * static_cast<double>(delta))) {
        if (mx > (1. - static_cast<double>(hardThreshold))) {
          peaks.push_back(correlations[mxpos]);
        }
        mn = currentVal;
        lookForMax = false;
      }
    } else {
      if (currentVal > (mn + mn * (1. - static_cast<double>(delta)))) {
        mx = currentVal;
        mxpos = currentIndex;
        lookForMax = true;
      }
    }
  }

  if (peaks.empty()) {
    // if no peak stands up given the thresholds, at least return the global minimum
    std::vector<offset_t>::const_iterator itMin = std::min_element(correlations.begin(), correlations.end());
    if (itMin == correlations.end()) {
      return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError,
              "Could not determine the smallest element in correlations"};
    }
    peaks.push_back(*itMin);
  } else {
    nbElem = std::min(nbElem, peaks.size());
    std::partial_sort(peaks.begin(), peaks.begin() + nbElem, peaks.end());
    peaks.resize(nbElem);
  }
  return Status::OK();
}

Status getAllConsistentEdges(const std::vector<Graph<readerid_t, int>::WeightedEdge> &mst, std::size_t nbVertices,
                             const std::vector<std::vector<std::vector<double> > > &allOffsets, int minOffset,
                             std::vector<Graph<readerid_t, int>::WeightedEdge> &allConsitentEdges) {
  if (mst.size() != nbVertices - 1) {
    allConsitentEdges.clear();
    return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Inconsistent edge values"};
  }
  if (allOffsets.size() != nbVertices) {
    allConsitentEdges.clear();
    return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Inconsistent edge values"};
  }

  std::set<Graph<videoreaderid_t, int>::WeightedEdge> tempEdgesSet(mst.begin(), mst.end());
  std::pair<std::set<Graph<videoreaderid_t, int>::WeightedEdge>::iterator, bool> ret;
  bool added = true;

  // loop until we stop adding new edges
  while (added) {
    added = false;
    std::vector<Graph<videoreaderid_t, int>::WeightedEdge> tempEdgesVect(tempEdgesSet.begin(), tempEdgesSet.end());

    for (std::size_t indexFirstEdge = 0; indexFirstEdge < tempEdgesVect.size(); ++indexFirstEdge) {
      const Graph<videoreaderid_t, int>::WeightedEdge &firstEdge = tempEdgesVect[indexFirstEdge];
      for (std::size_t indexSecondEdge = indexFirstEdge + 1; indexSecondEdge < tempEdgesVect.size();
           ++indexSecondEdge) {
        const Graph<videoreaderid_t, int>::WeightedEdge &secondEdge = tempEdgesVect[indexSecondEdge];

        int newEdgeSource, newEdgeDest, newEdgeOffset;
        if (firstEdge.getFirst() == secondEdge.getFirst()) {
          if (firstEdge.getSecond() < secondEdge.getSecond()) {
            newEdgeSource = firstEdge.getSecond();
            newEdgeDest = secondEdge.getSecond();
            newEdgeOffset = secondEdge.getPayload() - firstEdge.getPayload();
          } else {
            newEdgeSource = secondEdge.getSecond();
            newEdgeDest = firstEdge.getSecond();
            newEdgeOffset = firstEdge.getPayload() - secondEdge.getPayload();
          }
        } else if (firstEdge.getSecond() == secondEdge.getSecond()) {
          if (firstEdge.getFirst() < secondEdge.getFirst()) {
            newEdgeSource = firstEdge.getFirst();
            newEdgeDest = secondEdge.getFirst();
            newEdgeOffset = firstEdge.getPayload() - secondEdge.getPayload();
          } else {
            newEdgeSource = secondEdge.getFirst();
            newEdgeDest = firstEdge.getFirst();
            newEdgeOffset = secondEdge.getPayload() - firstEdge.getPayload();
          }
        } else if (firstEdge.getFirst() == secondEdge.getSecond()) {
          newEdgeSource = secondEdge.getFirst();
          newEdgeDest = firstEdge.getSecond();
          newEdgeOffset = firstEdge.getPayload() + secondEdge.getPayload();
        } else if (firstEdge.getSecond() == secondEdge.getFirst()) {
          newEdgeSource = firstEdge.getFirst();
          newEdgeDest = secondEdge.getSecond();
          newEdgeOffset = firstEdge.getPayload() + secondEdge.getPayload();
        } else {
          // the two edges don't have any vertex in common
          continue;
        }

        double newEdgeCost;
        if ((newEdgeOffset < minOffset) ||
            (newEdgeOffset - minOffset >= static_cast<int>(allOffsets.at(newEdgeSource).at(newEdgeDest).size()))) {
          // this offset is out of bounds and does not correspond to any computed cross-correlation.
          // Add an infinite cost to penalize this solution
          newEdgeCost = std::numeric_limits<double>::max();
        } else {
          newEdgeCost = allOffsets.at(newEdgeSource).at(newEdgeDest).at(newEdgeOffset - minOffset);
        }
        ret = tempEdgesSet.insert(
            Graph<videoreaderid_t, int>::WeightedEdge(newEdgeCost, newEdgeSource, newEdgeDest, newEdgeOffset));
        if (ret.second) {
          added = true;
        }
      }
    }
  }
  allConsitentEdges.resize(tempEdgesSet.size());
  std::copy(tempEdgesSet.begin(), tempEdgesSet.end(), allConsitentEdges.begin());
  std::sort(allConsitentEdges.begin(), allConsitentEdges.end());
  return Status::OK();
}

Status reweightGraphUsingOffsetConsistencyCriterion(
    const Graph<videoreaderid_t, int> &inputGraph, size_t nbVertices,
    std::vector<Graph<videoreaderid_t, int>::WeightedEdge> &reweightedEdges) {
  std::vector<std::vector<int> > vectInitialCosts(nbVertices), vectNewCosts(nbVertices);
  for (std::size_t i = 0; i < vectInitialCosts.size(); ++i) {
    vectInitialCosts[i].resize(nbVertices);
    vectNewCosts[i].resize(nbVertices, 0);
  }

  reweightedEdges = inputGraph.getEdges();

  // initial costs
  for (std::size_t edgeIndex = 0; edgeIndex < reweightedEdges.size(); ++edgeIndex) {
    int edgeSrc = reweightedEdges[edgeIndex].getFirst();
    int edgeDst = reweightedEdges[edgeIndex].getSecond();
    int edgeOffset = reweightedEdges[edgeIndex].getPayload();
    vectInitialCosts[edgeSrc][edgeDst] = edgeOffset;
  }

  // compute new costs
  for (std::size_t firstIndex = 0; firstIndex < nbVertices - 2; ++firstIndex) {
    for (std::size_t middleIndex = firstIndex + 1; middleIndex < nbVertices - 1; ++middleIndex) {
      for (std::size_t lastIndex = middleIndex + 1; lastIndex < nbVertices; ++lastIndex) {
        int costCycle = abs(vectInitialCosts[firstIndex][middleIndex] + vectInitialCosts[middleIndex][lastIndex] -
                            vectInitialCosts[firstIndex][lastIndex]);
        vectNewCosts[firstIndex][middleIndex] += costCycle;
        vectNewCosts[firstIndex][lastIndex] += costCycle;
        vectNewCosts[middleIndex][lastIndex] += costCycle;
      }
    }
  }

  // reweight the set of edges
  for (std::size_t edgeIndex = 0; edgeIndex < reweightedEdges.size(); ++edgeIndex) {
    int edgeSrc = reweightedEdges[edgeIndex].getFirst();
    int edgeDst = reweightedEdges[edgeIndex].getSecond();
    reweightedEdges[edgeIndex].setWeight(vectNewCosts[edgeSrc][edgeDst]);
  }

  return Status::OK();
}

bool isHighConfidenceMST(const std::vector<Graph<readerid_t, int>::WeightedEdge> &mst, int minimumOffset,
                         const std::vector<std::vector<std::vector<double> > > &allOffsets) {
  if (allOffsets.empty()) {
    Logger::get(Logger::Verbose) << "isHighConfidenceMST(): allOffsets container is empty" << std::endl;
    return false;
  }
  bool success = true;

  std::vector<Graph<readerid_t, int>::WeightedEdge>::const_iterator itEdge;
  for (itEdge = mst.begin(); itEdge != mst.end(); ++itEdge) {
    const readerid_t &firstV = itEdge->getFirst();
    const readerid_t &secondV = itEdge->getSecond();
    const int &offsetEdge = itEdge->getPayload();

    double totalSum = 0.;
    for (std::size_t currentIndexToSum = 0; currentIndexToSum < allOffsets[firstV][secondV].size();
         ++currentIndexToSum) {
      totalSum += 1.0 - allOffsets[firstV][secondV][currentIndexToSum];
    }
    double average = totalSum / allOffsets[firstV][secondV].size();
    totalSum = 0.;
    for (std::size_t currentIndexToSum = 0; currentIndexToSum < allOffsets[firstV][secondV].size();
         ++currentIndexToSum) {
      double currentVal = (1.0 - allOffsets[firstV][secondV][currentIndexToSum]) - average;
      if (currentVal > 0.) {
        totalSum += currentVal;
      }
    }

    if (totalSum == 0) {
      Logger::get(Logger::Verbose) << "isHighConfidenceMST(): totalSum is 0" << std::endl;
      return false;
    }

    int firstIndexToSum = static_cast<int>((offsetEdge - minimumOffset) - 0.005 * allOffsets[firstV][secondV].size());
    int lastIndexToSum = static_cast<int>((offsetEdge - minimumOffset) + 0.005 * allOffsets[firstV][secondV].size());
    if (firstIndexToSum < 0) {
      firstIndexToSum = 0;
    }
    if (lastIndexToSum >= static_cast<int>(allOffsets[firstV][secondV].size())) {
      lastIndexToSum = static_cast<int>(allOffsets[firstV][secondV].size()) - 1;
    }
    double currentSum = 0.;
    for (int currentIndexToSum = firstIndexToSum; currentIndexToSum <= lastIndexToSum; ++currentIndexToSum) {
      double currentVal = (1.0 - allOffsets[firstV][secondV][currentIndexToSum]) - average;
      if (currentVal > 0.) {
        currentSum += currentVal;
      }
    }

    if ((currentSum / totalSum) < 0.1) {
      success = false;
    }
  }

  return success;
}

}  // namespace Synchro
}  // namespace VideoStitch
