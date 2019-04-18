// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common/graph.hpp"

#include "libvideostitch/status.hpp"

#include <vector>

namespace VideoStitch {
namespace Synchro {

typedef std::pair<double, int> offset_t;

bool sortByIncreasingOffset(const offset_t& lhs, const offset_t& rhs);

/** @brief Initialize a 1D convolution Kernel
 *
 * @param sigma (input): standard deviation of the gaussian kernel
 * @param kernelSize (input): size of the convolution mask. Must be an odd value (otherwise the function returns an
 * empty convKernel)
 * @param convKernel (output): convolution mask
 * @return code: returns error if @param kernelSize is even
 */
Status initGaussianKernel1D(const float sigma, const std::size_t kernelSize, std::vector<float>& convKernel);

/** @brief convolve using the specified kernel
 *
 * @param correlations (input): sequence of pairs (corr, offset). Expected to be sorted by increasing offsets.
 * @param convKernel (input): convolution mask
 * @param smoothedCorrelations (output): sequence of pair (corr, offset)
 * @return code: returns error if @param correlations are not sorted by increasing offsets
 */
Status smoothUsingKernel(const std::vector<offset_t>& correlations, const std::vector<float>& convKernel,
                         std::vector<offset_t>& smoothedCorrelations);

/** @brief Find the k more relevant local minimums
 *  inspired by this algorithm : www.billauer.co.il/peakdet.html
 *
 * @param correlations (input): sequence of pairs (corr, offset). Expected to be sorted by increasing offsets.
 * @param nb (intput): number of max peak we want to retrieve
 * @param delta (input): the next valley should be lower than the previous peak multiplied by this factor
 * @param hardThreshold (input): values over which local downward peaks are disregarded
 * @param peaks (output): principal downward peaks, sorted by increasing cost
 *
 * If no relevant peak can be found given the threshold parameters, the function will return the global minimum
 */
Status findMinPeaks(const std::vector<offset_t>& correlations, const std::size_t nb, float delta, float hardThreshold,
                    std::vector<offset_t>& peaks);

/** @brief Retrieve the full list of edges consistent with the MST
 *
 * Assumes the graph is densely connected, with edges going from lower vertex indices to upper vertex indices
 *
 * @param mst (input): set of edges which constitute the MST
 * @param nbVertices (input): number of vertices in the underlying graph
 * @param allOffsets (input): structure which contains the pairwise costs for all sources and offsets
 *                            allOffsets.size() must be consistent with nbVertices
 * @param minOffset (intput): minimum offset in allOffsets
 * @param allConsistentEdges (output): all n*(n-1)/2 edges consistent with the MST
 */
Status getAllConsistentEdges(const std::vector<Graph<readerid_t, int>::WeightedEdge>& mst, std::size_t nbVertices,
                             const std::vector<std::vector<std::vector<double> > >& allOffsets, int minOffset,
                             std::vector<Graph<readerid_t, int>::WeightedEdge>& allConsitentEdges);

/** @brief Reweight graph using consistency criterion
 * inspired by this paper : Temporal Synchronization of Multiple Audio Signals (ICASSP 2014)
 * http://research.google.com/pubs/pub42193.html
 *
 * If the offsets were consistent in the input graph, the sum of the offsets along each cycle would be 0.
 * Let us define the cost of a cycle as being equal to the sum of the offsets along this cycle.
 * Let us define the new weight of an edge as being equal to the sum of the costs of the cycles along all
 * 3-cliques this edge is a member of.
 *
 * Assumes the graph is densely connected, with edges going from lower vertex indices to upper vertex indices
 *
 * @param inputGraph (input): graph composed of best pairwise costs
 * @param nbVertices (input): number of vertices in the graph
 * @param reweightedEdges (output): set of reweighted edges
 */
Status reweightGraphUsingOffsetConsistencyCriterion(const Graph<readerid_t, int>& inputGraph, std::size_t nbVertices,
                                                    std::vector<Graph<readerid_t, int>::WeightedEdge>& reweightedEdges);

/** @brief Check if mst is high confidence, or if other solution were also eligible
 *
 * @param mst (input): set of edges which constitute the MST
 * @param minimumOffset (input): value of the minimal offset if allOffsets
 * @param allOffsets (input): structure which contains the pairwise consts for all sources and offsets
 *
 * @return false if the mst is low confidence (other low cost solutions could be found).
 *
 */
bool isHighConfidenceMST(const std::vector<Graph<readerid_t, int>::WeightedEdge>& mst, int minimumOffset,
                         const std::vector<std::vector<std::vector<double> > >& allOffsets);

}  // namespace Synchro
}  // namespace VideoStitch
