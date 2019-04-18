// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "seamFinder.hpp"

#include "dijkstraShortestPath.hpp"

#include "gpu/hostBuffer.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/core1/voronoi.hpp"
#include "util/pngutil.hpp"
#include "util/geometryProcessingUtils.hpp"

#include <unordered_map>
#include <queue>
#include <stack>
#include <cmath>

//#define SEAMFINDER_DEBUG

#ifdef SEAMFINDER_DEBUG
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include "util/pngutil.hpp"
#include "util/pnm.hpp"
#include "util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace MergerMask {

Potential<SeamFinder> SeamFinder::create(const int seamFeatheringSize, const int kernelSize, const int wrapWidth,
                                         const Core::Rect& rect0, GPU::Buffer<const uint32_t> input0Buffer,
                                         const Core::Rect& rect1, GPU::Buffer<const uint32_t> input1Buffer,
                                         GPU::Stream& stream, GPU::SharedBuffer<unsigned char> workMask,
                                         GPU::SharedBuffer<uint32_t> work1, GPU::SharedBuffer<uint32_t> work2) {
  std::unique_ptr<SeamFinder> seamFinder;
  seamFinder.reset(new SeamFinder(seamFeatheringSize, kernelSize, wrapWidth, rect0, input0Buffer, rect1, input1Buffer,
                                  stream, workMask, work1, work2));
  FAIL_RETURN(seamFinder->setup());
  return seamFinder.release();
}

Status SeamFinder::findCostMap() {
  // I need to do this step at construction before anything else
  // due to memory allocation failure

  Core::Rect iRect, uRect;
  Core::Rect::getInterAndUnion(rect0, rect1, iRect, uRect, wrapWidth);
  // The cost function is computed on a grid that cross intersects with the original array
  // So its width and height are increased by 1
  Core::Rect costsRect = Core::Rect::fromInclusiveTopLeftBottomRight(
      iRect.top(), iRect.left(), iRect.top() + iRect.getHeight(), iRect.left() + iRect.getWidth());

  // Get data on the host buffer
  costs.resize(costsRect.getArea() * SEAM_DIRECTION);
  GPU::UniqueBuffer<float> costsBuffer;
  PROPAGATE_FAILURE_STATUS(costsBuffer.alloc(costsRect.getArea() * SEAM_DIRECTION, "Seam Finder"));
  PROPAGATE_FAILURE_STATUS(prepareSeamCostBuffer(costsRect, costsBuffer.borrow()));
  PROPAGATE_FAILURE_STATUS(memcpyBlocking(&costs[0], costsBuffer.borrow_const()));
  return Status::OK();
}

Status SeamFinder::setup() {
  FAIL_RETURN(findCostMap());

  if (!workMask.borrow().wasAllocated()) {
    FAIL_RETURN(workMask.alloc(rect.getArea(), "Seam Finder"));
  }
  if (!work1.borrow().wasAllocated()) {
    FAIL_RETURN(work1.alloc(rect.getArea(), "Seam Finder"));
  }
  if (!work2.borrow().wasAllocated()) {
    FAIL_RETURN(work2.alloc(rect.getArea(), "Seam Finder"));
  }
  FAIL_RETURN(outputBuffer.alloc(rect.getArea(), "Seam Finder"));
  FAIL_RETURN(outputsMapBuffer.alloc(rect.getArea(), "Seam Finder"));
  FAIL_RETURN(inputsMapBuffer.alloc(rect.getArea(), "Seam Finder"));
  return Status::OK();
}

SeamFinder::SeamFinder(const int seamFeatheringSize, const int kernelSize, const int wrapWidth, const Core::Rect& rect0,
                       GPU::Buffer<const uint32_t> input0Buffer, const Core::Rect& rect1,
                       GPU::Buffer<const uint32_t> input1Buffer, GPU::Stream& stream,
                       GPU::SharedBuffer<unsigned char> workMask, GPU::SharedBuffer<uint32_t> work1,
                       GPU::SharedBuffer<uint32_t> work2)
    : wrapWidth(wrapWidth),
      id0(0),
      rect0(rect0),
      input0Buffer(input0Buffer),
      id1(1),
      rect1(rect1),
      input1Buffer(input1Buffer),
      stream(stream),
      kernelSize(kernelSize),
      seamFeatheringSize(seamFeatheringSize),
      rect(rect0),
      workMask(workMask),
      work1(work1),
      work2(work2) {}

Status SeamFinder::addDistanceCost(const int supportDistance, const int2& size, const std::vector<int2>& cachedPoints,
                                   const std::vector<float>& inputCost, std::vector<float>& outputCost) {
  const int2 inputSize = make_int2((int)rect.getWidth(), (int)rect.getHeight());
  std::vector<uint32_t> inputIndex(inputSize.x * inputSize.y, 0);
  for (size_t i = 0; i < cachedPoints.size(); i++) {
    inputIndex[cachedPoints[i].y * inputSize.x + cachedPoints[i].x] = 1;
  }
  PROPAGATE_FAILURE_STATUS(
      GPU::memcpyBlocking(outputBuffer.borrow(), &inputIndex[0], inputSize.x * inputSize.y * sizeof(uint32_t)));
  const int supportWidth = std::min(supportDistance, 100);
  PROPAGATE_FAILURE_STATUS(Core::computeEuclideanDistanceMap(workMask.borrow(), outputBuffer.borrow(), work1.borrow(),
                                                             work2.borrow(), inputSize.x, inputSize.y, 1, true,
                                                             (float)supportWidth, 1, stream));

  std::vector<unsigned char> distance(inputSize.x * inputSize.y);
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking(&distance[0], workMask.borrow()));

#ifdef SEAMFINDER_DEBUG
  /*{
    std::string workingPath =
  "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/"; std::stringstream ss;
    ss << workingPath << "data/seam/distanceMap-" << "-result.png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), workMask.borrow(), inputSize.x, inputSize.y);
  }
  {
    std::string workingPath =
  "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/"; std::stringstream ss;
    ss << workingPath << "data/seam/cachedPoint-" << "-result.png";
    Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), inputIndexBuffer.borrow(), inputSize.x, inputSize.y);
  }*/
#endif
  outputCost.clear();
  outputCost.resize(size.x * size.y * 4);
  for (int i = 0; i < size.x; i++)
    for (int j = 0; j < size.y; j++) {
      for (int t = 0; t < 4; t++) {
        const int index = (j * size.x + i);
        if (i < inputSize.x && j < inputSize.y && distance[j * inputSize.x + i] < 255) {
          outputCost[4 * index + t] = inputCost[4 * index + t];
        } else {
          outputCost[4 * index + t] = inputCost[4 * index + t] + PENALTY_COST;
        }
      }
    }
  return Status::OK();
}

Status SeamFinder::addFeatheringCost(const int2& size, const std::vector<float>& inputCost,
                                     std::vector<float>& outputCost) {
  const int2 inputSize = make_int2((int)rect.getWidth(), (int)rect.getHeight());

  PROPAGATE_FAILURE_STATUS(findFeatheringMask(inputSize, getInputsMap(), outputBuffer.borrow(), stream));
  PROPAGATE_FAILURE_STATUS(Core::computeEuclideanDistanceMap(workMask.borrow(), outputBuffer.borrow(), work1.borrow(),
                                                             work2.borrow(), inputSize.x, inputSize.y, 1, true,
                                                             (float)seamFeatheringSize, 1, stream));

#ifdef SEAMFINDER_DEBUG
  /*{
    std::string workingPath =
  "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/"; std::stringstream ss;
    ss << workingPath << "data/seam/featheringMap-" << "-result.png";
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(ss.str().c_str(), workMask.borrow(), inputSize.x, inputSize.y);
  }
  {
    std::string workingPath =
  "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/"; std::stringstream ss;
    ss << workingPath << "data/seam/inputsMap-" << "-result.png";
    Debug::dumpRGBAIndexDeviceBuffer<unsigned char>(ss.str().c_str(), getInputsMap(), inputSize.x, inputSize.y);
  }
  {
    std::string workingPath =
  "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/"; std::stringstream ss;
    ss << workingPath << "data/seam/feathers_input-" << "-result.png";
    Debug::dumpRGBAIndexDeviceBuffer<uint32_t>(ss.str().c_str(), src.borrow(), inputSize.x, inputSize.y);
  }*/
#endif
  std::vector<unsigned char> distance(inputSize.x * inputSize.y);
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking(&distance[0], workMask.borrow()));
  outputCost.clear();
  outputCost.resize(size.x * size.y * 4);
  for (int i = 0; i < size.x; i++)
    for (int j = 0; j < size.y; j++) {
      for (int t = 0; t < 4; t++) {
        const int index = (j * size.x + i);
        if (i < inputSize.x && j < inputSize.y && distance[j * inputSize.x + i] < 255) {
          outputCost[4 * index + t] = inputCost[4 * index + t] + PENALTY_COST;
        } else {
          outputCost[4 * index + t] = inputCost[4 * index + t];
        }
      }
    }
  return Status::OK();
}

Status SeamFinder::getRidOfOnePixelWidthInputsMap(const unsigned char borderIndex) {
  const int2 size = make_int2((int)rect.getWidth(), (int)rect.getHeight());
  std::vector<unsigned char> inputsMap(inputsMapBuffer.borrow().numElements());
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking(&inputsMap[0], inputsMapBuffer.borrow()));
  // Only consider borders at the overlapping area
  for (int i = 0; i < size.x; i++)
    for (int j = 0; j < size.y; j++)
      if ((inputsMap[j * size.x + i] & borderIndex) > 0) {
        int2 coord = make_int2(i, j);
        int2 connectedPoint;
        while (isConnectedByOnePixel(size, inputsMap, borderIndex, coord, connectedPoint)) {
          inputsMap[coord.y * size.x + coord.x] -= borderIndex;
          coord = connectedPoint;
        }
      }
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking(inputsMapBuffer.borrow(), &inputsMap[0], inputsMap.size()));
  return Status::OK();
}

void SeamFinder::findDisconnectedComponent(const int wrapWidth, const int componentIndex, const int2& startPoint,
                                           const int2& offset, const int2& size, const std::vector<bool>& edges,
                                           const int2& inputsOffset, const int2& inputsSize,
                                           const std::vector<unsigned char>& inputsMap, std::vector<int>& components) {
  // NOTE: set wrapWidth < 0 to treat the special case of full overlapping images
  std::stack<int2> pointStack;
  pointStack.push(startPoint);
  components[startPoint.y * size.x + startPoint.x] = componentIndex;
  while (!pointStack.empty()) {
    const int2 headPoint = pointStack.top();
    const int2 headInputsPoint =
        make_int2(headPoint.x + offset.x - inputsOffset.x, headPoint.y + offset.y - inputsOffset.y);
    const int headInputId = inputsMap[headInputsPoint.y * inputsSize.x + headInputsPoint.x];
    pointStack.pop();
    for (int t = 0; t < 4; t++)
      if (edges[t * size.x * size.y + headPoint.y * size.x + headPoint.x]) {
        const int2 next =
            make_int2((headPoint.x + seam_dir_rows[t] + wrapWidth) % wrapWidth, headPoint.y + seam_dir_cols[t]);
        const int2 nextInputsPoint = make_int2((headInputsPoint.x + seam_dir_rows[t] + wrapWidth) % wrapWidth,
                                               headInputsPoint.y + seam_dir_cols[t]);
        if (next.x >= 0 && next.x < size.x && next.y >= 0 && next.y < size.y && nextInputsPoint.x >= 0 &&
            nextInputsPoint.x < inputsSize.x && nextInputsPoint.y >= 0 && nextInputsPoint.y < inputsSize.y) {
          const int nextInputId = inputsMap[nextInputsPoint.y * inputsSize.x + nextInputsPoint.x];
          if (components[next.y * size.x + next.x] < 0 && nextInputId == headInputId) {
            components[next.y * size.x + next.x] = componentIndex;
            pointStack.push(next);
          }
        }
      }
  }
}

Status SeamFinder::findSeam() {
  startPoints.clear();
  endPoints.clear();
  all_directions.clear();
  // First, setup the inputsMapBuffer
  Core::Rect iRect, uRect;
  Core::Rect::getInterAndUnion(rect0, rect1, iRect, uRect, wrapWidth);
  rect = iRect;
  PROPAGATE_FAILURE_STATUS(findInputsMap());
  const int id = 1;
  PROPAGATE_FAILURE_STATUS(getRidOfOnePixelWidthInputsMap(1 << id));

  // The cost function is computed on a grid that cross intersects with the original array
  // So its width and height are increased by 1
  Core::Rect costsRect = Core::Rect::fromInclusiveTopLeftBottomRight(
      iRect.top(), iRect.left(), iRect.top() + iRect.getHeight(), iRect.left() + iRect.getWidth());
  // Find the intersecting segments
  PROPAGATE_FAILURE_STATUS(findIntersectingSegments(1 << id));
  if (startPoints.size() > 0) {
    // Find the shortest path per pair in the CostRect direction
    DijkstraShortestPath shortestPath;
    // Couldn't decide whether to add the feathering cost term, turn it off for now
    if (seamFeatheringSize) {
      std::vector<float> featheringCosts;
      PROPAGATE_FAILURE_STATUS(
          addFeatheringCost(make_int2((int)costsRect.getWidth(), (int)costsRect.getHeight()), costs, featheringCosts));
      costs = featheringCosts;
    }
    for (size_t t = 0; t < startPoints.size(); t++) {
      const int2 startPoint = startPoints[t];
      const int2 endPoint = endPoints[t];
      const bool wrapPath = wrapPaths[t];
      const int distance = distanceCosts[t];
      std::vector<float> distanceCosts;

      // Modify the cost function, enforce the seam to go around the borders.
      // This part is used for the cycle seam
      if (distance) {
        PROPAGATE_FAILURE_STATUS(addDistanceCost(distance,
                                                 make_int2((int)costsRect.getWidth(), (int)costsRect.getHeight()),
                                                 all_cachePoints[t], costs, distanceCosts));
      }
      // The number of horizontal vertices is wrapWidth + 1
      std::vector<unsigned char> directions;
      shortestPath.find(wrapWidth + 1, costsRect, distance ? distanceCosts : costs, startPoint, endPoint, directions,
                        wrapPath);
      all_directions.push_back(directions);
    }
    std::vector<int> components;
    PROPAGATE_FAILURE_STATUS(findConnectedComponentsAfterCuts(id, components));
    PROPAGATE_FAILURE_STATUS(findOutputsMap(id, components));
  }
  return Status::OK();
}

bool SeamFinder::seamFound() const { return startPoints.size() > 0; }

bool SeamFinder::isConnectedByOnePixel(const int2& size, const std::vector<unsigned char>& inputsMap,
                                       const unsigned char value, const int2& coord, int2& connectedPoint) {
  int hitCount = 0;
  for (int t = 0; t < 8; t++) {
    const int2 nextPoint = make_int2(coord.x + seam_dir_rows[t], coord.y + seam_dir_cols[t]);
    if (nextPoint.x >= 0 && nextPoint.y >= 0 && nextPoint.x < size.x && nextPoint.y < size.y &&
        (inputsMap[nextPoint.y * size.x + nextPoint.x] & value) == value) {
      hitCount++;
      connectedPoint = nextPoint;
    }
  }
  return (hitCount == 1);
}

bool SeamFinder::getCurveEnds(const int wrapWidth, const int2& startPoint, const unsigned char value, const int2& size,
                              const std::vector<unsigned char>& borders, std::vector<bool>& visited,
                              std::vector<std::vector<int2>>& curveCachedPoints,
                              std::vector<std::vector<int2>>& curveRootPoints, int& distanceCost) {
  distanceCost = 0;
  curveCachedPoints.clear();
  curveRootPoints.clear();

  std::vector<int2> cachedPoints;
  std::vector<int2> rootPoints;
  cachedPoints.clear();

  // Perform DFS to find all to cache all the point and potential start and end of the curves
  std::stack<int2> pointStack;
  pointStack.push(startPoint);
  visited[startPoint.y * size.x + startPoint.x] = true;
  if ((borders[startPoint.y * size.x + startPoint.x] & value) == value) {
    cachedPoints.push_back(startPoint);
  }

  while (!pointStack.empty()) {
    const int2 topPoint = pointStack.top();
    pointStack.pop();
    int hitCount = 0;
    for (int t = 0; t < 8; t++) {
      const int2 nextPoint =
          make_int2((topPoint.x + seam_dir_rows[t] + wrapWidth) % wrapWidth, topPoint.y + seam_dir_cols[t]);
      if (nextPoint.x >= 0 && nextPoint.y >= 0 && nextPoint.x < size.x && nextPoint.y < size.y &&
          (borders[nextPoint.y * size.x + nextPoint.x] & value) == value) {
        hitCount++;
        // Get rid of the 2nd bit, it is used for other purpose
        if (!visited[nextPoint.y * size.x + nextPoint.x]) {
          visited[nextPoint.y * size.x + nextPoint.x] = true;
          cachedPoints.push_back(nextPoint);
          pointStack.push(nextPoint);
        }
      }
    }
    if (hitCount == 1) {
      rootPoints.push_back(topPoint);
    }
  }

  // Check if this is a wrap-around situation
  if (rootPoints.size() == 0 && cachedPoints.size() > 2) {  // This is a loop, just find the two points randomly
    bool wrapAround = true;

    std::vector<bool> horizontal(size.x, false);
    std::vector<int> pointIndex(size.x, -1);
    for (size_t i = 0; i < cachedPoints.size(); i++) {
      horizontal[cachedPoints[i].x] = true;
      pointIndex[cachedPoints[i].x] = (int)i;
    }

    // Check if this is a wrap around circle, if yes, pick points at both ends and do not allow wraping
    wrapAround = false;
    for (bool b : horizontal) {
      if (!b) {
        wrapAround = true;
        break;
      }
    }
    if (!wrapAround) {
      // If it is the circular case
      if (pointIndex[0] >= 0 && pointIndex[size.x - 1] >= 0) {
        rootPoints.push_back(cachedPoints[pointIndex[0]]);
        rootPoints.push_back(cachedPoints[pointIndex[size.x - 1]]);
        // Get the set of point from here
        curveRootPoints.push_back(rootPoints);
        curveCachedPoints.push_back(cachedPoints);
        distanceCost = 100;
        return wrapAround;
      }
    } else {
      // else, pick the pair that is "probably" largest
      // and make 2 curves out of this
      std::vector<int2> tmpCachePoints;
      tmpCachePoints.insert(tmpCachePoints.begin(), cachedPoints.begin(),
                            cachedPoints.begin() + cachedPoints.size() / 2);
      curveCachedPoints.push_back(tmpCachePoints);
      rootPoints.clear();
      rootPoints.push_back(tmpCachePoints[0]);
      rootPoints.push_back(tmpCachePoints.back());
      curveRootPoints.push_back(rootPoints);

      tmpCachePoints.clear();
      tmpCachePoints.insert(tmpCachePoints.begin(), cachedPoints.begin() + cachedPoints.size() / 2 + 1,
                            cachedPoints.end());
      curveCachedPoints.push_back(tmpCachePoints);
      rootPoints.clear();
      rootPoints.push_back(tmpCachePoints[0]);
      rootPoints.push_back(tmpCachePoints.back());
      curveRootPoints.push_back(rootPoints);

      // Find the distance cost is the max size of the bounding box
      const int2 p0 = curveCachedPoints[0][curveCachedPoints[0].size() / 2];
      const int2 p1 = curveCachedPoints[1][curveCachedPoints[1].size() / 2];
      distanceCost = int(std::sqrt(float(p0.x - p1.x) * (p0.x - p1.x) + (p0.y - p1.y) * (p0.y - p1.y)) * 0.3f);
    }
  } else {
    if (cachedPoints.size() >= 2 && rootPoints.size() >= 2) {
      // Pick the first and last as the root points of the curve
      rootPoints[1] = rootPoints.back();
      rootPoints.resize(2);
      curveRootPoints.push_back(rootPoints);
      curveCachedPoints.push_back(cachedPoints);
    }
  }
  return true;
}

Status SeamFinder::findIntersectingSegments(const unsigned char borderIndex) {
  const int2 size = make_int2((int)rect.getWidth(), (int)rect.getHeight());
  const int2 offset = make_int2((int)rect.left(), (int)rect.top());

  // Prepare the initial border buffer
  PROPAGATE_FAILURE_STATUS(findBordersBuffer(rect, inputsMapBuffer.borrow_const(), workMask.borrow(), 4));
  std::vector<unsigned char> borders(rect.getArea());
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking(&borders[0], workMask.borrow_const()));

  // Find the inputsMap
  std::vector<unsigned char> inputsMap(inputsMapBuffer.borrow().numElements());
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking(&inputsMap[0], inputsMapBuffer.borrow()));

  // Only consider borders at the overlapping area
  for (int i = 0; i < size.x; i++)
    for (int j = 0; j < size.y; j++)
      if (inputsMap[j * size.x + i] != 3) {
        borders[j * size.x + i] = 0;
      }

#ifdef SEAMFINDER_DEBUG
  {
    const int2 size = make_int2((int)rect.getWidth(), (int)rect.getHeight());
    std::stringstream ss;
    ss.str("");
    ss << "border-overlapping.png";
    Debug::dumpRGBAIndexDeviceBuffer<unsigned char>(ss.str().c_str(), borders, size.x, size.y);
  }
#endif

  std::vector<bool> visited;
  visited.assign(size.x * size.y, false);
  startPoints.clear();
  endPoints.clear();
  for (int i = 0; i < size.x; i++) {
    for (int j = 0; j < size.y; j++) {
      // Find overlapping point of both borders
      if (((borders[j * size.x + i] & borderIndex) > 0) && (!visited[j * size.x + i])) {
        std::vector<std::vector<int2>> curveCachedPoints;
        std::vector<std::vector<int2>> curveRootPoints;
        int distanceCost = 0;
        // Cache all points surrounding current points and have the same bit set
        bool wrapPath = getCurveEnds(wrapWidth, make_int2(i, j), borderIndex, size, borders, visited, curveCachedPoints,
                                     curveRootPoints, distanceCost);
        for (size_t k = 0; k < curveRootPoints.size(); k++) {
          if (curveRootPoints[k].size() >= 2) {
            for (size_t t = 0; t < curveRootPoints[k].size() / 2; t++) {
              startPoints.push_back(
                  make_int2(curveRootPoints[k][2 * t].x + offset.x, curveRootPoints[k][2 * t].y + offset.y));
              endPoints.push_back(
                  make_int2(curveRootPoints[k][2 * t + 1].x + offset.x, curveRootPoints[k][2 * t + 1].y + offset.y));
              all_cachePoints.push_back(curveCachedPoints[k]);
              wrapPaths.push_back(wrapPath);
              distanceCosts.push_back(distanceCost);
            }
          }
        }
      }
    }
  }
  return Status::OK();
}

Status SeamFinder::findConnectedComponentsAfterCuts(const int bufferIndex, std::vector<int>& components) {
  // Note: due to aliasing, there can be many "little" components
  std::vector<unsigned char> inputsMap(inputsMapBuffer.borrow().numElements());
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking(&inputsMap[0], inputsMapBuffer.borrow()));
  const int2 inputsOffset = make_int2((int)this->rect.left(), (int)this->rect.top());
  const int2 inputsSize = make_int2((int)this->rect.getWidth(), (int)this->rect.getHeight());

  const Core::Rect rect = (bufferIndex == 0) ? rect0 : rect1;
  const GPU::Buffer<const uint32_t> inputBuffer = (bufferIndex == 0) ? input0Buffer : input1Buffer;
  std::vector<unsigned char> mask;
  if (!findValidMask(rect, inputBuffer, mask, GPU::Stream::getDefault()).ok()) {
    return Status::OK();
  }

  const int2 offset = make_int2((int)rect.left(), (int)rect.top());
  const int2 size = make_int2((int)rect.getWidth(), (int)rect.getHeight());
  std::vector<bool> edges(size.x * size.y * 4, true);
  for (size_t i = 0; i < all_directions.size(); i++) {
    const int2 source = make_int2(startPoints[i].x - offset.x, startPoints[i].y - offset.y);
    const int2 target = make_int2(endPoints[i].x - offset.x, endPoints[i].y - offset.y);

    // First, set source and target to isolated island
    for (int t = 0; t < 4; t++) {
      edges[t * size.x * size.y + source.y * size.x + source.x] = false;
      const int2 nextSource =
          make_int2((source.x + seam_dir_rows[t] + wrapWidth) % wrapWidth, source.y + seam_dir_cols[t]);
      if (nextSource.x >= 0 && nextSource.x < size.x && nextSource.y >= 0 && nextSource.y < size.y) {
        edges[(3 - t) * size.x * size.y + nextSource.y * size.x + nextSource.x] = false;
      }

      edges[t * size.x * size.y + target.y * size.x + target.x] = false;
      const int2 nextTarget =
          make_int2((target.x + seam_dir_rows[t] + wrapWidth) % wrapWidth, target.y + seam_dir_cols[t]);
      if (nextTarget.x >= 0 && nextTarget.x < size.x && nextTarget.y >= 0 && nextTarget.y < size.y) {
        edges[(3 - t) * size.x * size.y + nextTarget.y * size.x + nextTarget.x] = false;
      }
    }

    const std::vector<unsigned char> directions = all_directions[i];
    // Now find the cut and turn these edges to 0
    int2 u = source;
    for (int t = (int)directions.size() - 1; t >= 0; t--) {
      const unsigned char direction = directions[t];
      switch (direction) {
        case 0: {
          if (u.x != 0) {  // If u.x == 0, should break no edge
            int2 v0 = make_int2((u.x - 1 + wrapWidth) % wrapWidth, u.y - 1);
            if (v0.x >= 0 && v0.y >= 0 && v0.x < size.x && v0.y < size.y) {
              edges[2 * size.x * size.y + v0.y * size.x + v0.x] = false;
            }
            int2 v1 = make_int2((u.x - 1 + wrapWidth) % wrapWidth, u.y);
            if (v1.x >= 0 && v1.y >= 0 && v1.x < size.x && v1.y < size.y) {
              edges[1 * size.x * size.y + v1.y * size.x + v1.x] = false;
            }
          }
        } break;
        case 1: {
          if (u.y != 0) {
            int2 v0 = make_int2((u.x - 1 + wrapWidth) % wrapWidth, u.y - 1);
            if (v0.x >= 0 && v0.y >= 0 && v0.x < size.x && v0.y < size.y) {
              edges[3 * size.x * size.y + v0.y * size.x + v0.x] = false;
            }
            int2 v1 = make_int2(
                u.x % wrapWidth,
                u.y - 1);  // u.x % wrapWidth : deal with case when u.x = wrapWidth --> ib\src\test\data\seam\test9
            if (v1.x >= 0 && v1.y >= 0 && v1.x < size.x && v1.y < size.y) {
              edges[0 * size.x * size.y + v1.y * size.x + v1.x] = false;
            }
          }
        } break;
        case 2: {
          if (u.y != size.y) {
            int2 v0 = make_int2((u.x - 1 + wrapWidth) % wrapWidth, u.y);
            if (v0.x >= 0 && v0.y >= 0 && v0.x < size.x && v0.y < size.y) {
              edges[3 * size.x * size.y + v0.y * size.x + v0.x] = false;
            }
            int2 v1 = make_int2(
                u.x % wrapWidth,
                u.y);  // u.x % wrapWidth : deal with case when u.x = wrapWidth --> ib\src\test\data\seam\test9
            if (v1.x >= 0 && v1.y >= 0 && v1.x < size.x && v1.y < size.y) {
              edges[0 * size.x * size.y + v1.y * size.x + v1.x] = false;
            }
          }
        } break;
        case 3: {
          if (u.x != size.x) {
            int2 v0 = make_int2(u.x, u.y - 1);
            if (v0.x >= 0 && v0.y >= 0 && v0.x < size.x && v0.y < size.y) {
              edges[2 * size.x * size.y + v0.y * size.x + v0.x] = false;
            }
            int2 v1 = make_int2(u.x, u.y);
            if (v1.x >= 0 && v1.y >= 0 && v1.x < size.x && v1.y < size.y) {
              edges[1 * size.x * size.y + v1.y * size.x + v1.x] = false;
            }
          }
        } break;
        default:
          break;
      }
      int2 v = make_int2((u.x + seam_dir_rows[direction] + (wrapWidth + 1)) % (wrapWidth + 1),
                         u.y + seam_dir_cols[direction]);
      u = v;
    }
  }

  // @NOTE: Component are number from 1 onward
  components.assign(size.x * size.y, -1);
  // Find the connect components
  // The 0 component is for invalid pixel
  for (size_t i = 0; i < mask.size(); i++) {
    if (mask[i] == 0) {
      components[i] = 0;
    }
  }

  int componentIndex = 1;
  for (int i = 0; i < size.x; i++)
    for (int j = 0; j < size.y; j++) {
      if (components[j * size.x + i] < 0) {
        findDisconnectedComponent(wrapWidth, componentIndex, make_int2(i, j), offset, size, edges, inputsOffset,
                                  inputsSize, inputsMap, components);
        componentIndex++;
      }
    }
  return Status::OK();
}

Status SeamFinder::findOutputsMap(const int bufferIndex, const std::vector<int>& components) {
  std::vector<unsigned char> outputsMap;
  std::vector<unsigned char> inputsMap(inputsMapBuffer.borrow().numElements());
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking(&inputsMap[0], inputsMapBuffer.borrow()));

  const int2 size = make_int2((int)rect.getWidth(), (int)rect.getHeight());
  const int2 offset = make_int2((int)rect.left(), (int)rect.top());

  std::vector<bool> cachedBorder(size.x * size.y, false);
  for (size_t i = 0; i < all_cachePoints.size(); i++) {
    for (size_t j = 0; j < all_cachePoints[i].size(); j++) {
      cachedBorder[all_cachePoints[i][j].y * size.x + all_cachePoints[i][j].x] = true;
    }
  }

  const int2 componentSize = (bufferIndex == 0) ? make_int2((int)rect0.getWidth(), (int)rect0.getHeight())
                                                : make_int2((int)rect1.getWidth(), (int)rect1.getHeight());
  const int2 offsetComponent = (bufferIndex == 0) ? make_int2((int)rect0.left(), (int)rect0.top())
                                                  : make_int2((int)rect1.left(), (int)rect1.top());

  int maxComponentValue = -1;
  for (int i = 0; i < componentSize.x; i++)
    for (int j = 0; j < componentSize.y; j++) {
      if (components[j * componentSize.x + i] > maxComponentValue) {
        maxComponentValue = components[j * componentSize.x + i];
      }
    }

  // Count the number of border pixel of a component, if it is larger than 1, it will belong to id1
  std::vector<int> componentAreas(maxComponentValue + 1, 0);
  std::vector<int> countBorderComponents(maxComponentValue + 1, 0);
  std::vector<unsigned char> componentInputIds(maxComponentValue + 1, 0);
  for (int i = 0; i < componentSize.x; i++)
    for (int j = 0; j < componentSize.y; j++) {
      if (components[j * componentSize.x + i] > 0) {
        componentAreas[components[j * componentSize.x + i]]++;
        const int oriI = i + offsetComponent.x - offset.x;
        const int oriJ = j + offsetComponent.y - offset.y;
        if (oriI >= 0 && oriI < size.x && oriJ >= 0 && oriJ < size.y) {
          componentInputIds[(int)components[j * componentSize.x + i]] = inputsMap[oriJ * size.x + oriI];
          if (inputsMap[oriJ * size.x + oriI] == uint32_t((1 << id0) + (1 << id1))) {
            if (cachedBorder[j * componentSize.x + i]) {
              countBorderComponents[(int)components[j * componentSize.x + i]]++;
            }
          }
        }
      }
    }

  // Now deciding the id of every component
  for (size_t i = 0; i < componentInputIds.size(); i++) {
    unsigned char id = componentInputIds[i];
    if (id == (uint32_t)((1 << id0) + (1 << id1))) {
      if (countBorderComponents[i] > 5) {
        componentInputIds[i] = static_cast<unsigned char>((1 << id0) + (1 << id1) - (1 << bufferIndex));
      }
    }
  }

  // Now re-assign the map
  outputsMap.clear();
  for (int j = 0; j < size.y; j++)
    for (int i = 0; i < size.x; i++) {
      unsigned char id = inputsMap[j * size.x + i];
      if (id == ((1 << id0) + (1 << id1))) {
        const int componentI = i + offset.x - offsetComponent.x;
        const int componentJ = j + offset.y - offsetComponent.y;
        if (componentI >= 0 && componentI < componentSize.x && componentJ >= 0 && componentJ < componentSize.y) {
          int component = components[componentJ * componentSize.x + componentI];
          if (component > 0) {
            id = componentInputIds[component];
          }
        }
      }
      outputsMap.push_back(id);
    }

  // Turn all the overlapping areas into 1
  for (size_t i = 0; i < outputsMap.size(); i++) {
    if (outputsMap[i] == ((1 << id0) + (1 << id1))) {
      outputsMap[i] = static_cast<unsigned char>(1 << bufferIndex);
    }
  }
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking(outputsMapBuffer.borrow(), &outputsMap[0], outputsMap.size()));

  return Status::OK();
}

Status SeamFinder::saveSeamToBuffer(const int bufferIndex, std::vector<unsigned char>& data) {
  if (bufferIndex == 2) {
    PROPAGATE_FAILURE_STATUS(blendImages(outputBuffer.borrow()));
  }
  const Core::Rect rect = (bufferIndex == 0) ? rect0 : (bufferIndex == 1 ? rect1 : this->rect);
  const GPU::Buffer<const uint32_t> inputBuffer =
      (bufferIndex == 0) ? input0Buffer : (bufferIndex == 1 ? input1Buffer : outputBuffer.borrow_const());

  const int2 offset = make_int2((int)rect.left(), (int)rect.top());
  const int2 size = make_int2((int)rect.getWidth(), (int)rect.getHeight());
  data.resize(rect.getArea() * 4, 0);
  PROPAGATE_FAILURE_STATUS(GPU::memcpyBlocking<uint32_t>(reinterpret_cast<uint32_t*>(&data[0]), inputBuffer));
  for (size_t i = 0; i < all_directions.size(); i++) {
    const int2 source = make_int2(startPoints[i].x - offset.x, startPoints[i].y - offset.y);
    int2 u = source;
    const std::vector<unsigned char> directions = all_directions[i];
    for (int t = (int)directions.size() - 1; t >= 0; t--) {
      const unsigned char direction = directions[t];
      if ((u.x % wrapWidth < size.x) && (u.y < size.y) && (u.x % wrapWidth >= 0) && (u.y >= 0)) {
        data[4 * ((u.y) * size.x + (u.x % wrapWidth))] = 255;
        data[4 * ((u.y) * size.x + (u.x % wrapWidth)) + 1] =
            (unsigned char)(255.0 * float(t) / float(directions.size() - 1));
        data[4 * ((u.y) * size.x + (u.x % wrapWidth)) + 2] = 0;
        data[4 * ((u.y) * size.x + (u.x % wrapWidth)) + 3] = 255;
      }
      int2 v = make_int2((u.x + seam_dir_rows[direction] + (wrapWidth + 1)) % (wrapWidth + 1),
                         u.y + seam_dir_cols[direction]);
      u = v;
    }
    if ((u.x % wrapWidth < size.x) && (u.y < size.y) && (u.x >= 0) && (u.y >= 0)) {
      data[4 * ((u.y) * size.x + (u.x % wrapWidth))] = 255;
      data[4 * ((u.y) * size.x + (u.x % wrapWidth)) + 1] = 0;
      data[4 * ((u.y) * size.x + (u.x % wrapWidth)) + 2] = 0;
      data[4 * ((u.y) * size.x + (u.x % wrapWidth)) + 3] = 255;
    }
  }

  for (size_t i = 0; i < startPoints.size(); i++) {
    data[4 * ((startPoints[i].y) * size.x + (startPoints[i].x))] = 0;
    data[4 * ((startPoints[i].y) * size.x + (startPoints[i].x)) + 1] = 255;
    data[4 * ((startPoints[i].y) * size.x + (startPoints[i].x)) + 2] = 0;
    data[4 * ((startPoints[i].y) * size.x + (startPoints[i].x)) + 3] = 255;

    data[4 * ((endPoints[i].y) * size.x + (endPoints[i].x))] = 0;
    data[4 * ((endPoints[i].y) * size.x + (endPoints[i].x)) + 1] = 0;
    data[4 * ((endPoints[i].y) * size.x + (endPoints[i].x)) + 2] = 255;
    data[4 * ((endPoints[i].y) * size.x + (endPoints[i].x)) + 3] = 255;
  }
  return Status::OK();
}

Status SeamFinder::saveSeamImage(const std::string& filename, const int bufferIndex) {
  std::vector<unsigned char> data;
  PROPAGATE_FAILURE_STATUS(saveSeamToBuffer(bufferIndex, data));
  const Core::Rect rect = (bufferIndex == 0) ? rect0 : (bufferIndex == 1 ? rect1 : this->rect);
  Util::PngReader writer;
  if (!writer.writeRGBAToFile(filename.c_str(), rect.getWidth(), rect.getHeight(), &data.front())) {
    return {Origin::BlendingMaskAlgorithm, ErrType::RuntimeError, "Cannot write seam PNG image"};
  }
  return Status::OK();
}

Status SeamFinder::replaceBuffers(GPU::Buffer<const uint32_t> newBuffer0, GPU::Buffer<const uint32_t> newBuffer1) {
  if (newBuffer0.numElements() != input0Buffer.numElements() ||
      newBuffer1.numElements() != input1Buffer.numElements()) {
    return {Origin::BlendingMaskAlgorithm, ErrType::ImplementationError,
            "Cannot replace seam finder buffers of different sizes"};
  }
  input0Buffer = newBuffer0;
  input1Buffer = newBuffer1;
  return Status::OK();
}

}  // namespace MergerMask
}  // namespace VideoStitch
