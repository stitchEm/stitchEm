// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "dijkstraShortestPath.hpp"

#include <queue>
#include <limits>

namespace VideoStitch {
namespace MergerMask {

// iPair ==>  Integer Pair
typedef std::pair<float, int2> iPair;

struct compare {
  bool operator()(const iPair& l, const iPair& r) { return l.first > r.first; }
};

// http://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-using-priority_queue-stl/
Status DijkstraShortestPath::find(const int wrapWidth, const Core::Rect& rect, const std::vector<float>& buffer,
                                  const int2& source, const int2& target, std::vector<unsigned char>& directions,
                                  const bool wrapPath) {
  const int2 offset = make_int2((int)rect.left(), (int)rect.top());
  const int2 size = make_int2((int)rect.getWidth(), (int)rect.getHeight());

  directions.clear();
  if (source.x < offset.x || source.y < offset.y || target.x < offset.x || target.y < offset.y) {
    return Status::OK();
  }

  int2 u = make_int2(source.x - offset.x, source.y - offset.y);
  int2 uFinal = make_int2(target.x - offset.x, target.y - offset.y);
  std::priority_queue<iPair, std::vector<iPair>, compare> minHeap;
  // Create a vector for distances and initialize all
  // distances as infinite (INF)
  std::vector<float> dist(size.x * size.y, std::numeric_limits<float>::max());
  std::vector<signed char> prev(size.x * size.y, -1);
  // Insert source itself in priority queue and initialize
  // its distance as 0.
  minHeap.push(std::make_pair(0.0f, u));
  dist[u.y * size.x + u.x] = 0;
  while ((u.x != uFinal.x || u.y != uFinal.y) && !minHeap.empty()) {
    // The first vertex in pair is the minimum distance
    // vertex, extract it from priority queue.
    // vertex label is stored in second of pair (it
    // has to be done this way to keep the vertices
    // sorted distance (distance must be first item
    // in pair)
    u = minHeap.top().second;
    float minDist = minHeap.top().first;
    minHeap.pop();
    for (int t = 0; t < 4; t++) {
      int2 v = wrapPath ? make_int2((u.x + seam_dir_rows[t] + wrapWidth) % wrapWidth, u.y + seam_dir_cols[t])
                        : make_int2(u.x + seam_dir_rows[t], u.y + seam_dir_cols[t]);
      if (v.x >= 0 && v.x < size.x && v.y >= 0 && v.y < size.y) {
        float newDist = minDist + buffer[SEAM_DIRECTION * (u.y * size.x + u.x) + t];
        //  If there is shorted path to v through u.
        if (newDist < dist[v.y * size.x + v.x]) {
          // Updating distance of v
          dist[v.y * size.x + v.x] = newDist;
          minHeap.push(std::make_pair(newDist, v));
          prev[v.y * size.x + v.x] = (char)t;
          // @@NOTE: Need to release mem here but not yet to be done
        }
      }
    }
  }

  // Now trace back the direction
  u = uFinal;
  while (prev[u.y * size.x + u.x] >= 0) {
    const signed char t = (signed char)prev[u.y * size.x + u.x];
    directions.push_back(t);
    int2 v = wrapPath ? make_int2((u.x - seam_dir_rows[t] + wrapWidth) % wrapWidth, u.y - seam_dir_cols[t])
                      : make_int2(u.x - seam_dir_rows[t], u.y - seam_dir_cols[t]);
    u = v;
  }
  return Status::OK();
}

}  // namespace MergerMask
}  // namespace VideoStitch
