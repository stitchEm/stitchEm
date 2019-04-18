// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <mask/dijkstraShortestPath.hpp>
#include "../util/pngutil.hpp"

#include <sstream>
#include "gpu/testing.hpp"

namespace VideoStitch {
namespace Testing {

void testDijkstra() {
  MergerMask::DijkstraShortestPath dijkstraShortestPath;
  std::vector<float> costsBuffer = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1};

  std::vector<unsigned char> target_directions = {3, 3, 2, 0, 2, 0, 2, 0};
  std::vector<unsigned char> directions;
  dijkstraShortestPath.find(4, Core::Rect(0, 0, 3, 3), costsBuffer, make_int2(0, 0), make_int2(3, 3), directions);
  ENSURE_EQ(directions.size(), target_directions.size());
  for (size_t i = 0; i < directions.size(); i++) {
    ENSURE_EQ(directions[i], target_directions[i]);
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testDijkstra();

  return 0;
}
