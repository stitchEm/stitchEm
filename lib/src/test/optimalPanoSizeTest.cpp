// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "common/ptv.hpp"

#include <array>

namespace VideoStitch {
namespace Testing {

struct PanoSize {
  unsigned width;
  unsigned height;
  float minimumRigSphere;
};

static std::array<PanoSize, 8> expectedPanoSizes{{{3552u, 1776u, 0.f},
                                                  {3168u, 1584u, 0.f},
                                                  {4096u, 2048u, 0.f},
                                                  {4672u, 2336u, 0.f},
                                                  {4736u, 2368u, 0.f},
                                                  {7520u, 3760u, 0.f},
                                                  {4544u, 2272u, 0.0362075f},
                                                  {3360u, 1680u, 1.0f}}};

void testOptimalPanoSize() {
  for (size_t test = 0; test < expectedPanoSizes.size(); test++) {
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    ENSURE(parser.ok());
    ENSURE(parser->parse("data/ptv/test" + std::to_string(test) + ".ptv"));
    ENSURE(parser->getRoot().has("pano"));
    std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(*parser->getRoot().has("pano")));

    unsigned width, height;
    panoDef->computeOptimalPanoSize(width, height);

    ENSURE_EQ(expectedPanoSizes[test].width, width);
    ENSURE_EQ(expectedPanoSizes[test].height, height);

    ENSURE_APPROX_EQ((float)panoDef->computeMinimumRigSphereRadius(), expectedPanoSizes[test].minimumRigSphere, 1e-6f);
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::testOptimalPanoSize();
  return 0;
}
