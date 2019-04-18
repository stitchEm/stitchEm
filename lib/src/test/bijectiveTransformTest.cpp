// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "common/ptv.hpp"

#include "backend/common/core/types.hpp"

#include <core/geoTransform.hpp>

#include <string>
#include <sstream>
#include <algorithm>

/*
 * This test could be used to debug the precomputed coordinate buffers in the Transform class
 */

//#define DUMP_TEST_RESULT

#if defined(DUMP_TEST_RESULT)
//#undef NDEBUG
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include <util/debugUtils.hpp>
#endif

namespace VideoStitch {
namespace Testing {

float wrapDifference(const float x0, const float x1, const size_t wrapWidth) {
  const float z0 = (float)std::fmod(x0 + wrapWidth / 2 + 10 * wrapWidth, wrapWidth);
  const float z1 = (float)std::fmod(x1 + wrapWidth / 2 + 10 * wrapWidth, wrapWidth);
  const float y0 = std::min<float>(z0, z1);
  const float y1 = std::max<float>(z0, z1);
  return std::min<float>(std::abs(y0 - y1), std::abs(y0 + wrapWidth - y1));
}

bool testBijectionFromInput(const Core::PanoDefinition& pano, const Core::InputDefinition& im,
                            Core::TopLeftCoords2 inputCoord, const float threshold = 0.5f) {
  std::unique_ptr<Core::TransformStack::GeoTransform> geoTransform(
      Core::TransformStack::GeoTransform::create(pano, im));
  if (geoTransform->isWithinInputBounds(im, inputCoord)) {
    Core::CenterCoords2 uv(inputCoord, Core::TopLeftCoords2(im.getWidth() / 2, im.getHeight() / 2));
    Core::CenterCoords2 uvPanorama = geoTransform->mapInputToPanorama(im, uv, 0);
    if (uvPanorama.x == INVALID_INVERSE_DISTORTION && uvPanorama.y == INVALID_INVERSE_DISTORTION) {
      std::cout << "*** Invalid mapping from input to pano " << inputCoord.x << " " << inputCoord.y << std::endl;
      return true;
    }
    Core::CenterCoords2 uvInput = geoTransform->mapPanoramaToInput(im, uvPanorama, 0);
    bool isBijection =
        (wrapDifference(uvInput.x, uv.x, pano.getWidth()) < threshold) && (std::abs(uvInput.y - uv.y) < threshold);

    if (!isBijection) {
      std::cout << "*** Test failed at input pixel " << inputCoord.x << " " << inputCoord.y << std::endl
                << uvInput.x << "!=" << uv.x << " || " << uvInput.y << "!=" << uv.y << std::endl;
    }
    return isBijection;
  }
  return true;
}

bool testBijectionFromPano(const Core::PanoDefinition& pano, const Core::InputDefinition& im,
                           Core::TopLeftCoords2 panoCoord, const float threshold = 0.5f) {
  std::unique_ptr<Core::TransformStack::GeoTransform> geoTransform(
      Core::TransformStack::GeoTransform::create(pano, im));
  Core::CenterCoords2 uv(panoCoord, Core::TopLeftCoords2(pano.getWidth() / 2, pano.getHeight() / 2));
  Core::CenterCoords2 uvInput = geoTransform->mapPanoramaToInput(im, uv, 0);
  Core::TopLeftCoords2 uvInputTopLeft =
      Core::TopLeftCoords2(uvInput.x + im.getWidth() / 2, uvInput.y + im.getHeight() / 2);
  if (geoTransform->isWithinInputBounds(im, uvInputTopLeft)) {
    Core::CenterCoords2 uvPanorama = geoTransform->mapInputToPanorama(im, uvInput, 0);

    bool isBijection = (wrapDifference(uvPanorama.x, uv.x, pano.getWidth()) < threshold) &&
                       (std::abs(uvPanorama.y - uv.y) < threshold);
    if (!isBijection) {
      wrapDifference(uvPanorama.x, uv.x, pano.getWidth());
      std::cout << "*** Test failed at pano pixel " << panoCoord.x << " " << panoCoord.y << std::endl
                << uvPanorama.x << "!=" << uv.x << " || " << uvPanorama.y << "!=" << uv.y << std::endl;
    }
    return isBijection;
  }
  return true;
}

void testCoordTransform() {
#ifdef DUMP_TEST_RESULT
  std::string workingPath =
      "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/";
#else
  std::string workingPath = "";
#endif
  // workingPath = "C:/Users/Chuong.VideoStitch-09/Documents/GitHub/VideoStitch/VideoStitch-master/lib/src/test/";
  std::vector<std::string> coordTransformTests;
  for (int i = 0; i <= 7; i++) {
    coordTransformTests.push_back(workingPath + "data/ptv/test" + std::to_string(i) + ".ptv");
  }

  const int numTestCountPerTransform = 1000;
  srand(0);

  for (size_t test = 0; test < coordTransformTests.size(); test++) {
    std::string ptvFile = coordTransformTests[test];
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    ENSURE(parser.ok());
    // Load the project and parse it
    ENSURE(parser->parse(ptvFile));
    ENSURE(parser->getRoot().has("pano"));
    // Create a runtime panorama from the parsed project.
    std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(*parser->getRoot().has("pano")));
    int2 size = make_int2((int)panoDef->getWidth(), (int)panoDef->getHeight());
    const std::vector<std::reference_wrapper<Core::InputDefinition>>& inputDefs = panoDef->getVideoInputs();
    for (size_t k = 0; k < inputDefs.size(); k++) {
      const Core::InputDefinition& inputDef = inputDefs[k];
      for (int t = 0; t < numTestCountPerTransform; t++) {
        // To avoid extreme distortion, tests are picked in the range [1..max_size-2]

        const int x = 1 + (rand() % (size.x - 2));
        const int y = 1 + (rand() % (size.y - 2));
        // Test pano --> input --> pano : All mappings from pano to input are valid
        ENSURE(testBijectionFromPano(*panoDef.get(), inputDef, Core::TopLeftCoords2(x, y)));

        const int u = 1 + (rand() % (inputDef.getWidth() - 2));
        const int v = 1 + (rand() % (inputDef.getHeight() - 2));
        // Test input --> pano --> input : Note that not all mapping from input to pano is valid
        ENSURE(testBijectionFromInput(*panoDef.get(), inputDef, Core::TopLeftCoords2(u, v)));
      }
      /*
      for (int i = 1; i < size.x-1; i++) {
        for (int j = 1; j < size.y-1; j++) {
          ENSURE(testBijectionFromPano(*panoDef.get(), inputDef, Core::TopLeftCoords2(i, j)));
        }
      }
      for (int i = 0; i < inputDef.getWidth(); i++) {
        for (int j = 0; j < inputDef.getHeight(); j++) {
          ENSURE(testBijectionFromInput(*panoDef.get(), inputDef, Core::TopLeftCoords2(i, j)));
        }
      }
      */
    }

    std::cout << "*** Test " << test << " passed." << std::endl;
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char** argv) {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testCoordTransform();

  return 0;
}
