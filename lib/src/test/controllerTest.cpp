// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"
#include "libvideostitch/controller.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/status.hpp"
#include <parallax/noFlow.hpp>
#include <parallax/noWarper.hpp>
#include "libvideostitch/parse.hpp"
#include "libvideostitch/output.hpp"
#include "common/fakeReader.hpp"
#include "common/fakeMerger.hpp"
#include "common/fakeWriter.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/output.hpp"

#include <cassert>
#include <iostream>
#include <memory>

namespace VideoStitch {
namespace Testing {

Core::PanoDefinition* getTestPanoDef() {
  Potential<Ptv::Parser> parser(Ptv::Parser::create());
  if (!parser->parseData("{"
                         " \"width\": 513, "
                         " \"height\": 315, "
                         " \"hfov\": 90.0, "
                         " \"proj\": \"rectilinear\", "
                         " \"inputs\": [ "
                         "  { "
                         "   \"width\": 17, "
                         "   \"height\": 13, "
                         "   \"hfov\": 90.0, "
                         "   \"yaw\": 0.0, "
                         "   \"pitch\": 0.0, "
                         "   \"roll\": 0.0, "
                         "   \"proj\": \"rectilinear\", "
                         "   \"viewpoint_model\": \"ptgui\", "
                         "   \"response\": \"linear\", "
                         "   \"filename\": \"\" "
                         "  }, "
                         "  { "
                         "   \"width\": 17, "
                         "   \"height\": 13, "
                         "   \"hfov\": 90.0, "
                         "   \"yaw\": 0.0, "
                         "   \"pitch\": 0.0, "
                         "   \"roll\": 0.0, "
                         "   \"proj\": \"rectilinear\", "
                         "   \"viewpoint_model\": \"ptgui\", "
                         "   \"response\": \"linear\", "
                         "   \"filename\": \"\" "
                         "  } "
                         " ]"
                         "}")) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false, "could not parse");
    return NULL;
  }
  std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(parser->getRoot()));
  ENSURE((bool)panoDef);
  return panoDef.release();
}

void testNotInteractive() {
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef());
  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef(Core::AudioPipeDefinition::createDefault());

  FakeReaderFactory* fakeReaderFactory = new FakeReaderFactory(0);
  std::atomic<int> totalNumSetups(0);
  FakeImageMergerFactory factory1(Core::ImageMergerFactory::CoreVersion1, &totalNumSetups);
  FakeImageMergerFactory factory2(Core::ImageMergerFactory::CoreVersion1, &totalNumSetups);

  factory1.setHash("hash1");
  factory2.setHash("hash2");

  ENSURE_EQ(0, (int)totalNumSetups);
  Core::PotentialController controller = Core::createController(
      *panoDef, factory1, Core::NoWarper::Factory(), Core::NoFlow::Factory(), fakeReaderFactory, *audioPipeDef);
  ENSURE_EQ(0, (int)totalNumSetups);

  ENSURE(controller.status());

  // Resetting merger factory without stitchers
  ENSURE(controller->resetMergerFactory(factory1, false));
  ENSURE_EQ(0, (int)totalNumSetups);  // Same factory
  ENSURE(controller->resetMergerFactory(factory2, false));
  ENSURE_EQ(0, (int)totalNumSetups);  // Different factory, setup not forced.
  ENSURE(controller->resetMergerFactory(factory1, true));
  ENSURE_EQ(0, (int)totalNumSetups);  // Different factory, setup forced, but no stitchers.

  {
    ENSURE(controller->createStitcher());
    ENSURE_EQ((int)panoDef->numInputs(), (int)totalNumSetups);  // Initial setup.

    // Resetting merger factory with stitchers
    ENSURE(controller->resetMergerFactory(factory1, false));
    ENSURE_EQ((int)panoDef->numInputs(), (int)totalNumSetups);  // Same factory

    ENSURE(controller->resetMergerFactory(factory2, false));
    ENSURE_EQ((int)panoDef->numInputs(), (int)totalNumSetups);  // Different factory, setup not forced.

    ENSURE(controller->resetMergerFactory(factory1, true));
    ENSURE_EQ(2 * (int)panoDef->numInputs(), (int)totalNumSetups);  // Different factory, setup forced.
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::ENSURE(VideoStitch::GPU::setDefaultBackendDevice(0));
  VideoStitch::Testing::testNotInteractive();

  return 0;
}
