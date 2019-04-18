// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/panoDef.hpp"

#include "exposure/metadataProcessor.hpp"

namespace VideoStitch {
namespace Testing {

static const int NUM_INPUTS{4};

std::unique_ptr<Core::PanoDefinition> getTestPanoDef() {
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
  return panoDef;
}

void ENSURE_CURVE_POINTS(const Core::Curve& curve, const std::vector<std::pair<int, double>> points,
                         const char* msg = "") {
  for (const std::pair<int, double>& point : points) {
    ENSURE_EQ(curve.at(point.first), point.second, msg);
  }
}

void testAppendExposure() {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();
  ENSURE(panoDef->numVideoInputs() == NUM_INPUTS);

  FrameRate frameRate{30, 1};

  std::vector<std::pair<int, GPU::Buffer<const uint32_t>>> frames{{0, {}}};

  Exposure::MetadataProcessor mp;

  frameid_t exposureDataFrame{1000};
  mtime_t exposureDataTime = frameRate.frameToTimestamp(exposureDataFrame);
  Metadata::Exposure exposure{exposureDataTime, 800, 0.03f, 0.03f};

  Input::MetadataChunk metadata;
  metadata.exposure.push_back({{0, exposure}});

  auto applyExposureToPanoAtFrame = [&mp, &metadata, &frameRate, &panoDef](frameid_t currentStitchingFrame) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  ENSURE_CURVE_POINTS(panoDef->getExposureValue(), {{0, 0.}, {1, 0.}}, "Test pano exposure should be uninitialized");
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {1, 0.}},
                      "First reader exposure value should be uninitialized");
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(1).getExposureValue(), {{0, 0.}, {1, 0.}},
                      "Second reader exposure value should be uninitialized");

  applyExposureToPanoAtFrame(0);

  // checking global exposure
  {
    // global exposure compensation should be average of inputs at all time
    auto globalExposureComp = exposure.computeEv() / NUM_INPUTS;

    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                          // current frame unchanged
        {exposureDataFrame / 2, globalExposureComp / 2},  // interpolation --> half time at half value
        {exposureDataFrame, globalExposureComp},          // final exposure value that was set
        {exposureDataFrame + 100, globalExposureComp},    // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getExposureValue(), expectedCurve, "Panorama global exposure, see comments");
  }

  // checking exposure for input 0
  {
    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                             // current frame unchanged
        {exposureDataFrame / 2, -exposure.computeEv() / 2},  // interpolation --> half time at half value
        {exposureDataFrame, -exposure.computeEv()},          // final exposure value that was set
        {exposureDataFrame + 100, -exposure.computeEv()},    // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), expectedCurve,
                        "Exposure of input 0, see comments");
  }

  // exposure for input 1 shouldn't have changed
  {
    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},
        {exposureDataFrame / 2, 0.},
        {exposureDataFrame, 0.},
        {exposureDataFrame + 100, 0.},
    };

    ENSURE_CURVE_POINTS(panoDef->getVideoInput(1).getExposureValue(), expectedCurve,
                        "Exposure of input 1 should remain unchanged");
  }

  frameid_t nextExposureDataFrame = 2000;
  mtime_t nextExposureDataTime = frameRate.frameToTimestamp(nextExposureDataFrame);

  Metadata::Exposure nextExposure{nextExposureDataTime, 400, 0.03f, 0.03f};

  metadata.clear();
  metadata.exposure.push_back({{0, nextExposure}});

  applyExposureToPanoAtFrame(0);

  // checking global exposure
  {
    // global exposure compensation should be average of inputs at all time
    auto globalExposureComp = exposure.computeEv() / NUM_INPUTS;
    auto nextGlobalExposureComp = nextExposure.computeEv() / NUM_INPUTS;

    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                                // current frame unchanged
        {exposureDataFrame, globalExposureComp},                // previous exposure value that was set
        {nextExposureDataFrame, nextGlobalExposureComp},        // final exposure value that was set
        {nextExposureDataFrame + 100, nextGlobalExposureComp},  // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getExposureValue(), expectedCurve, "Panorama global exposure, see comments");
  }

  // checking exposure for input 0
  {
    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                                   // current frame unchanged
        {exposureDataFrame, -exposure.computeEv()},                // previous exposure value that was set
        {nextExposureDataFrame, -nextExposure.computeEv()},        // exposure value that was set last
        {nextExposureDataFrame + 100, -nextExposure.computeEv()},  // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), expectedCurve,
                        "Exposure of input 0, see comments");
  }

  // interpolation --> half time at ~half value, exact interpolation value implementation defined by the curve
  ENSURE_APPROX_EQ(panoDef->getVideoInput(0).getExposureValue().at(exposureDataFrame / 2), -2.06899, 0.1,
                   "Regression test: implementation defined curve interpolation value");

  // exposure for input 1 shouldn't have changed
  {
    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},
        {exposureDataFrame / 2, 0.},
        {exposureDataFrame, 0.},
        {exposureDataFrame + 100, 0.},
        {nextExposureDataFrame, 0.},
        {nextExposureDataFrame + 100, 0.},
    };

    ENSURE_CURVE_POINTS(panoDef->getVideoInput(1).getExposureValue(), expectedCurve,
                        "Exposure of input 1 should remain unchanged");
  }
}

void testTryAppendInvalidExposure() {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();
  ENSURE(panoDef->numVideoInputs() == NUM_INPUTS);

  FrameRate frameRate{30, 1};

  std::vector<std::pair<int, GPU::Buffer<const uint32_t>>> frames{{0, {}}};

  Exposure::MetadataProcessor mp;

  frameid_t exposureDataFrame{1000};
  mtime_t exposureDataTime = frameRate.frameToTimestamp(exposureDataFrame);

  // invalid exposure with ISO = 0
  Metadata::Exposure exposure{exposureDataTime, 0, 0.02f, 0.02f};

  Input::MetadataChunk metadata;
  metadata.exposure.push_back({{0, exposure}});

  auto applyExposureToPanoAtFrame = [&mp, &metadata, &frameRate, &panoDef](frameid_t currentStitchingFrame) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  applyExposureToPanoAtFrame(0);

  ENSURE_CURVE_POINTS(panoDef->getExposureValue(), {{0, 0.}, {1, 0.}}, "Test pano exposure should remain at EV 0");
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {1, 0.}},
                      "First reader exposure value should remain at EV 0");
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(1).getExposureValue(), {{0, 0.}, {1, 0.}},
                      "Second reader exposure value should be remain EV 0");

  // metadata for non-existing input, should be ignored, not crash
  metadata.clear();
  metadata.exposure.push_back({{NUM_INPUTS, exposure}});
  applyExposureToPanoAtFrame(0);

  ENSURE_CURVE_POINTS(panoDef->getExposureValue(), {{0, 0.}, {1, 0.}}, "Test pano exposure should remain at EV 0");
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {1, 0.}},
                      "First reader exposure value should remain at EV 0");
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(1).getExposureValue(), {{0, 0.}, {1, 0.}},
                      "Second reader exposure value should be remain EV 0");
}

void testAddMeasurementsForTwoSensors() {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();
  FrameRate frameRate{30, 1};

  std::vector<std::pair<int, GPU::Buffer<const uint32_t>>> frames{{0, {}}};

  Exposure::MetadataProcessor mp;

  frameid_t exposureDataFrame{1000};
  mtime_t exposureDataTime = frameRate.frameToTimestamp(exposureDataFrame);
  Metadata::Exposure exposure_0{exposureDataTime, 800, 0.03f, 0.03f};
  Metadata::Exposure exposure_1{exposureDataTime, 800, 0.03f, 0.03f};

  Input::MetadataChunk metadata;
  metadata.exposure.push_back({{0, exposure_0}, {1, exposure_1}});

  auto applyExposureToPanoAtFrame = [&mp, &metadata, &frameRate, &panoDef](frameid_t currentStitchingFrame) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  ENSURE_CURVE_POINTS(panoDef->getExposureValue(), {{0, 0.}, {1, 0.}}, "Test pano exposure should be uninitialized");
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {1, 0.}},
                      "First reader exposure value should be uninitialized");
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(1).getExposureValue(), {{0, 0.}, {1, 0.}},
                      "Second reader exposure value should be uninitialized");

  applyExposureToPanoAtFrame(0);

  // checking global exposure
  {
    // global exposure compensation should be average of inputs at all time
    auto globalExposureComp = (exposure_0.computeEv() + exposure_1.computeEv()) / NUM_INPUTS;

    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                          // current frame unchanged
        {exposureDataFrame / 2, globalExposureComp / 2},  // interpolation --> half time at half value
        {exposureDataFrame, globalExposureComp},          // final exposure value that was set
        {exposureDataFrame + 100, globalExposureComp},    // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getExposureValue(), expectedCurve, "Panorama global exposure, see comments");
  }

  // checking exposure for input 0
  {
    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                               // current frame unchanged
        {exposureDataFrame / 2, -exposure_0.computeEv() / 2},  // interpolation --> half time at half value
        {exposureDataFrame, -exposure_0.computeEv()},          // final exposure value that was set
        {exposureDataFrame + 100, -exposure_0.computeEv()},    // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), expectedCurve,
                        "Exposure of input 0, see comments");
  }

  // checking exposure for input 0
  {
    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                               // current frame unchanged
        {exposureDataFrame / 2, -exposure_1.computeEv() / 2},  // interpolation --> half time at half value
        {exposureDataFrame, -exposure_1.computeEv()},          // final exposure value that was set
        {exposureDataFrame + 100, -exposure_1.computeEv()},    // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), expectedCurve,
                        "Exposure of input 1, see comments");
  }

  frameid_t nextExposureDataFrame = 2000;
  mtime_t nextExposureDataTime = frameRate.frameToTimestamp(nextExposureDataFrame);

  Metadata::Exposure nextExposure_0{nextExposureDataTime, 200, 0.03f, 0.03f};
  Metadata::Exposure nextExposure_1{nextExposureDataTime, 800, 0.03f, 0.03f};

  metadata.clear();
  metadata.exposure.push_back({{0, nextExposure_0}, {1, nextExposure_1}});

  applyExposureToPanoAtFrame(0);

  // checking global exposure
  {
    // global exposure compensation should be average of inputs at all time
    auto globalExposureComp = (exposure_0.computeEv() + exposure_1.computeEv()) / NUM_INPUTS;
    auto nextGlobalExposureComp = (nextExposure_0.computeEv() + nextExposure_1.computeEv()) / NUM_INPUTS;

    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                                // current frame unchanged
        {exposureDataFrame, globalExposureComp},                // previous exposure value that was set
        {nextExposureDataFrame, nextGlobalExposureComp},        // final exposure value that was set
        {nextExposureDataFrame + 100, nextGlobalExposureComp},  // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getExposureValue(), expectedCurve, "Panorama global exposure, see comments");
  }

  // checking exposure for input 0
  {
    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                                     // current frame unchanged
        {exposureDataFrame, -exposure_0.computeEv()},                // previous exposure value that was set
        {nextExposureDataFrame, -nextExposure_0.computeEv()},        // exposure value that was set last
        {nextExposureDataFrame + 100, -nextExposure_0.computeEv()},  // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), expectedCurve,
                        "Exposure of input 0, see comments");
  }

  // interpolation --> half time at ~half value, exact interpolation value implementation defined by the curve
  ENSURE_APPROX_EQ(panoDef->getVideoInput(0).getExposureValue().at(exposureDataFrame / 2), -2.13149, 0.1,
                   "Regression test: implementation defined curve interpolation value");

  // checking exposure for input 1
  {
    std::vector<std::pair<int, double>> expectedCurve = {
        {0, 0.},                                                     // current frame unchanged
        {exposureDataFrame, -exposure_1.computeEv()},                // previous exposure value that was set
        {nextExposureDataFrame, -nextExposure_1.computeEv()},        // exposure value that was set last
        {nextExposureDataFrame + 100, -nextExposure_1.computeEv()},  // final exposure value that was set continues
    };

    ENSURE_CURVE_POINTS(panoDef->getVideoInput(1).getExposureValue(), expectedCurve,
                        "Exposure of input 1, see comments");
  }
}

void testNoJumps() {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();
  FrameRate frameRate{30, 1};

  std::vector<std::pair<int, GPU::Buffer<const uint32_t>>> frames{{0, {}}};

  Exposure::MetadataProcessor mp;

  frameid_t exposureDataFrame{1000};
  mtime_t exposureDataTime = frameRate.frameToTimestamp(exposureDataFrame);
  Metadata::Exposure exposure{exposureDataTime, 800, 0.02f, 0.02f};

  Input::MetadataChunk metadata;
  metadata.exposure.push_back({{0, exposure}});

  auto applyExposureToPanoAtFrame = [&mp, &metadata, &frameRate, &panoDef](frameid_t currentStitchingFrame) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  // covered by other tests
  applyExposureToPanoAtFrame(0);

  std::unique_ptr<Core::PanoDefinition> setupPanoDef(panoDef->clone());

  for (frameid_t nextExposureDataFrame : {1000, 1001, 1010, 1500, 2000}) {
    for (int currentStitcherFrame :
         {500, 990, 999, 1000, 1001, 1010, 1500, 1950, 1990, 1995, 1999, 2000, 2001, 2005, 2010, 2020}) {
      panoDef.reset(setupPanoDef->clone());

      mtime_t nextExposureDataTime = frameRate.frameToTimestamp(nextExposureDataFrame);

      Metadata::Exposure nextExposure{nextExposureDataTime, 400, 0.02f, 0.02f};

      metadata.clear();
      metadata.exposure.push_back({{0, nextExposure}});

      applyExposureToPanoAtFrame(currentStitcherFrame);

      // checking global exposure
      {
        auto globalExposureComp = setupPanoDef->getExposureValue().at(currentStitcherFrame);
        auto nextGlobalExposureComp = nextExposure.computeEv() / NUM_INPUTS;
        std::vector<std::pair<int, double>> expectedCurve = {
            {currentStitcherFrame, globalExposureComp},  // the value for the current frame should not have changed
            {2100, nextGlobalExposureComp},              // but it is set some time in the future
        };
        ENSURE_CURVE_POINTS(panoDef->getExposureValue(), expectedCurve, "Panorama global exposure, see comments");
      }

      // checking exposure for input 0
      {
        std::vector<std::pair<int, double>> expectedCurve = {
            {currentStitcherFrame,
             setupPanoDef->getVideoInput(0).getExposureValue().at(
                 currentStitcherFrame)},        // the value for the current frame should not have changed
            {2100, -nextExposure.computeEv()},  // but it is set some time in the future
        };
        ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), expectedCurve,
                            "Exposure of input 0, see comments");
      }

      // exposure for input 1 shouldn't have changed
      {
        std::vector<std::pair<int, double>> expectedCurve = {
            {0, 0.},
            {exposureDataFrame / 2, 0.},
            {exposureDataFrame, 0.},
            {exposureDataFrame + 100, 0.},
            {nextExposureDataFrame, 0.},
            {nextExposureDataFrame + 100, 0.},
        };

        ENSURE_CURVE_POINTS(panoDef->getVideoInput(1).getExposureValue(), expectedCurve,
                            "Exposure of input 1 should remain unchanged");
      }
    }
  }
}

void testPruning() {
  Exposure::MetadataProcessor mp;

  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();

  FrameRate frameRate{30, 1};
  Input::MetadataChunk metadata;
  Metadata::Exposure exposure{1000, 800, 0.02f, 0.02f};

  // some fake exposure data for camera 1
  // while we do pruning tests on curves for camera 0
  metadata.exposure.push_back({{1, exposure}});

  Core::Spline* exposureSpline = Core::Spline::point(0, 0.);
  exposureSpline->cubicTo(10, 100.)->cubicTo(20, 200.)->cubicTo(30, 300.);

  Core::Curve* exposureCurve = new Core::Curve(exposureSpline);
  panoDef->getVideoInput(0).replaceExposureValue(exposureCurve);

  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {10, 100.}, {20, 200.}, {30, 300.}},
                      "Curve should have been set");

  auto applyExposureToPanoAtFrame = [&mp, &metadata, &frameRate, &panoDef](frameid_t currentStitchingFrame) {
    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  };

  // update pano at frame 0, no pruning
  applyExposureToPanoAtFrame(0);
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {10, 100.}, {20, 200.}, {30, 300.}},
                      "Curve should remain unchanged");

  // update pano at frame 5, no pruning
  applyExposureToPanoAtFrame(5);
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {10, 100.}, {20, 200.}, {30, 300.}},
                      "Curve should remain unchanged");

  // update pano at frame 10, no pruning
  applyExposureToPanoAtFrame(10);

  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {10, 100.}, {20, 200.}, {30, 300.}},
                      "Curve should remain unchanged");

  // update pano at frame 11, prune first point
  applyExposureToPanoAtFrame(11);
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {10, 100.}, {20, 200.}, {30, 300.}},
                      "Curve should remain unchanged");

  // update pano at frame 11, no change
  applyExposureToPanoAtFrame(11);
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 0.}, {10, 100.}, {20, 200.}, {30, 300.}},
                      "Curve should remain unchanged");

  // update pano at frame 29, prune two points
  applyExposureToPanoAtFrame(29);
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 100.}, {10, 100.}, {20, 200.}, {30, 300.}},
                      "First point should have been pruned");

  // update pano at frame 30, no change
  applyExposureToPanoAtFrame(30);
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 100.}, {10, 100.}, {20, 200.}, {30, 300.}},
                      "First point should have been pruned");

  // update pano at frame 31, single point remains
  applyExposureToPanoAtFrame(31);
  ENSURE_CURVE_POINTS(panoDef->getVideoInput(0).getExposureValue(), {{0, 200.}, {10, 200.}, {20, 200.}, {30, 300.}},
                      "First two points should have been pruned");
}

void test_VSA_6423() {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();
  FrameRate frameRate{30, 1};
  Exposure::MetadataProcessor mp;
  Input::MetadataChunk metadata;
  auto inputID = 0;

  {
    frameid_t frame = 10;
    mtime_t exposureDataTime = frameRate.frameToTimestamp(frame);
    Metadata::Exposure exposure{exposureDataTime, (uint16_t)(1), 1.00999f, 1.f};
    metadata.exposure.push_back({{inputID, exposure}});
  }

  {
    frameid_t frame = 20;
    mtime_t exposureDataTime = frameRate.frameToTimestamp(frame);
    Metadata::Exposure exposure{exposureDataTime, (uint16_t)(1), 0.99999f, 1.f};
    metadata.exposure.push_back({{inputID, exposure}});
  }

  // metadata insertion has a useful assertion self-check that compares
  // that curves return the value of newly inserted measurements
  // make sure it's not over-eager, but allows for slight floating point errors
  std::unique_ptr<Core::PanoDefinition> updated = mp.createUpdatedPano(metadata, *panoDef, frameRate, 0);
}

void benchmarkAddingData(int numFrames) {
  std::unique_ptr<Core::PanoDefinition> panoDef = getTestPanoDef();
  FrameRate frameRate{30, 1};

  std::vector<std::pair<int, GPU::Buffer<const uint32_t>>> frames{{0, {}}};

  Exposure::MetadataProcessor mp;

  for (int i = 0; i < numFrames; i++) {
    frameid_t currentStitchingFrame = i;

    frameid_t metadataAhead = i % 50;

    frameid_t exposureDataFrame{currentStitchingFrame + metadataAhead};
    mtime_t exposureDataTime = frameRate.frameToTimestamp(exposureDataFrame);

    Input::MetadataChunk metadata;

    for (int inputID = 0; inputID < NUM_INPUTS; inputID++) {
      Metadata::Exposure exposure{exposureDataTime, (uint16_t)(200 + i % (50 * (inputID + 1))), 0.02f, 0.02f};
      metadata.exposure.push_back({{inputID, exposure}});
    }

    std::unique_ptr<Core::PanoDefinition> updated =
        mp.createUpdatedPano(metadata, *panoDef, frameRate, currentStitchingFrame);
    if (updated) {
      panoDef.reset(updated.release());
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testAppendExposure();

  VideoStitch::Testing::testTryAppendInvalidExposure();

  VideoStitch::Testing::testAddMeasurementsForTwoSensors();

  VideoStitch::Testing::testNoJumps();

  VideoStitch::Testing::testPruning();

  VideoStitch::Testing::test_VSA_6423();

  VideoStitch::Testing::benchmarkAddingData(3);
  return 0;
}
