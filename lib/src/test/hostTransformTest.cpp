// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Host transformation tests

#include "gpu/testing.hpp"
#include "common/ptv.hpp"
#include "gpu/util.hpp"

#include <core/geoTransform.hpp>
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/projections.hpp"

#include <iostream>
#include <memory>

static const int NUM_STEPS = 10;

namespace VideoStitch {
namespace Testing {

Core::PanoDefinition* getPanoDef(const Core::PanoProjection panoProj, const Core::InputDefinition::Format inputProj) {
  std::unique_ptr<Core::PanoDefinition> pano(ensureParsePanoDefinition(R"({
"width" : 1200,
"height" : 600,
"hfov" : 120.0,
"proj" : "equirectangular",
"inputs" : [
{
"width" : 12,
"height" : 10,
"hfov" : 120.0,
"reader_config" : {},
"proj" : "rectilinear",
"yaw" : 1.2,
"pitch" : 3.4,
"roll" : 5.6,
"viewpoint_model" : "ptgui",
"response" : "emor"
}
]
})"));
  pano->setProjection(panoProj);
  switch (panoProj) {
    case Core::PanoProjection::Rectilinear:
      pano->setHFOV(120.0);
      break;
    case Core::PanoProjection::Stereographic:
      pano->setHFOV(320.0);
      break;
    case Core::PanoProjection::EquiangularCubemap:
    case Core::PanoProjection::Equirectangular:
    case Core::PanoProjection::Cylindrical:
    case Core::PanoProjection::FullFrameFisheye:
    case Core::PanoProjection::CircularFisheye:
    case Core::PanoProjection::Cubemap:
      pano->setHFOV(360.0);
      break;
  }

  pano->getInput(0).setFormat(inputProj);
  /*We MUST reset the fov after input format change (as the focal depends on the type of lens) !*/
  VideoStitch::Core::GeometryDefinition def = pano->getInput(0).getGeometries().at(0);
  def.setEstimatedHorizontalFov(pano->getInput(0), 120.0);
  pano->getInput(0).replaceGeometries(new VideoStitch::Core::GeometryDefinitionCurve(def));

  return pano.release();
}

/**
 * Explicit mapping test (for regressions).
 */
void testMapping(const Core::PanoProjection panoProj, const Core::InputDefinition::Format inputProj,
                 const Core::CenterCoords2& panoCoords, const Core::SphericalCoords3& expectedSphericalCoords,
                 const Core::CenterCoords2& expectedInputCoords) {
  const std::unique_ptr<Core::PanoDefinition> pano(getPanoDef(panoProj, inputProj));

  const Core::InputDefinition& inputDef = pano->getInput(0);
  const std::unique_ptr<Core::TransformStack::GeoTransform> geoTransform(
      Core::TransformStack::GeoTransform::create(*pano, inputDef));
  ENSURE(geoTransform.get(), "could not create transform");

  {
    const Core::CenterCoords2 uv = geoTransform->mapPanoramaToInput(inputDef, panoCoords, 0);
    ENSURE_APPROX_EQ(expectedInputCoords.x, uv.x, 0.0005f);
    ENSURE_APPROX_EQ(expectedInputCoords.y, uv.y, 0.0005f);
  }

  {
    const Core::SphericalCoords3 pt =
        geoTransform->mapInputToScaledCameraSphereInRigBase(inputDef, expectedInputCoords, 0);
    ENSURE_APPROX_EQ(expectedSphericalCoords.x, pt.x, 0.0005f);
    ENSURE_APPROX_EQ(expectedSphericalCoords.y, pt.y, 0.0005f);
    ENSURE_APPROX_EQ(expectedSphericalCoords.z, pt.z, 0.0005f);
  }

  {
    const Core::CenterCoords2 uv = geoTransform->mapRigSphericalToInput(inputDef, expectedSphericalCoords, 0);
    ENSURE_APPROX_EQ(expectedInputCoords.x, uv.x, 0.0005f);
    ENSURE_APPROX_EQ(expectedInputCoords.y, uv.y, 0.0005f);
  }
}

void testRoundtrip(const Core::PanoProjection panoProj, const Core::InputDefinition::Format inputProj, int numSteps) {
  const std::unique_ptr<Core::PanoDefinition> pano(getPanoDef(panoProj, inputProj));
  const Core::InputDefinition& inputDef = pano->getInput(0);
  const std::unique_ptr<Core::TransformStack::GeoTransform> geoTransform(
      Core::TransformStack::GeoTransform::create(*pano, inputDef));
  ENSURE(geoTransform.get(), "could not create transform");

  std::cout << "input -> sphere -> input for " << getPanoProjectionName(panoProj) << " + "
            << Core::InputDefinition::getFormatName(inputProj) << std::endl;
  for (int y = -numSteps; y < numSteps; ++y) {
    for (int x = -numSteps; x < numSteps; ++x) {
      const Core::CenterCoords2 inputUv((float)inputDef.getWidth() * (float)x / (float)numSteps,
                                        (float)inputDef.getHeight() * (float)y / (float)numSteps);
      const Core::CenterCoords2 outputUv = geoTransform->mapRigSphericalToInput(
          inputDef, geoTransform->mapInputToScaledCameraSphereInRigBase(inputDef, inputUv, 0), 0);
      ENSURE_APPROX_EQ(inputUv.x, outputUv.x, 0.0001f);
      ENSURE_APPROX_EQ(inputUv.y, outputUv.y, 0.0001f);
    }
  }
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  struct TestProj {
    VideoStitch::Core::PanoProjection::Type pano;
    VideoStitch::Core::InputDefinition::Format input;
    float2 coords;
    float3 expected3d;
    float2 expected2d;
  };

  TestProj tests[] = {
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {0.000000f, 0.000000f},
       {0.000000f, 0.000000f, 1.000000f},
       {-0.052261f, 0.211917f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {50.000000f, 100.000000f},
       {0.224144f, 0.500000f, 0.836516f},
       {1.108500f, 2.250764f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {0.000000f, 0.000000f},
       {0.000000f, -0.000000f, 1.000000f},
       {-0.086432f, 0.350031f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {50.000000f, 100.000000f},
       {0.224144f, 0.500000f, 0.836516f},
       {1.774448f, 3.175040f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {50.000000f, 100.000000f},
       {0.224144f, 0.500000f, 0.836516f},
       {1.586783f, 3.221898f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {50.000000f, 100.000000f},
       {0.224144f, 0.500000f, 0.836516f},
       {1.586783f, 3.221898f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.224144f, 0.500000f, 0.836516f},
       {1.488099f, 3.021524f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::Equirectangular,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.224144f, 0.500000f, 0.836516f},
       {1.488099f, 3.021524f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {0.000000f, 0.000000f},
       {0.000000f, 0.000000f, 1.000000f},
       {-0.052261f, 0.211917f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {50.000000f, 100.000000f},
       {0.137361f, 0.274721f, 0.951662f},
       {0.551691f, 1.175718f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {0.000000f, 0.000000f},
       {0.000000f, -0.000000f, 1.000000f},
       {-0.086432f, 0.350031f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {50.000000f, 100.000000f},
       {0.137361f, 0.274721f, 0.951662f},
       {0.904891f, 1.852995f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {50.000000f, 100.000000f},
       {0.137361f, 0.274721f, 0.951662f},
       {0.873017f, 1.860501f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {50.000000f, 100.000000f},
       {0.137361f, 0.274721f, 0.951662f},
       {0.873017f, 1.860501f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.137361f, 0.274721f, 0.951662f},
       {0.800338f, 1.705613f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::Rectilinear,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.137361f, 0.274721f, 0.951662f},
       {0.800338f, 1.705613f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {0.000000f, 0.000000f},
       {0.000000f, 0.000000f, 1.000000f},
       {-0.052261f, 0.211917f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {5.000000f, 10.0000000f},
       {0.093477f, 0.186954f, 0.977911f},
       {0.3457852f, 0.847127f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {0.000000f, 0.000000f},
       {0.000000f, -0.000000f, 1.000000f},
       {-0.086432f, 0.350031f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {50.000000f, 100.000000f},
       {0.446532f, 0.893065f, -0.055171f},
       {10.050594f, 5.723076f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {50.000000f, 100.000000f},
       {0.446532f, 0.893065f, -0.055171f},
       {5.116043f, 8.083425f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {50.000000f, 100.000000f},
       {0.446532f, 0.893065f, -0.055171f},
       {5.116043f, 8.083425f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.446532f, 0.893065f, -0.055171f},
       {6.136199f, 9.695287f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::Stereographic,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.446532f, 0.893065f, -0.055171f},
       {6.1361989f, 9.695287f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {0.000000f, 0.000000f},
       {0.000000f, 0.000000f, 1.000000f},
       {-0.052261f, 0.211917f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.207266f, 2.221899f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {0.000000f, 0.000000f},
       {0.000000f, -0.000000f, 1.000000f},
       {-0.086432f, 0.350031f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.921383f, 3.120240f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.724863f, 3.174506f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.724863f, 3.174506f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.618255f, 2.978301f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::FullFrameFisheye,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.618255f, 2.978301f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {0.000000f, 0.000000f},
       {0.000000f, 0.000000f, 1.000000f},
       {-0.052261f, 0.211917f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::Rectilinear,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.207266f, 2.221899f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {0.000000f, 0.000000f},
       {0.000000f, -0.000000f, 1.000000f},
       {-0.086432f, 0.350031f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::Equirectangular,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.921383f, 3.120240f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.724863f, 3.174506f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.086324f, 0.350046f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.724863f, 3.174506f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.618255f, 2.978301f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {0.000000f, 0.000000f},
       {-0.000000f, 0.000000f, 1.000000f},
       {-0.078313f, 0.317560f}},
      {VideoStitch::Core::PanoProjection::CircularFisheye,
       VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt,
       {50.000000f, 100.000000f},
       {0.247101f, 0.494201f, 0.833490f},
       {1.618255f, 2.978301f}},
  };

  /* Exact tests for regressions */
  for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
    TestProj tp = tests[i];
    VideoStitch::Testing::testMapping(
        VideoStitch::Core::PanoProjection(tp.pano), tp.input,
        VideoStitch::Core::CenterCoords2(tp.coords.x, tp.coords.y),
        VideoStitch::Core::SphericalCoords3(tp.expected3d.x, tp.expected3d.y, tp.expected3d.z),
        VideoStitch::Core::CenterCoords2(tp.expected2d.x, tp.expected2d.y));
  }

  /* Roundtrip tests for consistency. */
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Equirectangular),
      VideoStitch::Core::InputDefinition::Format::Rectilinear, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Rectilinear),
                                      VideoStitch::Core::InputDefinition::Format::Rectilinear, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::FullFrameFisheye),
      VideoStitch::Core::InputDefinition::Format::Rectilinear, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Stereographic),
      VideoStitch::Core::InputDefinition::Format::Rectilinear, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Equirectangular),
      VideoStitch::Core::InputDefinition::Format::CircularFisheye, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Rectilinear),
                                      VideoStitch::Core::InputDefinition::Format::CircularFisheye, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::FullFrameFisheye),
      VideoStitch::Core::InputDefinition::Format::CircularFisheye, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Stereographic),
      VideoStitch::Core::InputDefinition::Format::CircularFisheye, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Equirectangular),
      VideoStitch::Core::InputDefinition::Format::FullFrameFisheye, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Rectilinear),
                                      VideoStitch::Core::InputDefinition::Format::FullFrameFisheye, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::FullFrameFisheye),
      VideoStitch::Core::InputDefinition::Format::FullFrameFisheye, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Stereographic),
      VideoStitch::Core::InputDefinition::Format::FullFrameFisheye, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Equirectangular),
      VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Rectilinear),
                                      VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::FullFrameFisheye),
      VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Stereographic),
      VideoStitch::Core::InputDefinition::Format::CircularFisheye_Opt, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Equirectangular),
      VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Rectilinear),
                                      VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::FullFrameFisheye),
      VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt, NUM_STEPS);
  VideoStitch::Testing::testRoundtrip(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Stereographic),
      VideoStitch::Core::InputDefinition::Format::FullFrameFisheye_Opt, NUM_STEPS);

  return 0;
}
