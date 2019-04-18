// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Transformation tests
#include "gpu/testing.hpp"

#include "backend/cpp/core/transformStack.hpp"

#include "common/ptv.hpp"
#include "core/kernels/photoStack.cu"
#include "core/radial.hpp"
#include "core/photoTransform.hpp"
#include "core/transformGeoParams.hpp"
#include "core/radial.hpp"

#include "libvideostitch/emor.hpp"

#include "math.h"

//#define CHAIN_DEBUG 1

namespace VideoStitch {
namespace Testing {

void testColorCorrectionParams() {
  const double panoEv = -1.2;
  {
    const Core::PhotoTransform::ColorCorrectionParams input(2.1, 0.82, 1.0, 1.3);
    const Core::PhotoTransform::ColorCorrectionParams output =
        Core::PhotoTransform::ColorCorrectionParams::canonicalFromMultiplier(
            input.computeColorMultiplier(panoEv, 2.0, 0.5, 0.7), panoEv, 2.0, 0.5, 0.7);
    ENSURE_APPROX_EQ(input.ev, output.ev, 0.0001);
    ENSURE_APPROX_EQ(input.redCB, output.redCB, 0.0001);
    ENSURE_APPROX_EQ(input.greenCB, output.greenCB, 0.0001);
    ENSURE_APPROX_EQ(input.blueCB, output.blueCB, 0.0001);
  }
  {
    const Core::PhotoTransform::ColorCorrectionParams input(2.1, 0.95, 0.9, 1.11);
    const float3 inputMult = input.computeColorMultiplier(panoEv, 2.0, 0.5, 0.7);
    const float3 outputMult =
        Core::PhotoTransform::ColorCorrectionParams::canonicalFromMultiplier(inputMult, panoEv, 1.0, 1.0, 1.0)
            .computeColorMultiplier(panoEv, 1.0, 1.0, 1.0);
    ENSURE_APPROX_EQ(inputMult.x, outputMult.x, 0.0001f);
    ENSURE_APPROX_EQ(inputMult.y, outputMult.y, 0.0001f);
    ENSURE_APPROX_EQ(inputMult.z, outputMult.z, 0.0001f);
  }
  {
    const Core::PhotoTransform::ColorCorrectionParams input(2.1, 0.95, 0.9, 1.11);
    const float3 inputMult = input.computeColorMultiplier(panoEv, 1.0, 1.0, 1.0);
    const float3 outputMult =
        Core::PhotoTransform::ColorCorrectionParams::canonicalFromMultiplier(inputMult, panoEv, 2.0, 0.5, 0.7)
            .computeColorMultiplier(panoEv, 2.0, 0.5, 0.7);
    ENSURE_APPROX_EQ(inputMult.x, outputMult.x, 0.0001f);
    ENSURE_APPROX_EQ(inputMult.y, outputMult.y, 0.0001f);
    ENSURE_APPROX_EQ(inputMult.z, outputMult.z, 0.0001f);
  }
}

//#define CHAIN_DEBUG

template <Core::Convert2D3DFnT toSphere, Core::Convert3D2DFnT fromSphere>
void testStack(float2 uv, float2 expUv, float sphereDist, float scale) {
  uv /= sphereDist;

  /* From pano to sphere */
  float3 pt = toSphere(uv);

  /* Transform stack */
  VideoStitch::Core::vsfloat3x4 pose = {{{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}}};
  pt = Core::TransformStack::transformSphere(pt, pose);

  /* convert sphere to image-specific space */
  uv = fromSphere(pt);
  uv *= scale;

  ENSURE_APPROX_EQ(expUv.x, uv.x, 0.5f);
  ENSURE_APPROX_EQ(expUv.y, uv.y, 0.5f);
}

template <Core::Convert2D3DFnT toSphere, Core::Convert3D2DFnT fromSphere>
void testRoundtrip(Core::PanoDefinition::Format panoFmt, Core::InputDefinition::Format imFmt) {
  const float sphereDist = Core::TransformGeoParams::computePanoScale(panoFmt, 1000, 90.0f);
  const float scale = Core::TransformGeoParams::computeInputScale(imFmt, 1000, 90.0f);

  for (int x = -4; x < 5; ++x) {
    for (int y = 0; y < 3; ++y) {
      float2 uv;
      uv.x = ((float)x / 5.0f) * sphereDist;
      uv.y = ((float)y / 3.0f) * sphereDist;

      testStack<toSphere, fromSphere>(uv, uv, sphereDist, scale);
    }
  }
}

template <class PhotoCorrT>
void testPhotoStack(Core::TransformPhotoParam photoParam, bool plot, float eps) {
  for (int i = 0; i < 256; ++i) {
    const float f = (float)i;
    if (plot) {
      std::cout << f << "\t";
    }
    float3 color = make_float3(f, f, f);
    color = PhotoCorrT::corr(color, photoParam.floatParam, (float*)photoParam.transformData);
    if (plot) {
      std::cout << color.x << "\t";
    }
    color = PhotoCorrT::invCorr(color, photoParam.floatParam, (float*)photoParam.transformData);
    if (plot) {
      std::cout << color.x << std::endl;
    }
    ENSURE_APPROX_EQ(color.x, f, eps);
    ENSURE_APPROX_EQ(color.y, f, eps);
    ENSURE_APPROX_EQ(color.z, f, eps);
  }
}

void testRadialCommon(const float2 refUv, VideoStitch::Core::GeometryDefinition& geometry,
                      VideoStitch::Core::InputDefinition& im) {
  float p0, p1, p2, p3, p4;

  computeRadialParams(im, geometry, p0, p1, p2, p3, p4);

  float2 uv = refUv;

  VideoStitch::Core::vsDistortion distortion = {{p0, p1, p2, p3, p4}};
  Core::TransformStack::distortionScaled(uv, distortion);
  if ((float)fabs(uv.x) > 1.0f) {
    return;
  }

  Core::TransformStack::inverseDistortionScaled(uv, distortion);

  ENSURE_APPROX_EQ(refUv.x, uv.x, std::abs(refUv.x / 100.f));
  ENSURE_APPROX_EQ(refUv.y, uv.y, std::abs(refUv.y / 100.f));
}

void testRadial() {
  std::string mergedConfig =
      "{"
      " \"width\": 20,"
      " \"height\": 10,"
      " \"viewpoint_model\": \"ptgui\","
      " \"response\": \"gamma\","
      " \"reader_config\": {},"       // Dummy
      " \"hfov\": 123.4,"             // Dummy
      " \"yaw\": 0.0,"                // Dummy
      " \"pitch\": 0.0,"              // Dummy
      " \"roll\": 0.0,"               // Dummy
      " \"proj\": \"rectilinear\"}";  // Dummy

  const std::unique_ptr<Ptv::Value> inputDefPtv(makePtvValue(mergedConfig));
  const std::unique_ptr<Core::InputDefinition> inputDef(Core::InputDefinition::create(*inputDefPtv));
  ENSURE((bool)inputDef, "cannot create inputDef");

  const int steps = 3;
  const int pSteps = 3;
  VideoStitch::Core::GeometryDefinition geometry;
  for (int i1 = -pSteps + 1; i1 < pSteps; ++i1) {
    for (int i2 = -pSteps + 1; i2 < pSteps; ++i2) {
      for (int i3 = -pSteps + 1; i3 < pSteps; ++i3) {
        geometry.setDistortA(0.1f * (float)i1 / (float)pSteps);
        geometry.setDistortB(0.1f * (float)i2 / (float)pSteps);
        geometry.setDistortC(0.1f * (float)i3 / (float)pSteps);
        for (int x = -steps + 1; x < steps; ++x) {
          for (int y = -steps + 1; y < steps; ++y) {
            testRadialCommon(make_float2((inputDef->getWidth() * (float)x) / (float)steps,
                                         (inputDef->getHeight() * (float)y) / (float)steps),
                             geometry, *inputDef);
          }
        }
      }
    }
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testColorCorrectionParams();

  std::cout << "Testing Rectilinear round-trip..." << std::endl;
  VideoStitch::Testing::testRoundtrip<VideoStitch::Core::TransformStack::RectToSphere,
                                      VideoStitch::Core::TransformStack::SphereToRect>(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Rectilinear),
      VideoStitch::Core::InputDefinition::Format::Rectilinear);
  std::cout << "  OK" << std::endl;

  std::cout << "Testing Equirectangular round-trip..." << std::endl;
  VideoStitch::Testing::testRoundtrip<VideoStitch::Core::TransformStack::ErectToSphere,
                                      VideoStitch::Core::TransformStack::SphereToErect>(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Equirectangular),
      VideoStitch::Core::InputDefinition::Format::Equirectangular);
  std::cout << "  OK" << std::endl;

  std::cout << "Testing Fisheye round-trip..." << std::endl;
  VideoStitch::Testing::testRoundtrip<VideoStitch::Core::TransformStack::FisheyeToSphere,
                                      VideoStitch::Core::TransformStack::SphereToFisheye>(
      VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::FullFrameFisheye),
      VideoStitch::Core::InputDefinition::Format::FullFrameFisheye);
  std::cout << "  OK" << std::endl;

  VideoStitch::Core::TransformPhotoParam param;

  param.floatParam = 0.0f;
  param.transformData = nullptr;
  VideoStitch::Testing::testPhotoStack<VideoStitch::Core::LinearPhotoCorrection>(param, false, 0.5f);
  std::cout << "  OK" << std::endl;

  param.floatParam = 2.0f;
  param.transformData = nullptr;
  VideoStitch::Testing::testPhotoStack<VideoStitch::Core::GammaPhotoCorrection>(param, false, 0.5f);
  std::cout << "  OK" << std::endl;

  VideoStitch::Core::InvEmorResponseCurve response(3.441228, -0.551975, 0.294423, -0.095632, 0.072139);
  response.invert();
  param.transformData = const_cast<float*>(response.getResponseCurve());
  VideoStitch::Testing::testPhotoStack<VideoStitch::Core::EmorPhotoCorrection>(param, false, 0.5f);
  std::cout << "  OK" << std::endl;

  VideoStitch::Testing::testRadial();

  return 0;
}
