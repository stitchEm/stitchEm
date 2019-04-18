// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "photoTransform.hpp"

#include "kernels/photoStack.cu"
#include "backend/common/vectorOps.hpp"

#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"

#include "libvideostitch/emor.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"

// TODO_OPENCL_IMPL
// backend dependent code needs to be moved to the backend itself
#ifdef VS_OPENCL
#include <backend/cl/deviceBuffer.hpp>
#else
#include <backend/cuda/deviceBuffer.hpp>
#endif

namespace VideoStitch {
namespace Core {

/**
 * PhotoTransform implementation.
 */

namespace {

template <class PhotoCorrection>
class DevicePhotoTransformImpl : public DevicePhotoTransform {
 public:
  DevicePhotoTransformImpl(double idds, TransformPhotoParam devicePhotoParam)
      : DevicePhotoTransform(idds, devicePhotoParam) {}

  virtual ~DevicePhotoTransformImpl() {}
};

template <class PhotoCorrection>
class HostPhotoTransformImpl : public HostPhotoTransform {
 public:
  HostPhotoTransformImpl(double idds, TransformPhotoParam hostPhotoParam) : HostPhotoTransform(idds, hostPhotoParam) {}

  virtual ~HostPhotoTransformImpl() {}

  float3 mapPhotoInputToLinear(const InputDefinition& im, TopLeftCoords2 uv, float3 rgb) const;
  float3 mapPhotoLinearToPano(float3 rgb) const;
  float3 mapPhotoPanoToLinear(float3 rgb) const;
};

template <class PhotoCorrection>
float3 HostPhotoTransformImpl<PhotoCorrection>::mapPhotoInputToLinear(const InputDefinition& im, TopLeftCoords2 uv,
                                                                      float3 rgb) const {
  return (float)computeVignettingMult(im, uv) *
         PhotoCorrection::corr(rgb, hostPhotoParam.floatParam, (float*)hostPhotoParam.transformData);
}

template <class PhotoCorrection>
float3 HostPhotoTransformImpl<PhotoCorrection>::mapPhotoLinearToPano(float3 rgb) const {
  return PhotoCorrection::invCorr(rgb, hostPhotoParam.floatParam, (float*)hostPhotoParam.transformData);
}

template <class PhotoCorrection>
float3 HostPhotoTransformImpl<PhotoCorrection>::mapPhotoPanoToLinear(float3 rgb) const {
  return PhotoCorrection::corr(rgb, hostPhotoParam.floatParam, (float*)hostPhotoParam.transformData);
}

/**
 * A PhotoTransform that holds an Emor lookup table.
 */
class CustomHostCurveTransform : public HostPhotoTransformImpl<EmorPhotoCorrection> {
 public:
  static CustomHostCurveTransform* create(const ResponseCurve& response, double inverseDemiDiagonalSquared) {
    TransformPhotoParam hostPhotoParam;
    hostPhotoParam.floatParam = 0.;
    size_t lookupTableSize = ResponseCurve::totalLutSize() * sizeof(float);
    float* hostData = new float[ResponseCurve::totalLutSize()];
    if (!hostData) {
      return nullptr;
    }
    memcpy(hostData, response.getResponseCurve(), lookupTableSize);

    hostPhotoParam.transformData = hostData;

    return new CustomHostCurveTransform(inverseDemiDiagonalSquared, hostPhotoParam);
  }

  ~CustomHostCurveTransform() { delete[](float*) this->hostPhotoParam.transformData; }

 private:
  CustomHostCurveTransform(double idds, TransformPhotoParam hostPhotoParam)
      : HostPhotoTransformImpl<EmorPhotoCorrection>(idds, hostPhotoParam) {}
};

/**
 * A PhotoTransform that holds an Emor lookup table.
 */
class CustomCurveTransform : public DevicePhotoTransformImpl<EmorPhotoCorrection> {
 public:
  static CustomCurveTransform* create(const ResponseCurve& response, double inverseDemiDiagonalSquared) {
    TransformPhotoParam devicePhotoParam;
    size_t lookupTableSize = ResponseCurve::totalLutSize() * sizeof(float);

    auto deviceData = GPU::Buffer<float>::allocate(lookupTableSize, "Transform");
    if (!deviceData.ok()) {
      return nullptr;
    }

    if (!GPU::memcpyBlocking(deviceData.value(), response.getResponseCurve(), lookupTableSize).ok()) {
      deviceData.value().release();
      return nullptr;
    }

    devicePhotoParam.floatParam = 0.f;
    devicePhotoParam.transformData = deviceData.value().get().raw();

    return new CustomCurveTransform(inverseDemiDiagonalSquared, devicePhotoParam, deviceData.value().as_const());
  }

  ~CustomCurveTransform() { deviceData.release(); }

 private:
  CustomCurveTransform(double idds, TransformPhotoParam devicePhotoParam, GPU::Buffer<const float> deviceData)
      : DevicePhotoTransformImpl<EmorPhotoCorrection>(idds, devicePhotoParam), deviceData(deviceData) {}

  GPU::Buffer<const float> deviceData;
};
}  // namespace

HostPhotoTransform* HostPhotoTransform::create(const InputDefinition& im) {
  HostPhotoTransform* res = nullptr;
  TransformPhotoParam photoParam;
  double inverseDemiDiagonalSquared =
      im.hasCroppedArea()
          ? 4.0f / (float)(im.getCroppedWidth() * im.getCroppedWidth() + im.getCroppedHeight() * im.getCroppedHeight())
          : 4.0f / (float)(im.getWidth() * im.getWidth() + im.getHeight() * im.getHeight());
  switch (im.getPhotoResponse()) {
    case InputDefinition::PhotoResponse::LinearResponse:
      photoParam.floatParam = 0.0f; /*dummy*/
      photoParam.transformData = nullptr;
      res = new HostPhotoTransformImpl<LinearPhotoCorrection>(inverseDemiDiagonalSquared, photoParam);
      break;
    case InputDefinition::PhotoResponse::GammaResponse:
      photoParam.floatParam = (float)im.getGamma();
      photoParam.transformData = nullptr;
      res = new HostPhotoTransformImpl<GammaPhotoCorrection>(inverseDemiDiagonalSquared, photoParam);
      break;
    case InputDefinition::PhotoResponse::EmorResponse: {
      EmorResponseCurve response(im.getEmorA(), im.getEmorB(), im.getEmorC(), im.getEmorD(), im.getEmorE());
      res = CustomHostCurveTransform::create(response, inverseDemiDiagonalSquared);
      break;
    }
    case InputDefinition::PhotoResponse::InvEmorResponse: {
      InvEmorResponseCurve response(im.getEmorA(), im.getEmorB(), im.getEmorC(), im.getEmorD(), im.getEmorE());
      response.invert();
      res = CustomHostCurveTransform::create(response, inverseDemiDiagonalSquared);
      break;
    }
    case InputDefinition::PhotoResponse::CurveResponse: {
      const std::array<uint16_t, 256>* values = im.getValueBasedResponseCurve();
      if (!values) {
        break;
      }
      ValueResponseCurve response(*values);
      res = CustomHostCurveTransform::create(response, inverseDemiDiagonalSquared);
      break;
    }
  }
  return res;
}

DevicePhotoTransform* DevicePhotoTransform::create(const InputDefinition& im) {
  DevicePhotoTransform* res = nullptr;
  TransformPhotoParam photoParam;
  double inverseDemiDiagonalSquared =
      im.hasCroppedArea()
          ? 4.0f / (float)(im.getCroppedWidth() * im.getCroppedWidth() + im.getCroppedHeight() * im.getCroppedHeight())
          : 4.0f / (float)(im.getWidth() * im.getWidth() + im.getHeight() * im.getHeight());
  switch (im.getPhotoResponse()) {
    case InputDefinition::PhotoResponse::LinearResponse:
      photoParam.floatParam = 0.0f; /* dummy */
      photoParam.transformData = nullptr;
      res = new DevicePhotoTransformImpl<LinearPhotoCorrection>(inverseDemiDiagonalSquared, photoParam);
      break;
    case InputDefinition::PhotoResponse::GammaResponse:
      photoParam.floatParam = (float)im.getGamma();
      photoParam.transformData = nullptr;
      res = new DevicePhotoTransformImpl<GammaPhotoCorrection>(inverseDemiDiagonalSquared, photoParam);
      break;
    case InputDefinition::PhotoResponse::EmorResponse: {
      EmorResponseCurve response(im.getEmorA(), im.getEmorB(), im.getEmorC(), im.getEmorD(), im.getEmorE());
      res = CustomCurveTransform::create(response, inverseDemiDiagonalSquared);
      break;
    }
    case InputDefinition::PhotoResponse::InvEmorResponse: {
      InvEmorResponseCurve response(im.getEmorA(), im.getEmorB(), im.getEmorC(), im.getEmorD(), im.getEmorE());
      response.invert();
      res = CustomCurveTransform::create(response, inverseDemiDiagonalSquared);
      break;
    }
    case InputDefinition::PhotoResponse::CurveResponse: {
      const std::array<uint16_t, 256>* values = im.getValueBasedResponseCurve();
      if (!values) {
        break;
      }
      ValueResponseCurve response(*values);
      res = CustomCurveTransform::create(response, inverseDemiDiagonalSquared);
      break;
    }
  }
  return res;
}

// ----------------------- Photo Transform  -------------------------------

HostPhotoTransform::HostPhotoTransform(double idds, TransformPhotoParam hostPhotoParam)
    : PhotoTransform(idds), hostPhotoParam(hostPhotoParam) {}
DevicePhotoTransform::DevicePhotoTransform(double idds, TransformPhotoParam devicePhotoParam)
    : PhotoTransform(idds), devicePhotoParam(devicePhotoParam) {}

PhotoTransform::PhotoTransform(double idds) : inverseDemiDiagonalSquared(idds) {}

float3 PhotoTransform::ColorCorrectionParams::computeColorMultiplier(double panoEv, double panoRedCB,
                                                                     double panoGreenCB, double panoBlueCB) const {
  const double exposureRatio = pow(2.0, ev + panoEv);
  return make_float3((float)(exposureRatio / (panoRedCB * redCB)), (float)(exposureRatio / (panoGreenCB * greenCB)),
                     (float)(exposureRatio / (panoBlueCB * blueCB)));
}

PhotoTransform::ColorCorrectionParams PhotoTransform::ColorCorrectionParams::canonicalFromMultiplier(
    float3 colorMult, double panoEv, double panoRedCB, double panoGreenCB, double panoBlueCB) {
  // Make sure we don't create NaNs.
  const float normalizer = colorMult.y > std::numeric_limits<float>::min() ? (colorMult.y * (float)panoGreenCB) : 1.0f;
  const double redCB = colorMult.x > std::numeric_limits<float>::min() ? normalizer / ((float)panoRedCB * colorMult.x)
                                                                       : std::numeric_limits<float>::max();
  const double greenCB = colorMult.y > std::numeric_limits<float>::min()
                             ? normalizer / ((float)panoGreenCB * colorMult.y)
                             : std::numeric_limits<float>::max();
  const double blueCB = colorMult.z > std::numeric_limits<float>::min() ? normalizer / ((float)panoBlueCB * colorMult.z)
                                                                        : std::numeric_limits<float>::max();
  return ColorCorrectionParams(log(normalizer) / log(2.0) - panoEv, redCB, greenCB, blueCB);
}

double PhotoTransform::computeVignettingMult(const InputDefinition& im, const CenterCoords2& uv) const {
  const double dx = (double)uv.x - im.getVignettingCenterX();
  const double dy = (double)uv.y - im.getVignettingCenterY();
  const double vigRadiusSquared = (dx * dx + dy * dy) * inverseDemiDiagonalSquared;
  double vigMult = vigRadiusSquared * im.getVignettingCoeff3();
  vigMult += im.getVignettingCoeff2();
  vigMult *= vigRadiusSquared;
  vigMult += im.getVignettingCoeff1();
  vigMult *= vigRadiusSquared;
  vigMult += im.getVignettingCoeff0();
  return 1.0 / vigMult;
}

double PhotoTransform::computeVignettingMult(const InputDefinition& im, const TopLeftCoords2& uv) const {
  return computeVignettingMult(
      im, CenterCoords2(uv, TopLeftCoords2((float)im.getWidth() / 2.0f, (float)im.getHeight() / 2.0f)));
}

float3 HostPhotoTransform::mapPhotoInputToPano(int time, const PanoDefinition& pano, const InputDefinition& im,
                                               TopLeftCoords2 uv, float3 rgb) const {
  const float3 colorMult = ColorCorrectionParams(im.getExposureValue().at(time), im.getRedCB().at(time),
                                                 im.getGreenCB().at(time), im.getBlueCB().at(time))
                               .computeColorMultiplier(pano.getExposureValue().at(time), pano.getRedCB().at(time),
                                                       pano.getGreenCB().at(time), pano.getBlueCB().at(time));
  rgb = mapPhotoInputToLinear(im, uv, rgb);
  rgb = mapPhotoCorrectLinear(colorMult, rgb);
  return mapPhotoLinearToPano(rgb);
}

float3 HostPhotoTransform::mapPhotoCorrectLinear(float3 colorMult, float3 rgb) const {
  rgb.x *= colorMult.x;
  rgb.y *= colorMult.y;
  rgb.z *= colorMult.z;
  return rgb;
}
}  // namespace Core
}  // namespace VideoStitch
