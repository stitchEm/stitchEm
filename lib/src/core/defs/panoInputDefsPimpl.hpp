// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/config.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/stereoRigDef.hpp"
#include "libvideostitch/cameraDef.hpp"
#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/rigCameraDef.hpp"
#include "libvideostitch/mergerMaskDef.hpp"
#include "libvideostitch/curves.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/overlayInputDef.hpp"

#include <memory>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace VideoStitch {
namespace Core {

/**
 * RAII for non-null curve pointer.
 */
template <typename CurveT>
class NonNullCurve : public std::unique_ptr<CurveT> {
 public:
  explicit NonNullCurve(CurveT* curve) : std::unique_ptr<CurveT>(curve) {}

 private:
  NonNullCurve();
};

/**
 * Pimpl holder for ControlPointListDefinition.
 */
class ControlPointListDefinition::Pimpl {
 private:
  /**
   * Create with default values.
   */
  Pimpl();
  ~Pimpl();

#ifdef __GNUC__
  Pimpl(const Pimpl& other) = delete;
  bool operator==(const Pimpl& other) const = delete;
  Pimpl& operator=(const Pimpl& other) = delete;
#endif

  friend class ControlPointListDefinition;

  ControlPointList list;
};

/**
 * Pimpl holder for ReaderInputDefinition.
 */
class ReaderInputDefinition::Pimpl {
 private:
  /**
   * Create with default values.
   */
  Pimpl();
  ~Pimpl();

#ifdef __GNUC__
  Pimpl(const Pimpl& other) = delete;
  bool operator==(const Pimpl& other) const = delete;
  Pimpl& operator=(const Pimpl& other) = delete;
#endif

  friend class ReaderInputDefinition;
  friend class PanoDefinition;

  Ptv::Value* readerConfig;
  int64_t width;
  int64_t height;
  /**
   * These parameters are the direct or inverse parameters depending on the reponse type.
   */
  int frameOffset;
  bool isVideoEnabled;
  bool isAudioEnabled;
};

/**
 * Pimpl holder for InputDefinition.
 */
class InputDefinition::Pimpl {
 private:
  /**
   * Create with default values.
   */
  Pimpl();
  ~Pimpl();

#ifdef __GNUC__
  Pimpl(const Pimpl& other) = delete;
  bool operator==(const Pimpl& other) const = delete;
  Pimpl& operator=(const Pimpl& other) = delete;
#endif

  friend class InputDefinition;
  friend class PanoDefinition;

  std::string maskData;
  bool deleteMaskedPixels;
  // Cache for the decompressed pixel data in maskData.
  mutable MaskPixelData maskPixelDataCache;
  mutable bool maskPixelDataCacheValid;
  group_t group;
  int64_t cropLeft;
  int64_t cropRight;
  int64_t cropTop;
  int64_t cropBottom;
  Format format;
  NonNullCurve<Curve> redCB;
  NonNullCurve<Curve> greenCB;
  NonNullCurve<Curve> blueCB;
  NonNullCurve<Curve> exposureValue;
  NonNullCurve<GeometryDefinitionCurve> geometries;
  PhotoResponse photoResponse;
  bool useMeterDistortion;

  /**
   * These parameters are the direct or inverse parameters depending on the reponse type.
   */
  double emorA;
  double emorB;
  double emorC;
  double emorD;
  double emorE;
  double gamma;
  std::unique_ptr<std::array<uint16_t, 256>> valueBasedResponseCurve;
  double vignettingCoeff0;
  double vignettingCoeff1;
  double vignettingCoeff2;
  double vignettingCoeff3;
  double vignettingCenterX;
  double vignettingCenterY;
  double synchroCost;
  int stack;
  Ptv::Value* preprocessors;
  double huginTranslationPlaneYaw;    // Not handled.
  double huginTranslationPlanePitch;  // Not handled.
  LensModelCategory lensModelCategory;
};

/**
 * Pimpl holder for MergerMaskDefinition.
 */
class MergerMaskDefinition::Pimpl {
 private:
  /**
   * Create with default values.
   */
  Pimpl();
  ~Pimpl();

#ifdef __GNUC__
  Pimpl(const Pimpl& other) = delete;
  bool operator==(const Pimpl& other) const = delete;
  Pimpl& operator=(const Pimpl& other) = delete;
#endif
  friend class MergerMaskDefinition;
  // Cache for the decompressed pixel data in maskData.
  mutable InputIndexPixelData inputIndexPixelDataCache;
  std::vector<size_t> maskOrders;
  int64_t width;
  int64_t height;
  int inputScaleFactor;
  bool enabled;
  bool interpolationEnabled;
};

/**
 * Pimpl holder for PanoDefinition.
 */
class PanoDefinition::Pimpl {
 public:
  ~Pimpl();

 private:
  /**
   * Create with default values.
   */
  Pimpl();
#ifdef __GNUC__
  Pimpl(const Pimpl& other) = delete;
  bool operator==(const Pimpl& other) const = delete;
  Pimpl& operator=(const Pimpl& other) = delete;
#endif

  // Not taking ownership of the pointers
  static Potential<PanoDefinition> mergeCalibrationIntoPano(const PanoDefinition* calibration,
                                                            const PanoDefinition* pano);

  friend class PanoDefinition;
  std::vector<InputDefinition*> inputs;
  MergerMaskDefinition* mergerMask;
  ControlPointListDefinition* controlPointList;
  RigDefinition* rigDefinition;
  InputDefinition* audioInput;
  Ptv::Value* postprocessors;
  int64_t width;
  int64_t height;
  int64_t length;
  int64_t cropLeft;
  int64_t cropRight;
  int64_t cropTop;
  int64_t cropBottom;
  double hFOV;
  NonNullCurve<Curve> exposureValue;
  NonNullCurve<Curve> redCB;
  NonNullCurve<Curve> greenCB;
  NonNullCurve<Curve> blueCB;
  bool wrap;
  PanoProjection projection;
  NonNullCurve<QuaternionCurve> orientationCurve;
  NonNullCurve<QuaternionCurve> stabilizationCurve;
  NonNullCurve<Curve> stabilizationYawCurve;
  NonNullCurve<Curve> stabilizationPitchCurve;
  NonNullCurve<Curve> stabilizationRollCurve;
  double sphereScale;
  double calibrationCost;
  double calibrationInitialHFOV;
  bool calibrationDeshuffled;
  bool precomputedCoodinateBuffer;
  double precomputedCoodinateShrinkFactor;
  bool inputsMapInterpolationEnabled;
  std::vector<OverlayInputDefinition*> overlays;
};

/**
 * Pimpl holder for StereoRigDefinition.
 */
class StereoRigDefinition::Pimpl {
 private:
  /**
   * Create with default values.
   */
  Pimpl();
  ~Pimpl();
#ifdef __GNUC__
  Pimpl(const Pimpl& other) = delete;
  bool operator==(const Pimpl& other) const = delete;
  Pimpl& operator=(const Pimpl& other) = delete;
#endif

  friend class StereoRigDefinition;
  Orientation orientation;
  Geometry geometry;
  double diameter;
  double ipd;
  std::vector<int> leftInputs;
  std::vector<int> rightInputs;
};

/**
 * Pimpl holder for CameraDefinition.
 */
class CameraDefinition::Pimpl {
 private:
  /**
   * Create with default values.
   */
  Pimpl();
  ~Pimpl();

#ifdef __GNUC__
  Pimpl(const Pimpl& other) = delete;
  bool operator==(const Pimpl& other) const = delete;
  Pimpl& operator=(const Pimpl& other) = delete;
#endif

  friend class CameraDefinition;

  std::string name;
  size_t width;
  size_t height;
  NormalDouble fu;
  NormalDouble fv;
  NormalDouble cu;
  NormalDouble cv;
  NormalDouble distortion[3];
  InputDefinition::Format format;
};

/**
 * Pimpl holder for RigCameraDefinition.
 */
class RigCameraDefinition::Pimpl {
 public:
  Pimpl();
  ~Pimpl();
  Pimpl(const Pimpl& other);

 private:
  NormalDouble yaw;
  NormalDouble pitch;
  NormalDouble roll;
  NormalDouble translation_x;
  NormalDouble translation_y;
  NormalDouble translation_z;
  std::shared_ptr<CameraDefinition> camera;

  friend class RigCameraDefinition;
};

/**
 * Pimpl holder for RigDefinition.
 */
class RigDefinition::Pimpl {
 private:
  /**
   * Create with default values.
   */
  Pimpl();
  ~Pimpl();

#ifdef __GNUC__
  Pimpl(const Pimpl& other) = delete;
  bool operator==(const Pimpl& other) const = delete;
  Pimpl& operator=(const Pimpl& other) = delete;
#endif

  std::string name;
  std::vector<RigCameraDefinition> cameras;

  friend class RigDefinition;
};

class OverlayInputDefinition::Pimpl {
 private:
  /**
   * Create with default values.
   */
  Pimpl();
  ~Pimpl();

#ifdef __GNUC__
  Pimpl(const Pimpl& other) = delete;
  bool operator==(const Pimpl& other) const = delete;
  Pimpl& operator=(const Pimpl& other) = delete;
#endif
  bool globalOrientationApplied;
  NonNullCurve<Curve> scaleCurve;
  NonNullCurve<Curve> alphaCurve;
  NonNullCurve<Curve> transXCurve;
  NonNullCurve<Curve> transYCurve;
  NonNullCurve<Curve> transZCurve;
  NonNullCurve<QuaternionCurve> rotationCurve;

  friend class OverlayInputDefinition;
};
}  // namespace Core
}  // namespace VideoStitch

#define GENPOINTERGETTER(class, type, exportName, member) \
  const type& class ::get##exportName() const { return *pimpl->member; }

#define GENREFGETTER(class, type, exportName, member) \
  const type& class ::get##exportName() const { return pimpl->member; }

#define GENGETTER(class, type, exportName, member) \
  type class ::get##exportName() const { return pimpl->member; }

#define GENSETTER(class, type, exportName, member) \
  void class ::set##exportName(type member) { pimpl->member = member; }

#define GENREFSETTER(class, type, exportName, member) \
  void class ::set##exportName(const type& member) { pimpl->member = member; }

#define GENGETREFSETTER(class, type, exportName, member) \
  GENGETTER(class, type, exportName, member)             \
  GENREFSETTER(class, type, exportName, member)

#define GENGETSETTER(class, type, exportName, member) \
  GENGETTER(class, type, exportName, member)          \
  GENSETTER(class, type, exportName, member)

#define GENCURVESETTER(class, type, exportName, member) \
  void class ::replace##exportName(type* newCurve) { pimpl->member.reset(newCurve); }

#define GENCURVEDISPLACER(class, type, exportName, member) \
  type* class ::displace##exportName(type* newCurve) {     \
    type* tmp = pimpl->member.release();                   \
    pimpl->member.reset(newCurve);                         \
    return tmp;                                            \
  }

#define GENCURVERESETER(class, type, exportName, member, defaultValue) \
  void class ::reset##exportName() { pimpl->member.reset(new type(defaultValue)); }

#define GENCURVEFUNCTIONS_WITHOUT_RESETER(class, type, exportName, member) \
  GENPOINTERGETTER(class, type, exportName, member)                        \
  GENCURVESETTER(class, type, exportName, member)                          \
  GENCURVEDISPLACER(class, type, exportName, member)

#define GENCURVEFUNCTIONS(class, type, exportName, member, defaultValue) \
  GENCURVEFUNCTIONS_WITHOUT_RESETER(class, type, exportName, member)     \
  GENCURVERESETER(class, type, exportName, member, defaultValue)
