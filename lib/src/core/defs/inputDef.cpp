// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoInputDefsPimpl.hpp"

#include "util/base64.hpp"
#include "core/transformGeoParams.hpp"
#include "common/container.hpp"
#include "parse/json.hpp"
#include "util/pngutil.hpp"
#include "util/strutils.hpp"

#include "libvideostitch/config.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/curves.hpp"

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <limits>
#include <memory>
#include <string>

namespace VideoStitch {
namespace Core {
InputDefinition::Pimpl::Pimpl()
    : deleteMaskedPixels(true),
      maskPixelDataCacheValid(false),
      group(-1),
      cropLeft(std::numeric_limits<int64_t>::max()),
      cropRight(std::numeric_limits<int64_t>::max()),
      cropTop(std::numeric_limits<int64_t>::max()),
      cropBottom(std::numeric_limits<int64_t>::max()),
      format(Format::Rectilinear),
      redCB(new Curve(PTV_DEFAULT_INPUTDEF_REDCB)),
      greenCB(new Curve(PTV_DEFAULT_INPUTDEF_GREENCB)),
      blueCB(new Curve(PTV_DEFAULT_INPUTDEF_BLUECB)),
      exposureValue(new Curve(PTV_DEFAULT_INPUTDEF_EXPOSURE)),
      geometries(new GeometryDefinitionCurve(GeometryDefinition())),
      photoResponse(PhotoResponse::EmorResponse),
      useMeterDistortion(false),
      emorA(PTV_DEFAULT_INPUTDEF_EMORA),
      emorB(PTV_DEFAULT_INPUTDEF_EMORB),
      emorC(PTV_DEFAULT_INPUTDEF_EMORC),
      emorD(PTV_DEFAULT_INPUTDEF_EMORD),
      emorE(PTV_DEFAULT_INPUTDEF_EMORE),
      gamma(PTV_DEFAULT_INPUTDEF_GAMMA),
      valueBasedResponseCurve(nullptr),
      vignettingCoeff0(PTV_DEFAULT_INPUTDEF_VIG_COEFF0),
      vignettingCoeff1(PTV_DEFAULT_INPUTDEF_VIG_COEFF1),
      vignettingCoeff2(PTV_DEFAULT_INPUTDEF_VIG_COEFF2),
      vignettingCoeff3(PTV_DEFAULT_INPUTDEF_VIG_COEFF3),
      vignettingCenterX(PTV_DEFAULT_INPUTDEF_VIG_CENTER_X),
      vignettingCenterY(PTV_DEFAULT_INPUTDEF_VIG_CENTER_Y),
      synchroCost(-1.0),
      stack(0),
      preprocessors(nullptr),
      lensModelCategory(LensModelCategory::Legacy) {}

InputDefinition::Pimpl::~Pimpl() { delete preprocessors; }

InputDefinition* InputDefinition::clone() const {
  InputDefinition* result = new InputDefinition();
  cloneTo(result);

#define PIMPL_FIELD_COPY(field) result->pimpl->field = pimpl->field;
#define AUTO_CURVE_COPY(curve) result->replace##curve(get##curve().clone())
  PIMPL_FIELD_COPY(group);
  PIMPL_FIELD_COPY(cropLeft);
  PIMPL_FIELD_COPY(cropRight);
  PIMPL_FIELD_COPY(cropTop);
  PIMPL_FIELD_COPY(cropBottom);

  PIMPL_FIELD_COPY(format);
  AUTO_CURVE_COPY(RedCB);
  AUTO_CURVE_COPY(GreenCB);
  AUTO_CURVE_COPY(BlueCB);
  AUTO_CURVE_COPY(ExposureValue);
  AUTO_CURVE_COPY(Geometries);
  PIMPL_FIELD_COPY(photoResponse);
  PIMPL_FIELD_COPY(useMeterDistortion);
  PIMPL_FIELD_COPY(emorA);
  PIMPL_FIELD_COPY(emorB);
  PIMPL_FIELD_COPY(emorC);
  PIMPL_FIELD_COPY(emorD);
  PIMPL_FIELD_COPY(emorE);
  PIMPL_FIELD_COPY(gamma);
  if (pimpl->valueBasedResponseCurve) {
    result->pimpl->valueBasedResponseCurve.reset(new std::array<uint16_t, 256>(*pimpl->valueBasedResponseCurve));
  }
  PIMPL_FIELD_COPY(vignettingCoeff0);
  PIMPL_FIELD_COPY(vignettingCoeff1);
  PIMPL_FIELD_COPY(vignettingCoeff2);
  PIMPL_FIELD_COPY(vignettingCoeff3);
  PIMPL_FIELD_COPY(vignettingCenterX);
  PIMPL_FIELD_COPY(vignettingCenterY);
  PIMPL_FIELD_COPY(synchroCost);
  PIMPL_FIELD_COPY(stack);
  PIMPL_FIELD_COPY(maskData);
  result->pimpl->maskPixelDataCacheValid = false;   // Invalidate the cache.
  result->pimpl->maskPixelDataCache.realloc(0, 0);  // Also delete the buffer since the size may have changed.
  PIMPL_FIELD_COPY(deleteMaskedPixels);
  delete result->pimpl->preprocessors;
  result->pimpl->preprocessors = pimpl->preprocessors ? pimpl->preprocessors->clone() : nullptr;
  PIMPL_FIELD_COPY(lensModelCategory);
#undef PIMPL_FIELD_COPY
#undef AUTO_CURVE_COPY

  return result;
}

bool InputDefinition::operator==(const InputDefinition& other) const {
  if (!ReaderInputDefinition::operator==(other)) {
    return false;
  }

#define FIELD_EQUAL(getter) (getter() == other.getter())
  if (!(FIELD_EQUAL(getMaskData) && FIELD_EQUAL(deletesMaskedPixels) && FIELD_EQUAL(getGroup) &&
        FIELD_EQUAL(getCropLeft) && FIELD_EQUAL(getCropRight) && FIELD_EQUAL(getCropTop) &&
        FIELD_EQUAL(getCropBottom) && FIELD_EQUAL(getFormat) && FIELD_EQUAL(getRedCB) && FIELD_EQUAL(getGreenCB) &&
        FIELD_EQUAL(getBlueCB) && FIELD_EQUAL(getGeometries) && FIELD_EQUAL(getExposureValue) &&
        FIELD_EQUAL(getPhotoResponse) && FIELD_EQUAL(getUseMeterDistortion) && FIELD_EQUAL(getEmorA) &&
        FIELD_EQUAL(getEmorB) && FIELD_EQUAL(getEmorC) && FIELD_EQUAL(getEmorD) && FIELD_EQUAL(getEmorE) &&
        FIELD_EQUAL(getGamma) && FIELD_EQUAL(getVignettingCoeff0) && FIELD_EQUAL(getVignettingCoeff1) &&
        FIELD_EQUAL(getVignettingCoeff2) && FIELD_EQUAL(getVignettingCoeff3) && FIELD_EQUAL(getVignettingCenterX) &&
        FIELD_EQUAL(getVignettingCenterY) && FIELD_EQUAL(getSynchroCost) && FIELD_EQUAL(getStack) &&
        FIELD_EQUAL(hasCroppedArea))) {
    return false;
  }

  if (pimpl->preprocessors != nullptr && other.pimpl->preprocessors != nullptr) {
    if (!(*(pimpl->preprocessors) == *other.pimpl->preprocessors)) {
      return false;
    }
  } else if (pimpl->preprocessors != nullptr || other.pimpl->preprocessors != nullptr) {
    return false;
  }

  if (getValueBasedResponseCurve() && other.getValueBasedResponseCurve()) {
    if (*getValueBasedResponseCurve() != *other.getValueBasedResponseCurve()) {
      return false;
    }
  } else if (getValueBasedResponseCurve() != nullptr || other.getValueBasedResponseCurve() != nullptr) {
    return false;
  }

  return true;
#undef FIELD_EQUAL
}

InputDefinition::InputDefinition() : ReaderInputDefinition(), pimpl(new Pimpl()) {}

InputDefinition::~InputDefinition() { delete pimpl; }

bool InputDefinition::validate(std::ostream& os) const {
  if (!ReaderInputDefinition::validate(os)) {
    return false;
  }

  if (getCropLeft() >= getCropRight()) {
    os << "crop_left must be < crop_right." << std::endl;
    return false;
  }
  if (getCropLeft() >= (int)getWidth()) {
    os << "crop_left must be <= width - 1." << std::endl;
    return false;
  }
  if (getCropRight() < 0) {
    os << "crop_right must >= 0." << std::endl;
    return false;
  }
  if (getCropTop() >= getCropBottom()) {
    os << "crop_top must be < crop_bottom." << std::endl;
    return false;
  }
  if (getCropTop() >= (int64_t)getHeight()) {
    os << "crop_top must be <= height - 1." << std::endl;
    return false;
  }
  if (getCropBottom() < 0) {
    os << "crop_bottom must be >= 0." << std::endl;
    return false;
  }
  if (!getReaderConfigPtr()) {
    os << "Missing reader config." << std::endl;
    return false;
  }
  // TODO
  /*if (!exposureValue->validate()) {
    os << "Invalid exposure value." << std::endl;
    return false;
  }*/
  return true;
}

bool InputDefinition::fromPTFormat(const char* ptFmt, InputDefinition::Format* fmt) {
  std::string s(ptFmt);
  if (!s.compare("0")) {
    *fmt = Format::Rectilinear;
    return true;
  } else if (!s.compare("2")) {
    *fmt = Format::CircularFisheye;
    return true;
  } else if (!s.compare("3")) {
    *fmt = Format::FullFrameFisheye;
    return true;
  } else if (!s.compare("4")) {
    *fmt = Format::Equirectangular;
    return true;
  } else {
    assert(false && "Unrecognized PT format");
  }
  return false;
}

bool InputDefinition::getFormatFromName(const std::string& fmt, InputDefinition::Format& fmtOut) {
  if (!fmt.compare("rectilinear")) {
    fmtOut = Format::Rectilinear;
    return true;
  } else if (!fmt.compare("circular_fisheye")) {
    fmtOut = Format::CircularFisheye;
    return true;
  } else if (!fmt.compare("ff_fisheye")) {
    fmtOut = Format::FullFrameFisheye;
    return true;
  } else if (!fmt.compare("equirectangular")) {
    fmtOut = Format::Equirectangular;
    return true;
  } else if (!fmt.compare("circular_fisheye_opt")) {
    fmtOut = Format::CircularFisheye_Opt;
    return true;
  } else if (!fmt.compare("ff_fisheye_opt")) {
    fmtOut = Format::FullFrameFisheye_Opt;
    return true;
  } else {
    return false;
  }
}

InputDefinition::PhotoResponse InputDefinition::getPhotoResponse() const {
  if (pimpl->photoResponse == PhotoResponse::GammaResponse && fabs(getGamma() - 1.0) < 0.01) {
    return PhotoResponse::LinearResponse;
  } else {
    return pimpl->photoResponse;
  }
}

const std::array<uint16_t, 256>* InputDefinition::getValueBasedResponseCurve() const {
  return pimpl->valueBasedResponseCurve.get();
}

// Reseter for Geometries is defined explicitly with an argument, see resetGeometries(const double) below
GENCURVEFUNCTIONS_WITHOUT_RESETER(InputDefinition, CurveTemplate<GeometryDefinition>, Geometries, geometries);
GENCURVEFUNCTIONS(InputDefinition, Curve, RedCB, redCB, PTV_DEFAULT_INPUTDEF_REDCB);
GENCURVEFUNCTIONS(InputDefinition, Curve, GreenCB, greenCB, PTV_DEFAULT_INPUTDEF_GREENCB);
GENCURVEFUNCTIONS(InputDefinition, Curve, BlueCB, blueCB, PTV_DEFAULT_INPUTDEF_BLUECB);
GENCURVEFUNCTIONS(InputDefinition, Curve, ExposureValue, exposureValue, PTV_DEFAULT_INPUTDEF_EXPOSURE);

GENGETSETTER(InputDefinition, InputDefinition::group_t, Group, group)
GENGETSETTER(InputDefinition, InputDefinition::Format, Format, format)
GENGETTER(InputDefinition, InputDefinition::LensModelCategory, LensModelCategory, lensModelCategory)
GENGETSETTER(InputDefinition, bool, UseMeterDistortion, useMeterDistortion)
GENGETSETTER(InputDefinition, double, EmorA, emorA)
GENGETSETTER(InputDefinition, double, EmorB, emorB)
GENGETSETTER(InputDefinition, double, EmorC, emorC)
GENGETSETTER(InputDefinition, double, EmorD, emorD)
GENGETSETTER(InputDefinition, double, EmorE, emorE)
GENGETSETTER(InputDefinition, double, Gamma, gamma)
GENGETSETTER(InputDefinition, double, VignettingCoeff0, vignettingCoeff0)
GENGETSETTER(InputDefinition, double, VignettingCoeff1, vignettingCoeff1)
GENGETSETTER(InputDefinition, double, VignettingCoeff2, vignettingCoeff2)
GENGETSETTER(InputDefinition, double, VignettingCoeff3, vignettingCoeff3)
GENGETSETTER(InputDefinition, double, VignettingCenterX, vignettingCenterX)
GENGETSETTER(InputDefinition, double, VignettingCenterY, vignettingCenterY)
GENGETSETTER(InputDefinition, double, SynchroCost, synchroCost)
GENGETTER(InputDefinition, int, Stack, stack)
GENGETTER(InputDefinition, const std::string&, MaskData, maskData)

void InputDefinition::resetGeometries(const double HFOV) {
  GeometryDefinition geometry;
  geometry.setEstimatedHorizontalFov(*this, HFOV);
  replaceGeometries(new GeometryDefinitionCurve(geometry));
}

void InputDefinition::setCropLeft(int64_t left) {
  // Go through setCrop() to preserve lens centers and calibrations
  setCrop(left, getCropRight(), getCropTop(), getCropBottom());
}

void InputDefinition::setCropRight(int64_t right) {
  // Go through setCrop() to preserve lens centers and calibrations
  setCrop(getCropLeft(), right, getCropTop(), getCropBottom());
}

void InputDefinition::setCropTop(int64_t top) {
  // Go through setCrop() to preserve lens centers and calibrations
  setCrop(getCropLeft(), getCropRight(), top, getCropBottom());
}

void InputDefinition::setCropBottom(int64_t bottom) {
  // Go through setCrop() to preserve lens centers and calibrations
  setCrop(getCropLeft(), getCropRight(), getCropTop(), bottom);
}

void InputDefinition::resetCrop() {
  const auto maxVal = std::numeric_limits<int64_t>::max();
  setCrop(maxVal, maxVal, maxVal, maxVal);
}

int64_t InputDefinition::getCropLeft() const {
  return pimpl->cropLeft == std::numeric_limits<int64_t>::max() ? 0 : pimpl->cropLeft;
}
int64_t InputDefinition::getCropRight() const {
  return pimpl->cropRight == std::numeric_limits<int64_t>::max() ? (int64_t)getWidth() : pimpl->cropRight;
}
int64_t InputDefinition::getCropTop() const {
  return pimpl->cropTop == std::numeric_limits<int64_t>::max() ? 0 : pimpl->cropTop;
}
int64_t InputDefinition::getCropBottom() const {
  return pimpl->cropBottom == std::numeric_limits<int64_t>::max() ? (int64_t)getHeight() : pimpl->cropBottom;
}

int64_t InputDefinition::getCroppedWidth() const { return getCropRight() - getCropLeft(); }

int64_t InputDefinition::getCroppedHeight() const { return getCropBottom() - getCropTop(); }

bool InputDefinition::hasCroppedArea() const {
  return getCropLeft() != 0 || getCropTop() != 0 || getCroppedHeight() != getHeight() ||
         getCroppedWidth() != getWidth();
}

const char* InputDefinition::getFormatName(Format fmt) {
  switch (fmt) {
    case Format::Rectilinear:
      return "rectilinear";
    case Format::CircularFisheye:
      return "circular_fisheye";
    case Format::FullFrameFisheye:
      return "ff_fisheye";
    case Format::Equirectangular:
      return "equirectangular";
    case Format::CircularFisheye_Opt:
      return "circular_fisheye_opt";
    case Format::FullFrameFisheye_Opt:
      return "ff_fisheye_opt";
  }
  return nullptr;
}

bool InputDefinition::deletesMaskedPixels() const { return pimpl->deleteMaskedPixels; }

void InputDefinition::setDeletesMaskedPixels(bool value) { pimpl->deleteMaskedPixels = value; }

void InputDefinition::setMaskData(const std::string& maskData) {
  pimpl->maskData = maskData;
  // Invalidate the cache.
  pimpl->maskPixelDataCacheValid = false;
}

bool InputDefinition::MaskPixelData::realloc(int64_t newWidth, int64_t newHeight) {
  if (newWidth == width && newHeight == height) {
    return true;
  }
  free(data);
  data = nullptr;
  width = newWidth;
  height = newHeight;
  if (newWidth > 0 && newHeight > 0) {
    data = (unsigned char*)malloc((size_t)(getWidth() * getHeight()));
  }
  return data != nullptr;
}

InputDefinition::MaskPixelData::~MaskPixelData() { free(data); }

const InputDefinition::MaskPixelData& InputDefinition::getMaskPixelData() const {
  if (!pimpl->maskData.empty()) {
    if (!pimpl->maskPixelDataCacheValid) {
      pimpl->maskPixelDataCache.realloc(0, 0);
      // Cache miss.
      Util::PngReader pngReader;
      if (pngReader.readMaskFromMemory((unsigned char*)(pimpl->maskData.data()), pimpl->maskData.size(),
                                       pimpl->maskPixelDataCache.width, pimpl->maskPixelDataCache.height,
                                       (void**)&pimpl->maskPixelDataCache.data)) {
        pimpl->maskPixelDataCacheValid = true;
      } else {
        pimpl->maskPixelDataCache.realloc(0, 0);
      }
    }
  }
  return pimpl->maskPixelDataCache;
}
const unsigned char* InputDefinition::getMaskPixelDataIfValid() const {
  return validateMask() ? pimpl->maskPixelDataCache.getData() : nullptr;
}
bool InputDefinition::validateMask() const {
  if (pimpl->maskData.empty()) {
    return true;  // An empty mask is always OK.
  }
  const MaskPixelData& mpd = getMaskPixelData();
  return mpd.getWidth() == getWidth() && mpd.getHeight() == getHeight();
}

bool InputDefinition::setMaskPixelData(const char* buffer, uint64_t maskWidth, uint64_t maskHeight) {
  pimpl->maskPixelDataCacheValid = false;
  // Fill the cache.
  if (!pimpl->maskPixelDataCache.realloc(maskWidth, maskHeight)) {
    return false;
  }
  memcpy(pimpl->maskPixelDataCache.data, buffer,
         (size_t)(pimpl->maskPixelDataCache.getWidth() * pimpl->maskPixelDataCache.getHeight()));
  // Compress.
  Util::PngReader pngReader;
  if (pngReader.writeMaskToMemory(pimpl->maskData, pimpl->maskPixelDataCache.getWidth(),
                                  pimpl->maskPixelDataCache.getHeight(), pimpl->maskPixelDataCache.getData())) {
    pimpl->maskPixelDataCacheValid = true;
  } else {
    Logger::get(Logger::Warning) << "Could not encode mask data." << std::endl;
    return false;
  }
  return true;
}

void InputDefinition::setStack(int value) { pimpl->stack = value; }

const Ptv::Value* InputDefinition::getPreprocessors() const { return pimpl->preprocessors; }

double InputDefinition::getInputCenterX() const {
  return hasCroppedArea() ? static_cast<double>(getCropLeft()) + static_cast<double>(getCroppedWidth()) / 2.
                          : static_cast<double>(getWidth()) / 2.;
}

double InputDefinition::getInputCenterY() const {
  return hasCroppedArea() ? static_cast<double>(getCropTop()) + static_cast<double>(getCroppedHeight()) / 2.
                          : static_cast<double>(getHeight()) / 2.;
}

void InputDefinition::setCrop(const int64_t left, const int64_t right, const int64_t top, const int64_t bottom) {
  // Lens centers may be expressed relative to the crop area center, preserve its actual location
  double oldInputCenterX = getInputCenterX();
  double oldInputCenterY = getInputCenterY();

  // Update crop area - this can change the value of hasCroppedArea()
  pimpl->cropLeft = left;
  pimpl->cropRight = right;
  pimpl->cropTop = top;
  pimpl->cropBottom = bottom;

  double newInputCenterX = getInputCenterX();
  double newInputCenterY = getInputCenterY();

  if (oldInputCenterX != newInputCenterX || oldInputCenterY != newInputCenterY) {
    // Preserve lens centers of calibration data
    CurveTemplate<GeometryDefinition>* geometriesPtr = getGeometries().clone();
    SplineTemplate<GeometryDefinition>* splinesPtr = geometriesPtr->splines();
    if (splinesPtr) {
      while (splinesPtr) {
        splinesPtr->end.v.setCenterX(splinesPtr->end.v.getCenterX() + oldInputCenterX - newInputCenterX);
        splinesPtr->end.v.setCenterY(splinesPtr->end.v.getCenterY() + oldInputCenterY - newInputCenterY);
        splinesPtr = splinesPtr->next;
      }
    } else {
      // splinesPtr is nullptr, geometry is constant and must be retrieved by at(0)
      GeometryDefinition geometry = geometriesPtr->at(0);
      geometry.setCenterX(geometry.getCenterX() + oldInputCenterX - newInputCenterX);
      geometry.setCenterY(geometry.getCenterY() + oldInputCenterY - newInputCenterY);
      geometriesPtr->setConstantValue(geometry);
    }
    replaceGeometries(geometriesPtr);
  }
}

void InputDefinition::setEmorPhotoResponse(double emorA, double emorB, double emorC, double emorD, double emorE) {
  pimpl->photoResponse = PhotoResponse::EmorResponse;
  pimpl->emorA = emorA;
  pimpl->emorB = emorB;
  pimpl->emorC = emorC;
  pimpl->emorD = emorD;
  pimpl->emorE = emorE;
}

void InputDefinition::resetPhotoResponse() {
  setEmorPhotoResponse(PTV_DEFAULT_INPUTDEF_EMORA, PTV_DEFAULT_INPUTDEF_EMORB, PTV_DEFAULT_INPUTDEF_EMORC,
                       PTV_DEFAULT_INPUTDEF_EMORD, PTV_DEFAULT_INPUTDEF_EMORE);
}

void InputDefinition::setValueBasedResponseCurve(const std::array<uint16_t, 256>& values) {
  pimpl->photoResponse = PhotoResponse::CurveResponse;
  pimpl->valueBasedResponseCurve.reset(new std::array<uint16_t, 256>(values));
}

void InputDefinition::setRadialVignetting(double vignettingCoeff0, double vignettingCoeff1, double vignettingCoeff2,
                                          double vignettingCoeff3, double vignettingCenterX, double vignettingCenterY) {
  pimpl->vignettingCoeff0 = vignettingCoeff0;
  pimpl->vignettingCoeff1 = vignettingCoeff1;
  pimpl->vignettingCoeff2 = vignettingCoeff2;
  pimpl->vignettingCoeff3 = vignettingCoeff3;
  pimpl->vignettingCenterX = vignettingCenterX;
  pimpl->vignettingCenterY = vignettingCenterY;
}

void InputDefinition::resetVignetting() {
  pimpl->vignettingCoeff0 = 1;
  pimpl->vignettingCoeff1 = 0;
  pimpl->vignettingCoeff2 = 0;
  pimpl->vignettingCoeff3 = 0;
  pimpl->vignettingCenterX = 0;
  pimpl->vignettingCenterY = 0;
}

InputDefinition* InputDefinition::create(const Ptv::Value& value, bool enforceMandatoryFields) {
  std::unique_ptr<InputDefinition> res(new InputDefinition());
  if (!res->applyDiff(value, enforceMandatoryFields).ok()) {
    return nullptr;
  }
  return res.release();
}

Status InputDefinition::applyDiff(const Ptv::Value& value, bool enforceMandatoryFields) {
  Status stat;

  if (!Parse::checkType("InputDefinition", value, Ptv::Value::OBJECT)) {
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
            "Could not find valid 'InputDefinition', expected object type"};
  }

  stat = ReaderInputDefinition::applyDiff(value, enforceMandatoryFields);
  FAIL_RETURN(stat);

  // TODOLATERSTATUS: this should be handled in the populate function, not here through macros
#define POPULATE_INT_PROPAGATE_WRONGTYPE(config_name, varName, shouldEnforce)                 \
  if (Parse::populateInt("InputDefinition", value, config_name, varName, shouldEnforce) ==    \
      Parse::PopulateResult_WrongType) {                                                      \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                     \
            "Invalid type for '" config_name "' in InputDefinition, expected integer value"}; \
  }

#define POPULATE_BOOL_PROPAGATE_WRONGTYPE(config_name, varName, shouldEnforce)                \
  if (Parse::populateBool("InputDefinition", value, config_name, varName, shouldEnforce) ==   \
      Parse::PopulateResult_WrongType) {                                                      \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                     \
            "Invalid type for '" config_name "' in InputDefinition, expected boolean value"}; \
  }

#define POPULATE_DOUBLE_PROPAGATE_WRONGTYPE(config_name, varName, shouldEnforce)              \
  if (Parse::populateDouble("InputDefinition", value, config_name, varName, shouldEnforce) == \
      Parse::PopulateResult_WrongType) {                                                      \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                     \
            "Invalid type for '" config_name "' in InputDefinition, expected double value"};  \
  }

#define POPULATE_STRING_PROPAGATE_WRONGTYPE(config_name, varName, shouldEnforce)              \
  if (Parse::populateString("InputDefinition", value, config_name, varName, shouldEnforce) == \
      Parse::PopulateResult_WrongType) {                                                      \
    return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,                     \
            "Invalid type for '" config_name "' in InputDefinition, expected string"};        \
  }

  POPULATE_INT_PROPAGATE_WRONGTYPE("group", pimpl->group, false);

  // Support for audio-only inputs
  if (getIsAudioEnabled() && !getIsVideoEnabled()) {
    return Status::OK();
  }

  if (enforceMandatoryFields) {
    std::string tmp;
    if (Parse::populateString("InputDefinition", value, "proj", tmp, enforceMandatoryFields) !=
        Parse::PopulateResult_Ok) {
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
              "Could not parse 'proj' configuration of the InputDefinition"};
    }

    Format format;
    if (!getFormatFromName(tmp, format)) {
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Invalid input projection '" + tmp + "'"};
    }
    pimpl->format = format;
    // Keep track of the lens category
    pimpl->lensModelCategory = (format == Format::FullFrameFisheye_Opt || format == Format::CircularFisheye_Opt)
                                   ? LensModelCategory::Optimized
                                   : LensModelCategory::Legacy;

    std::string decoded;
    POPULATE_STRING_PROPAGATE_WRONGTYPE("mask_data", decoded, false);
    if (!decoded.empty() && !Util::startsWith(decoded.c_str(), "file:")) {
      pimpl->maskData = Util::base64Decode(decoded);
    } else {
      pimpl->maskData = decoded;
    }
    bool noDeleteMaskedPixels = !pimpl->deleteMaskedPixels;
    POPULATE_BOOL_PROPAGATE_WRONGTYPE("no_delete_masked_pixels", noDeleteMaskedPixels, false);
    pimpl->deleteMaskedPixels = !noDeleteMaskedPixels;
    POPULATE_INT_PROPAGATE_WRONGTYPE("crop_left", pimpl->cropLeft, false);
    POPULATE_INT_PROPAGATE_WRONGTYPE("crop_right", pimpl->cropRight, false);
    POPULATE_INT_PROPAGATE_WRONGTYPE("crop_top", pimpl->cropTop, false);
    POPULATE_INT_PROPAGATE_WRONGTYPE("crop_bottom", pimpl->cropBottom, false);
    tmp.clear();
    POPULATE_STRING_PROPAGATE_WRONGTYPE("response", tmp, false);
    if (tmp == "emor") {
      pimpl->photoResponse = PhotoResponse::EmorResponse;
    } else if (tmp == "inverse_emor") {
      pimpl->photoResponse = PhotoResponse::InvEmorResponse;
    } else if (tmp == "gamma") {
      pimpl->photoResponse = PhotoResponse::GammaResponse;
    } else if (tmp == "linear") {
      pimpl->photoResponse = PhotoResponse::LinearResponse;
    } else if (tmp == "curve") {
      pimpl->photoResponse = PhotoResponse::CurveResponse;
    } else {
      return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Invalid photo response '" + tmp + "'"};
    }
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("emor_a", pimpl->emorA, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("emor_b", pimpl->emorB, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("emor_c", pimpl->emorC, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("emor_d", pimpl->emorD, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("emor_e", pimpl->emorE, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("gamma", pimpl->gamma, false);

    {
      std::vector<int64_t> responseCurveList;
      Parse::PopulateResult r =
          Parse::populateIntList("InputDefinition", value, "response_curve", responseCurveList, false);
      switch (r) {
        case Parse::PopulateResult::WrongType:
          return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                  "Invalid type for 'response_curve' in InputDefinition, expected integer list"};
          break;
        case Parse::PopulateResult::DoesNotExist:
          break;
        case Parse::PopulateResult::OK:
          if (responseCurveList.size() == 256) {
            auto values = std::make_unique<std::array<uint16_t, 256>>();
            for (size_t i = 0; i < responseCurveList.size(); i++) {
              int64_t v = responseCurveList[i];
              if (v < 0 || v > std::numeric_limits<uint16_t>::max()) {
                return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                        "Invalid value in 'response_curve' in InputDefinition: " + std::to_string(v)};
              }
              values->operator[](i) = (uint16_t)v;
            }
            pimpl->valueBasedResponseCurve = std::move(values);
          } else {
            return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                    "Invalid number of entries for 'response_curve'. Expected 1024, got " +
                        std::to_string(responseCurveList.size()) + "."};
          }
          break;
      }
    }

    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("vign_a", pimpl->vignettingCoeff0, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("vign_b", pimpl->vignettingCoeff1, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("vign_c", pimpl->vignettingCoeff2, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("vign_d", pimpl->vignettingCoeff3, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("vign_x", pimpl->vignettingCenterX, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("vign_y", pimpl->vignettingCenterY, false);
    POPULATE_DOUBLE_PROPAGATE_WRONGTYPE("synchro_cost", pimpl->synchroCost, false);
    POPULATE_INT_PROPAGATE_WRONGTYPE("stack_order", pimpl->stack, false);

    if (Parse::populateBool("InputDefinition", value, "useMeterDistortion", pimpl->useMeterDistortion, false) !=
        Parse::PopulateResult_Ok) {
      pimpl->useMeterDistortion = false;
    }

    const GeometryDefinition gdef = getGeometries().at(0);
    GeometryDefinition mutablegdef = gdef;

    /*
     * In case some old fashion geometric calibration is found, put it in the first calibration object
     */
    double nval;
    if (Parse::populateDouble("InputDefinition", value, "yaw", nval, false) == Parse::PopulateResult_Ok) {
      mutablegdef.setYaw(nval);
    }
    if (Parse::populateDouble("InputDefinition", value, "pitch", nval, false) == Parse::PopulateResult_Ok) {
      mutablegdef.setPitch(nval);
    }
    if (Parse::populateDouble("InputDefinition", value, "roll", nval, false) == Parse::PopulateResult_Ok) {
      mutablegdef.setRoll(nval);
    }
    if (Parse::populateDouble("InputDefinition", value, "lens_dist_a", nval, false) == Parse::PopulateResult_Ok) {
      mutablegdef.setDistortA(nval);
    }
    if (Parse::populateDouble("InputDefinition", value, "lens_dist_b", nval, false) == Parse::PopulateResult_Ok) {
      mutablegdef.setDistortB(nval);
    }
    if (Parse::populateDouble("InputDefinition", value, "lens_dist_c", nval, false) == Parse::PopulateResult_Ok) {
      mutablegdef.setDistortC(nval);
    }
    if (Parse::populateDouble("InputDefinition", value, "dist_center_x", nval, false) == Parse::PopulateResult_Ok) {
      mutablegdef.setCenterX(nval);
    }
    if (Parse::populateDouble("InputDefinition", value, "dist_center_y", nval, false) == Parse::PopulateResult_Ok) {
      mutablegdef.setCenterY(nval);
    }
    if (Parse::populateDouble("InputDefinition", value, "hfov", nval, false) == Parse::PopulateResult_Ok) {
      double focal = TransformGeoParams::computeHorizontalScale(*this, nval);
      mutablegdef.setHorizontalFocal(focal);
    }

    pimpl->geometries.reset(new GeometryDefinitionCurve(mutablegdef));
  }

#undef POPULATE_INT_PROPAGATE_WRONGTYPE
#undef POPULATE_DOUBLE_PROPAGATE_WRONGTYPE
#undef POPULATE_BOOL_PROPAGATE_WRONGTYPE
#undef POPULATE_STRING_PROPAGATE_WRONGTYPE

  // Curves
  {
    const Ptv::Value* var = value.has("geometries");
    if (var) {
      GeometryDefinitionCurve* curve = GeometryDefinitionCurve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                "Cannot parse geometry definition curve ('geometries')"};
      }

      /*For backward compatibility, convert optional fov to focal*/
      SplineTemplate<GeometryDefinition>* gspline = curve->splines();
      if (gspline) {
        while (gspline) {
          gspline->end.v.convertLoadedFovToFocal(*this);
          gspline = gspline->next;
        }
      } else {
        // gspline is nullptr, geometry is constant and must be retrieved by at(0)
        GeometryDefinition geometry = curve->at(0);
        geometry.convertLoadedFovToFocal(*this);
        curve->setConstantValue(geometry);
      }

      replaceGeometries(curve);
    }
  }
  {
    const Ptv::Value* var = value.has("ev");
    if (var) {
      Curve* curve = Curve::create(*var);
      if (!curve) {
        Logger::get(Logger::Error) << "Cannot parse exposure value." << std::endl;
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration, "Cannot parse exposure value ('ev')"};
      }
      replaceExposureValue(curve);
    }
  }
  {
    const Ptv::Value* var = value.has("red_corr");
    if (var) {
      Curve* curve = Curve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                "Cannot parse red correction value ('red_corr')"};
      }
      replaceRedCB(curve);
    }
  }
  {
    const Ptv::Value* var = value.has("green_corr");
    if (var) {
      Curve* curve = Curve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                "Cannot parse green correction value ('green_corr')"};
      }
      replaceGreenCB(curve);
    }
  }
  {
    const Ptv::Value* var = value.has("blue_corr");
    if (var) {
      Curve* curve = Curve::create(*var);
      if (!curve) {
        return {Origin::PanoramaConfiguration, ErrType::InvalidConfiguration,
                "Cannot parse blue correction value ('blue_corr')"};
      }
      replaceBlueCB(curve);
    }
  }

  // Preprocessors:
  {
    const Ptv::Value* var = value.has("preprocessors");
    if (var) {
      if (pimpl->preprocessors) {
        delete pimpl->preprocessors;
        pimpl->preprocessors = nullptr;
      }
      pimpl->preprocessors = var->clone();
    }
  }

  return stat;
}

Ptv::Value* InputDefinition::serialize() const {
  Ptv::Value* res = ReaderInputDefinition::serialize();

  auto group = getGroup();
  if (group != -1) {
    res->push("group", new Parse::JsonValue((int64_t)getGroup()));
  }

  std::string encoded;

  if (!pimpl->maskData.empty() && !Util::startsWith(pimpl->maskData.c_str(), "file:")) {
    encoded = Util::base64Encode(pimpl->maskData);
  } else {
    encoded = pimpl->maskData;
  }

  res->push("mask_data", new Parse::JsonValue(encoded));
  res->push("no_delete_masked_pixels", new Parse::JsonValue(!deletesMaskedPixels()));
  res->push("proj", new Parse::JsonValue(getFormatName(getFormat())));

  if (pimpl->cropLeft != std::numeric_limits<int64_t>::max()) {
    res->push("crop_left", new Parse::JsonValue(pimpl->cropLeft));
  }
  if (pimpl->cropRight != std::numeric_limits<int64_t>::max()) {
    res->push("crop_right", new Parse::JsonValue(pimpl->cropRight));
  }
  if (pimpl->cropTop != std::numeric_limits<int64_t>::max()) {
    res->push("crop_top", new Parse::JsonValue(pimpl->cropTop));
  }
  if (pimpl->cropBottom != std::numeric_limits<int64_t>::max()) {
    res->push("crop_bottom", new Parse::JsonValue(pimpl->cropBottom));
  }
  res->push("ev", getExposureValue().serialize());
  res->push("red_corr", getRedCB().serialize());
  res->push("green_corr", getGreenCB().serialize());
  res->push("blue_corr", getBlueCB().serialize());

  switch (getPhotoResponse()) {
    case PhotoResponse::LinearResponse:
      res->push("response", new Parse::JsonValue("linear"));
      break;
    case PhotoResponse::GammaResponse:
      res->push("response", new Parse::JsonValue("gamma"));
      break;
    case PhotoResponse::EmorResponse:
      res->push("response", new Parse::JsonValue("emor"));
      break;
    case PhotoResponse::InvEmorResponse:
      res->push("response", new Parse::JsonValue("inverse_emor"));
      break;
    case PhotoResponse::CurveResponse:
      res->push("response", new Parse::JsonValue("curve"));
      break;
  }

  if (getUseMeterDistortion() == true) {
    res->push("useMeterDistortion", new Parse::JsonValue(getUseMeterDistortion()));
  }

  res->push("emor_a", new Parse::JsonValue(getEmorA()));
  res->push("emor_b", new Parse::JsonValue(getEmorB()));
  res->push("emor_c", new Parse::JsonValue(getEmorC()));
  res->push("emor_d", new Parse::JsonValue(getEmorD()));
  res->push("emor_e", new Parse::JsonValue(getEmorE()));
  res->push("gamma", new Parse::JsonValue(getGamma()));

  if (pimpl->valueBasedResponseCurve) {
    std::vector<int64_t> values;
    std::copy(pimpl->valueBasedResponseCurve->begin(), pimpl->valueBasedResponseCurve->end(),
              std::back_inserter(values));
    res->push("response_curve", new Parse::JsonValue(values));
  }

  res->push("vign_a", new Parse::JsonValue(getVignettingCoeff0()));
  res->push("vign_b", new Parse::JsonValue(getVignettingCoeff1()));
  res->push("vign_c", new Parse::JsonValue(getVignettingCoeff2()));
  res->push("vign_d", new Parse::JsonValue(getVignettingCoeff3()));
  res->push("vign_x", new Parse::JsonValue(getVignettingCenterX()));
  res->push("vign_y", new Parse::JsonValue(getVignettingCenterY()));
  res->push("synchro_cost", new Parse::JsonValue(getSynchroCost()));

  // And preprocessors:
  if (pimpl->preprocessors) {
    res->push("preprocessors", pimpl->preprocessors->clone());
  }
  res->push("stack_order", new Parse::JsonValue(getStack()));

  // Inputs:
  res->push("geometries", getGeometries().serialize());
  return res;
}

double InputDefinition::getCenterX(const GeometryDefinition& geometry) const {
  if (hasCroppedArea()) {
    return ((double)getCropLeft() + (double)(getCroppedWidth() - getWidth()) / 2.0 + geometry.getCenterX());
  } else {
    return geometry.getCenterX();
  }
}

double InputDefinition::getCenterY(const GeometryDefinition& geometry) const {
  if (hasCroppedArea()) {
    return ((double)getCropTop() + (double)(getCroppedHeight() - getHeight()) / 2.0 + geometry.getCenterY());
  } else {
    return geometry.getCenterY();
  }
}

void InputDefinition::resetExposure() {
  resetExposureValue();
  resetBlueCB();
  resetGreenCB();
  resetRedCB();
}

void InputDefinition::resetDistortion() {
  setVignettingCenterX(0.);
  setVignettingCenterY(0.);
  GeometryDefinition geometry = getGeometries().at(0);
  geometry.resetDistortion();
  replaceGeometries(new GeometryDefinitionCurve(geometry));
}

double InputDefinition::computeFocalWithoutDistortion() const {
  if (getFormat() == InputDefinition::Format::Equirectangular) {
    return getWidth() / 2.0 / M_PI;
  }
  return std::min(getWidth(), getHeight()) / 2.0;
}

}  // namespace Core
}  // namespace VideoStitch
