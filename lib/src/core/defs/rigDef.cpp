// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoInputDefsPimpl.hpp"

#include "parse/json.hpp"
#include "common/angles.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/cameraDef.hpp"

#include <sstream>

namespace VideoStitch {
namespace Core {

RigDefinition::Pimpl::Pimpl() : name("") {}

RigDefinition::RigDefinition() : pimpl(new Pimpl()) {}

RigDefinition* RigDefinition::clone() const {
  RigDefinition* result = new RigDefinition();

  result->pimpl->name = pimpl->name;
  result->pimpl->cameras = pimpl->cameras;

  return result;
}

bool RigDefinition::operator==(const RigDefinition& other) const {
  if (getName() != other.getName()) return false;

  if (other.pimpl->cameras.size() != pimpl->cameras.size()) return false;

  for (size_t i = 0; i < pimpl->cameras.size(); i++) {
    if (!(other.pimpl->cameras[i] == pimpl->cameras[i])) return false;
  }

  return true;
}

bool RigDefinition::validate(std::ostream& os, const size_t numCameras) const {
  if (getName() == "") {
    os << "Invalid name" << std::endl;
    return false;
  }

  if (getRigCameraDefinitionCount() != numCameras) {
    os << "Invalid number of cameras in rig definition" << std::endl;
    return false;
  }

  return true;
}

RigDefinition::~RigDefinition() { delete pimpl; }

RigDefinition::Pimpl::~Pimpl() {}

RigDefinition* RigDefinition::create(const std::map<std::string, std::shared_ptr<CameraDefinition>>& camerasmap,
                                     const Ptv::Value& value) {
  if (!Parse::checkType("Rig", value, Ptv::Value::OBJECT)) {
    return nullptr;
  }

  std::unique_ptr<RigDefinition> res(new RigDefinition());

  if (Parse::populateString("RigDefinition", value, "name", res->pimpl->name, true) != Parse::PopulateResult_Ok) {
    return nullptr;
  }

  /*Check that the cameras list if found*/
  const Ptv::Value* var = value.has("rigcameras");
  if (!Parse::checkVar("RigDefinition", "rigcameras", var, true)) {
    return nullptr;
  }

  /*Check that it is a list*/
  if (!Parse::checkType("rigcameras", *var, Ptv::Value::LIST)) {
    return nullptr;
  }

  std::vector<Ptv::Value*> lcam = var->asList();

  for (size_t i = 0; i < lcam.size(); ++i) {
    RigCameraDefinition cam;

    if (!cam.deserialize(camerasmap, *lcam[i])) {
      return nullptr;
    }

    res->pimpl->cameras.push_back(cam);
  }

  return res.release();
}

RigDefinition* RigDefinition::createBasicUnknownRig(const std::string& name, const InputDefinition::Format& lensformat,
                                                    size_t nbcameras, size_t image_width, size_t image_height,
                                                    size_t cropped_width, size_t cropped_height, double fov,
                                                    const PanoDefinition* pano) {
  if (fov < 1.0) {
    return nullptr;
  }

  if (nbcameras == 0) {
    return nullptr;
  }

  std::unique_ptr<RigDefinition> res(new RigDefinition);
  std::shared_ptr<CameraDefinition> cam(new CameraDefinition);
  res->pimpl->name = name;

  double dw = (double)image_width;
  double dh = (double)image_height;

  cam->setName(name);
  cam->setType(lensformat);

  /*Set width and height to cropped_width and cropped_height temporarily, for the FOV computation*/
  cam->setWidth(cropped_width);
  cam->setHeight(cropped_height);
  cam->setFov({fov, std::numeric_limits<double>::max()});
  /*Set the real width and height*/
  cam->setWidth(image_width);
  cam->setHeight(image_height);
  cam->setCu({dw / 2.0, dw * dw});
  cam->setCv({dh / 2.0, dh * dh});

  /*Lock 2 of 3 distortion parameters, or we're overfitting*/
  cam->setDistortionA({0.0, 0.0});
  cam->setDistortionC({0.0, 0.0});

  RigCameraDefinition caminstance;
  caminstance.setCamera(cam);
  caminstance.setYawRadians({0.0, RigCameraDefinition::max_yaw_variance});
  caminstance.setPitchRadians({0.0, RigCameraDefinition::max_pitch_variance});
  caminstance.setRollRadians({0.0, RigCameraDefinition::max_roll_variance});

  auto videoInputs =
      (pano != nullptr) ? pano->getVideoInputs() : std::vector<std::reference_wrapper<const InputDefinition>>();
  for (size_t i = 0; i < nbcameras; i++) {
    // initialize the pose of the cameras, if a PanoDefinition was passed
    if (videoInputs.size() > i) {
      const InputDefinition& idef = videoInputs[i];
      GeometryDefinition g = idef.getGeometries().at(0);

      caminstance.setYawRadians({degToRad(g.getYaw()), RigCameraDefinition::max_yaw_variance});
      caminstance.setPitchRadians({degToRad(g.getPitch()), RigCameraDefinition::max_pitch_variance});
      caminstance.setRollRadians({degToRad(g.getRoll()), RigCameraDefinition::max_roll_variance});

      caminstance.setTranslationX({g.getTranslationX(), RigCameraDefinition::max_translation_x_variance});
      caminstance.setTranslationY({g.getTranslationY(), RigCameraDefinition::max_translation_y_variance});
      caminstance.setTranslationZ({g.getTranslationZ(), RigCameraDefinition::max_translation_z_variance});
    }
    res->pimpl->cameras.push_back(caminstance);
  }

  return res.release();
}

// Helper functions
const auto varianceFromStdDevValuePercentage = [](const double val, const double percentage) {
  return (val * percentage / 100.) * (val * percentage / 100.);
};
const auto varianceFromStdDev = [](const double val) { return val * val; };

RigDefinition* RigDefinition::createFromPanoDefinitionTemplate(
    const std::string& name, const double focalStdDevValuePercentage, const double centerStdDevWidthPercentage,
    const double distortStdDevValuePercentage, const double yawStdDevDegrees, const double pitchStdDevDegrees,
    const double rollStdDevDegrees, const double translationXStdDev, const double translationYStdDev,
    const double translationZStdDev, const PanoDefinition& pano, const bool applyPanoGlobalOrientation) {
  std::unique_ptr<RigDefinition> res(new RigDefinition);
  res->pimpl->name = name;

  if (!pano.numVideoInputs()) {
    return nullptr;
  }

  size_t cameraId = 0;
  for (auto videoInput : pano.getVideoInputs()) {
    std::shared_ptr<CameraDefinition> cam(new CameraDefinition);
    const InputDefinition& idef = videoInput.get();

    std::stringstream camName;
    camName << "camera_" << cameraId++;

    cam->setName(camName.str());
    cam->setType(idef.getFormat());

    int64_t width = idef.getWidth();
    int64_t height = idef.getHeight();

    cam->setWidth(width);
    cam->setHeight(height);

    GeometryDefinition g = idef.getGeometries().at(0);

    if (applyPanoGlobalOrientation) {
      g.applyGlobalOrientation(pano.getGlobalOrientation().at(0));
    }

    /*Unlike CenterX and CenterY, Cu and Cv are relative to the image border*/
    double cu, cv;
    if (idef.hasCroppedArea()) {
      cu = g.getCenterX() + (idef.getCroppedWidth() + 2 * idef.getCropLeft()) / 2;
      cv = g.getCenterY() + (idef.getCroppedHeight() + 2 * idef.getCropTop()) / 2;
    } else {
      cu = g.getCenterX() + width / 2;
      cv = g.getCenterY() + height / 2;
    }

    cam->setCu({cu, varianceFromStdDevValuePercentage(double(width), centerStdDevWidthPercentage)});
    cam->setCv({cv, varianceFromStdDevValuePercentage(double(width), centerStdDevWidthPercentage)});

    cam->setFu({g.getHorizontalFocal(),
                varianceFromStdDevValuePercentage(g.getHorizontalFocal(), focalStdDevValuePercentage)});
    cam->setFv(
        {g.getVerticalFocal(), varianceFromStdDevValuePercentage(g.getVerticalFocal(), focalStdDevValuePercentage)});

    cam->setDistortionA(
        {g.getDistortA(), varianceFromStdDevValuePercentage(g.getDistortA(), distortStdDevValuePercentage)});
    cam->setDistortionB(
        {g.getDistortB(), varianceFromStdDevValuePercentage(g.getDistortB(), distortStdDevValuePercentage)});
    cam->setDistortionC(
        {g.getDistortC(), varianceFromStdDevValuePercentage(g.getDistortC(), distortStdDevValuePercentage)});

    RigCameraDefinition caminstance;
    caminstance.setCamera(cam);

    caminstance.setYawRadians({degToRad(g.getYaw()), varianceFromStdDev(degToRad(yawStdDevDegrees))});
    caminstance.setPitchRadians({degToRad(g.getPitch()), varianceFromStdDev(degToRad(pitchStdDevDegrees))});
    caminstance.setRollRadians({degToRad(g.getRoll()), varianceFromStdDev(degToRad(rollStdDevDegrees))});

    caminstance.setTranslationX({g.getTranslationX(), varianceFromStdDev(translationXStdDev)});
    caminstance.setTranslationY({g.getTranslationY(), varianceFromStdDev(translationYStdDev)});
    caminstance.setTranslationZ({g.getTranslationZ(), varianceFromStdDev(translationZStdDev)});

    res->pimpl->cameras.push_back(caminstance);
  }

  return res.release();
}

void RigDefinition::overridePresetsStandardDeviations(const double focalStdDevValuePercentage,
                                                      const double centerStdDevWidthPercentage,
                                                      const double distortStdDevValuePercentage,
                                                      const double yawStdDevDegrees, const double pitchStdDevDegrees,
                                                      const double rollStdDevDegrees, const double translationXStdDev,
                                                      const double translationYStdDev,
                                                      const double translationZStdDev) {
  for (auto& caminstance : pimpl->cameras) {
    // replace variances in lens parameters
    auto cam = caminstance.getCamera();
    cam->setFu({cam->getFu().mean, varianceFromStdDevValuePercentage(cam->getFu().mean, focalStdDevValuePercentage)});
    cam->setFv({cam->getFv().mean, varianceFromStdDevValuePercentage(cam->getFv().mean, focalStdDevValuePercentage)});

    cam->setCu(
        {cam->getCu().mean, varianceFromStdDevValuePercentage(double(cam->getWidth()), centerStdDevWidthPercentage)});
    cam->setCv(
        {cam->getCv().mean, varianceFromStdDevValuePercentage(double(cam->getWidth()), centerStdDevWidthPercentage)});

    cam->setDistortionA({cam->getDistortionA().mean,
                         varianceFromStdDevValuePercentage(cam->getDistortionA().mean, distortStdDevValuePercentage)});
    cam->setDistortionB({cam->getDistortionB().mean,
                         varianceFromStdDevValuePercentage(cam->getDistortionB().mean, distortStdDevValuePercentage)});
    cam->setDistortionC({cam->getDistortionC().mean,
                         varianceFromStdDevValuePercentage(cam->getDistortionC().mean, distortStdDevValuePercentage)});

    // replace variances for Yaw/Pitch/Roll and translations
    caminstance.setYawRadians({caminstance.getYawRadians().mean, varianceFromStdDev(degToRad(yawStdDevDegrees))});
    caminstance.setPitchRadians({caminstance.getPitchRadians().mean, varianceFromStdDev(degToRad(pitchStdDevDegrees))});
    caminstance.setRollRadians({caminstance.getRollRadians().mean, varianceFromStdDev(degToRad(rollStdDevDegrees))});
    caminstance.setTranslationX({caminstance.getTranslationX().mean, varianceFromStdDev(translationXStdDev)});
    caminstance.setTranslationY({caminstance.getTranslationY().mean, varianceFromStdDev(translationYStdDev)});
    caminstance.setTranslationZ({caminstance.getTranslationZ().mean, varianceFromStdDev(translationZStdDev)});
  }
}

Ptv::Value* RigDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();

  res->push("name", new Parse::JsonValue(getName()));

  /*Serialize list of rigcameras*/
  Ptv::Value* jsonCameras = new Parse::JsonValue((void*)nullptr);
  jsonCameras->asList();
  for (size_t i = 0; i < pimpl->cameras.size(); ++i) {
    jsonCameras->asList().push_back(pimpl->cameras[i].serialize());
  }
  res->push("rigcameras", jsonCameras);

  return res;
}

/************ Getters and Setters **********/

std::string RigDefinition::getName() const { return pimpl->name; }

void RigDefinition::setName(const std::string& val) { pimpl->name = val; }

bool RigDefinition::getRigCameraDefinition(RigCameraDefinition& cam, size_t n) const {
  if (n > pimpl->cameras.size()) {
    return false;
  }

  cam = pimpl->cameras[n];

  return true;
}

std::map<std::string, std::shared_ptr<CameraDefinition>> RigDefinition::getRigCameraDefinitionMap() const {
  std::map<std::string, std::shared_ptr<CameraDefinition>> map;
  for (auto it : pimpl->cameras) {
    map[it.getCamera()->getName()] = it.getCamera();
  }

  return map;
}

size_t RigDefinition::getRigCameraDefinitionCount() const { return pimpl->cameras.size(); }

void RigDefinition::removeRigCameraDefinition() {
  for (size_t i = 0; i < pimpl->cameras.size(); ++i) {
    pimpl->cameras.erase(pimpl->cameras.begin() + i);
  }
  pimpl->cameras.clear();
}

}  // namespace Core
}  // namespace VideoStitch
