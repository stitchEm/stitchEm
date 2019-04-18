// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "projectdefinition.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"

#include "libvideostitch-base/file.hpp"
#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/geometryDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/imageFlowFactory.hpp"
#include "libvideostitch/imageWarperFactory.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/projections.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/rigDef.hpp"
#include "libvideostitch/config.hpp"
#include "version.hpp"

#include <QSet>
#include <memory>

#define VS_MUTEX_LOCK QMutexLocker lock(&mutex);

ProjectDefinition::ProjectDefinition()
    : mutex(QMutex::Recursive),
      signalCompressor(SignalCompressionCaps::create()),
      isModified(false),
      drawInputNumbers(false),
      displayOrientationGrid(false) {}

ProjectDefinition::~ProjectDefinition() {
  VS_MUTEX_LOCK
  signalCompressor->autoDelete();
}

bool ProjectDefinition::validatePanorama(std::stringstream& errsink) {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return false;
  } else {
    return getPanoConst()->validate(errsink);
  }
}

void ProjectDefinition::setDefaultPtvValues(const VideoStitch::Ptv::Value& value) {
  VS_MUTEX_LOCK
  load(value);
}

QString ProjectDefinition::getInputName(int index) const {
  VideoStitch::Core::InputDefinition& input = getPano()->getInput(index);
  const std::string name = input.getDisplayName();
  QString formattedName = QString::fromStdString(name);
  if (formattedName.contains("%0")) {
    formattedName = formattedName.arg(index);
  }
  return formattedName;
}

void ProjectDefinition::copyValuesIntoVector(std::vector<VideoStitch::Ptv::Value*>& vector,
                                             const QVector<int> inputList) {
  for (auto value : inputList) {
    VideoStitch::Ptv::Value* input = VideoStitch::Ptv::Value::emptyObject();
    input->asInt() = value;
    vector.push_back(input);
  }
}

bool ProjectDefinition::load(const VideoStitch::Ptv::Value& value) {
  VS_MUTEX_LOCK
  bool oldHasImagesOrProceduralsOnly = hasImagesOrProceduralsOnly();
  bool oldHasSeveralVideos = hasSeveralVideos();
  bool oldHasSeveralVisualInputs = hasSeveralVisualInputs();

  // close the current project
  destroyDelegate();
  setModified(false);
  drawInputNumbers = false;
  displayOrientationGrid = false;
  // parse a new project definition
  createDelegate(value);

  bool newHasImagesOrProceduralsOnly = hasImagesOrProceduralsOnly();
  if (oldHasImagesOrProceduralsOnly != newHasImagesOrProceduralsOnly) {
    emit imagesOrProceduralsOnlyHasChanged(newHasImagesOrProceduralsOnly);
  }
  bool newHasSeveralVideos = hasSeveralVideos();
  if (oldHasSeveralVideos != newHasSeveralVideos) {
    emit severalVideosOnlyHasChanged(newHasSeveralVideos);
  }
  bool newHasSeveralVisualInputs = hasSeveralVisualInputs();
  if (oldHasSeveralVisualInputs != newHasSeveralVisualInputs) {
    emit severalVisualInputsHasChanged(newHasSeveralVisualInputs);
  }
  return getDelegate() != nullptr;
}

void ProjectDefinition::close() { destroyDelegate(); }

bool ProjectDefinition::isInit() const {
  VS_MUTEX_LOCK
  return getDelegate() != nullptr;
}

VideoStitch::Ptv::Value* ProjectDefinition::serialize() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return nullptr;
  } else {
    return getDelegate()->serialize();
  }
}

PanoDefinitionLocked ProjectDefinition::getPano() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return PanoDefinitionLocked(nullptr, this);
  } else {
    return PanoDefinitionLocked(&getDelegate()->getPanoDefinition(), this);
  }
}

const PanoDefinitionLocked ProjectDefinition::getPanoConst() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return PanoDefinitionLocked(nullptr, this);
  } else {
    return PanoDefinitionLocked(&getDelegate()->getPanoDefinition(), this);
  }
}

AudioPipeDefinitionLocked ProjectDefinition::getAudioPipe() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return AudioPipeDefinitionLocked(nullptr, this);
  } else {
    return AudioPipeDefinitionLocked(&getDelegate()->getAudioPipe(), this);
  }
}

const AudioPipeDefinitionLocked ProjectDefinition::getAudioPipeConst() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return AudioPipeDefinitionLocked(nullptr, this);
  } else {
    return AudioPipeDefinitionLocked(&getDelegate()->getAudioPipe(), this);
  }
}

StereoRigDefinitionLocked ProjectDefinition::getStereoRig() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return StereoRigDefinitionLocked(nullptr, this);
  } else {
    return StereoRigDefinitionLocked(getDelegate()->getStereoRigDefinition(), this);
  }
}

const StereoRigDefinitionLocked ProjectDefinition::getStereoRigConst() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return StereoRigDefinitionLocked(nullptr, this);
  } else {
    return StereoRigDefinitionLocked(getDelegate()->getStereoRigDefinition(), this);
  }
}

ImageMergerFactoryLocked ProjectDefinition::getImageMergerFactory() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return ImageMergerFactoryLocked(nullptr, this);
  } else {
    return ImageMergerFactoryLocked(&getDelegate()->getImageMergerFactory(), this);
  }
}

ImageFlowFactoryLocked ProjectDefinition::getImageFlowFactory() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return ImageFlowFactoryLocked(NULL, this);
  } else {
    return ImageFlowFactoryLocked(&getDelegate()->getImageFlowFactory(), this);
  }
}

ImageWarperFactoryLocked ProjectDefinition::getImageWarperFactory() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return ImageWarperFactoryLocked(NULL, this);
  } else {
    return ImageWarperFactoryLocked(&getDelegate()->getImageWarperFactory(), this);
  }
}

QString ProjectDefinition::getBlender() const {
  VS_MUTEX_LOCK
  std::unique_ptr<VideoStitch::Ptv::Value> mergerfactory(getImageMergerFactory()->serialize());
  std::string blender = mergerfactory->has("type") ? mergerfactory->has("type")->asString() : std::string();
  return QString::fromStdString(blender);
}

int ProjectDefinition::getFeather() const {
  VS_MUTEX_LOCK
  std::unique_ptr<VideoStitch::Ptv::Value> mergerfactory(getImageMergerFactory()->serialize());
  if (mergerfactory->has("feather")) {
    return mergerfactory->has("feather")->asInt();
  } else {
    return 100;
  }
}

double ProjectDefinition::getSphereScale() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return PTV_DEFAULT_PANODEF_SPHERE_SCALE;
  } else {
    return getPanoConst()->getSphereScale();
  }
}

double ProjectDefinition::computeMinimumSphereScale() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return PTV_DEFAULT_PANODEF_MIN_SPHERE_SCALE;
  } else {
    return getPanoConst()->computeMinimumRigSphereRadius();
  }
}

void ProjectDefinition::setSphereScale(double sphereScale) {
  VS_MUTEX_LOCK
  if (getDelegate()) {
    if (getPanoConst()->getSphereScale() != sphereScale) {
      getPano()->setSphereScale(sphereScale);
      setModified(true);
      emit reqReset(signalCompressor->add());
    }
  }
}

QString ProjectDefinition::getFlow() const {
  VS_MUTEX_LOCK
  return QString::fromStdString(getImageFlowFactory()->getImageFlowName());
}

QString ProjectDefinition::getWarper() const {
  VS_MUTEX_LOCK
  std::string warper = getImageWarperFactory()->getImageWarperName();
  return QString::fromStdString(warper);
}

readerid_t ProjectDefinition::getNumInputs() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return 0;
  } else {
    return getPanoConst()->numInputs();
  }
}

QStringList ProjectDefinition::getInputNames() const {
  VS_MUTEX_LOCK
  QStringList inputNames;
  if (getDelegate()) {
    for (readerid_t i = 0; i < getPano()->numInputs(); ++i) {
      QString inputName = getInputName(int(i));
      if (!inputName.isEmpty()) {
        inputNames.append(inputName);
      }
    }
  }
  return inputNames;
}

QStringList ProjectDefinition::getAudioInputNames() const {
  VS_MUTEX_LOCK
  QStringList inputNames;
  if (getDelegate()) {
    for (size_t i = 0; i < getAudioPipeConst()->numAudioMixes(); ++i) {
      inputNames.append(QString::fromStdString(getAudioPipeConst()->getMix(int(i))->getName()));
    }
  }
  return inputNames;
}

QStringList ProjectDefinition::getVideoInputNames() const {
  VS_MUTEX_LOCK
  QStringList inputNames;
  if (getDelegate()) {
    for (VideoStitch::Core::InputDefinition& input : getPano()->getVideoInputs()) {
      inputNames.append(QString::fromStdString(input.getDisplayName()));
    }
  }
  return inputNames;
}

VideoStitch::InputFormat::InputFormatEnum ProjectDefinition::getVideoInputType() const {
  VS_MUTEX_LOCK;
  for (readerid_t inputIndex = 0; inputIndex < getNumInputs(); ++inputIndex) {
    const VideoStitch::Core::InputDefinition& input = getPano()->getInput(inputIndex);
    if (input.getIsVideoEnabled()) {
      const VideoStitch::Ptv::Value& value = input.getReaderConfig();
      if (value.has("type") != nullptr) {
        return VideoStitch::InputFormat::getEnumFromString(QString::fromStdString(value.has("type")->asString()));
      } else {
        const QString name(QString::fromStdString(value.asString()));
        if (VideoStitch::InputFormat::isVideoStream(name)) {
          return VideoStitch::InputFormat::InputFormatEnum::NETWORK;
        } else if (VideoStitch::InputFormat::isVideoFile(name)) {
          return VideoStitch::InputFormat::InputFormatEnum::MEDIA;
        }
      }
    }
  }
  return VideoStitch::InputFormat::InputFormatEnum::INVALID;
}

bool ProjectDefinition::hasAnInputWithAudioOnly() const {
  VS_MUTEX_LOCK;
  for (readerid_t inputIndex = 0; inputIndex < getNumInputs(); ++inputIndex) {
    const VideoStitch::Core::InputDefinition& input = getPano()->getInput(inputIndex);
    if (input.getIsAudioEnabled() && !input.getIsVideoEnabled()) {
      return true;
    }
  }
  return false;
}

bool ProjectDefinition::hasAudio() const {
  VS_MUTEX_LOCK;
  return getAudioPipeConst().get() != nullptr && getAudioPipeConst()->numAudioInputs() > 0;
}

bool ProjectDefinition::hasImagesOrProceduralsOnly() const {
  VS_MUTEX_LOCK;
  for (readerid_t inputIndex = 0; inputIndex < getNumInputs(); ++inputIndex) {
    const VideoStitch::Core::InputDefinition& input = getPano()->getInput(inputIndex);
    const VideoStitch::Ptv::Value& value = input.getReaderConfig();
    if (input.getIsVideoEnabled() && value.getType() == VideoStitch::Ptv::Value::STRING) {
      const QString name(QString::fromStdString(value.asString()));
      bool isAProcedural = name.startsWith("procedural:");
      bool isAnImage = File::getTypeFromFile(name) == File::STILL_IMAGE;
      if (!isAProcedural && !isAnImage) {
        return false;
      }
    }
  }
  return true;
}

bool ProjectDefinition::hasSeveralVideos() const {
  VS_MUTEX_LOCK;
  int nbVideoInputs = 0;
  for (readerid_t inputIndex = 0; inputIndex < getNumInputs(); ++inputIndex) {
    const VideoStitch::Core::InputDefinition& input = getPano()->getInput(inputIndex);
    const VideoStitch::Ptv::Value& value = input.getReaderConfig();
    const QString name(QString::fromStdString(value.asString()));
    if (input.getIsVideoEnabled() && value.getType() == VideoStitch::Ptv::Value::STRING) {
      const QString name(QString::fromStdString(value.asString()));
      bool isAProcedural = name.startsWith("procedural:");
      bool isAnImage = File::getTypeFromFile(name) == File::STILL_IMAGE;
      if (isAProcedural || isAnImage) {
        return false;
      }
      ++nbVideoInputs;
    }
  }
  return nbVideoInputs >= 2;
}

bool ProjectDefinition::hasSeveralVisualInputs() const {
  VS_MUTEX_LOCK;
  return getPanoConst().get() != nullptr && getPanoConst()->numVideoInputs() >= 2;
}

InputLensClass::LensType ProjectDefinition::getProjectLensType() const {
  return InputLensClass::getLensTypeFromInputDefinitionFormat(getPanoConst()->getLensFormatFromInputSources());
}

bool ProjectDefinition::hasCroppedArea() const {
  for (const VideoStitch::Core::InputDefinition& inputDef : getPanoConst()->getVideoInputs()) {
    if (inputDef.hasCroppedArea()) {
      return true;
    }
  }
  return false;
}

QString ProjectDefinition::getProjection() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return "equirectangular";
  } else {
    return QString(getPanoConst()->getFormatName(getPanoConst()->getProjection()));
  }
}

double ProjectDefinition::getHFOV() const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    return 160.0;
  } else {
    return getPanoConst()->getHFOV();
  }
}

bool ProjectDefinition::hasFileFormatChanged() const {
  if (getDelegate()) {
    return getDelegate()->hasFileFormatChanged();
  } else {
    return false;
  }
}

void ProjectDefinition::updateFileFormat() {
  if (getDelegate()) {
    getDelegate()->updateFileFormat();
  }
}

void ProjectDefinition::setAudioPipe(VideoStitch::Core::AudioPipeDefinition* audioPipeDefinition) {
  VS_MUTEX_LOCK
  if (getDelegate() && audioPipeDefinition != nullptr && (&getDelegate()->getAudioPipe() != audioPipeDefinition)) {
    getDelegate()->setAudioPipe(audioPipeDefinition);
    setModified(true);
  }
}

void ProjectDefinition::setPano(VideoStitch::Core::PanoDefinition* panoDefinition) {
  VS_MUTEX_LOCK
  if (getDelegate() && panoDefinition != nullptr && (&getDelegate()->getPanoDefinition() != panoDefinition)) {
    getDelegate()->setPano(panoDefinition);
    setModified(true);
  }
}

void ProjectDefinition::setMergerFactory(VideoStitch::Core::ImageMergerFactory* merger) {
  VS_MUTEX_LOCK
  if (getDelegate() && merger != nullptr && (&getDelegate()->getImageMergerFactory() != merger)) {
    getDelegate()->setMergerFactory(merger);
    setModified(true);
  }
}

void ProjectDefinition::setFlowFactory(VideoStitch::Core::ImageFlowFactory* flow) {
  VS_MUTEX_LOCK
  if (getDelegate() && flow != nullptr && (&getDelegate()->getImageFlowFactory() != flow)) {
    getDelegate()->setFlowFactory(flow);
    setModified(true);
  }
}

void ProjectDefinition::setWarperFactory(VideoStitch::Core::ImageWarperFactory* warper) {
  VS_MUTEX_LOCK
  if (getDelegate() && warper != nullptr && (&getDelegate()->getImageWarperFactory() != warper)) {
    getDelegate()->setWarperFactory(warper);
    setModified(true);
  }
}

void ProjectDefinition::setStereoRigDefinition(VideoStitch::Core::StereoRigDefinition* rigDefinition) {
  VS_MUTEX_LOCK
  if (getDelegate() && rigDefinition != nullptr && (getDelegate()->getStereoRigDefinition() != rigDefinition)) {
    getDelegate()->setRig(rigDefinition);
    setModified(true);
  }
}

void ProjectDefinition::setModified(const bool modified) {
  VS_MUTEX_LOCK
  if (modified != isModified) {
    emit hasBeenModified(modified);
  }
  isModified = modified;
}

void ProjectDefinition::markAsSaved() { setModified(false); }

bool ProjectDefinition::hasLocalModifications() const {
  VS_MUTEX_LOCK
  return isModified;
}

void ProjectDefinition::updateSize(int width, int height) {
  VS_MUTEX_LOCK
  if (getDelegate()) {
    if (getPanoConst()->getWidth() != width || getPanoConst()->getHeight() != height) {
      getPano()->setWidth(width);
      getPano()->setHeight(height);
      setModified(true);
    }
  }
}

void ProjectDefinition::setProjection(QString projection) {
  VS_MUTEX_LOCK
  if (getDelegate()) {
    if (getPanoConst()->getFormatName(getPanoConst()->getProjection()) != projection) {
      getPano()->setProjection(getPanoConst()->getFormatFromName(projection.toStdString()));
      setModified(true);
    }
  }
}

bool ProjectDefinition::isDrawingInputNumbers() const {
  VS_MUTEX_LOCK
  return drawInputNumbers;
}

void ProjectDefinition::setHFov(double fov) {
  VS_MUTEX_LOCK
  if (getDelegate()) {
    if (getPanoConst()->getHFOV() != fov) {
      getPano()->setHFOV(fov);
      setModified(true);
    }
  }
}

void ProjectDefinition::setDrawInputNumbers(bool draw) {
  VS_MUTEX_LOCK
  drawInputNumbers = draw;
}

void ProjectDefinition::toggleInputNumberDrawing() {
  VS_MUTEX_LOCK
  drawInputNumbers = !drawInputNumbers;
  emit reqToggleInputNumbers(drawInputNumbers);
}

void ProjectDefinition::changeBlendingParameters(QString merger, int feather) {
  VS_MUTEX_LOCK

  std::unique_ptr<VideoStitch::Ptv::Value> value(VideoStitch::Ptv::Value::emptyObject());
  value->get("type")->asString() = merger.toStdString();
  value->get("feather")->asInt() = feather;
  auto fStatus = VideoStitch::Core::ImageMergerFactory::createMergerFactory(*value);
  if (!fStatus.ok()) {
    MsgBoxHandlerHelper::genericErrorMessage({VideoStitch::Origin::Stitcher, VideoStitch::ErrType::SetupFailure,
                                              tr("Could not create the image merger").toStdString(), fStatus.status()});
    return;
  }

  setMergerFactory(fStatus.release());
}

void ProjectDefinition::changeAdvancedBlendingParameters(const QString& flow, const QString& warper) {
  VS_MUTEX_LOCK

  std::unique_ptr<VideoStitch::Ptv::Value> flowValue(VideoStitch::Ptv::Value::emptyObject());
  flowValue->get("type")->asString() = flow.toStdString();
  auto flowStatus = VideoStitch::Core::ImageFlowFactory::createFlowFactory(flowValue.get());
  if (!flowStatus.ok()) {
    MsgBoxHandlerHelper::genericErrorMessage({VideoStitch::Origin::Stitcher, VideoStitch::ErrType::SetupFailure,
                                              tr("Could not create the image flow").toStdString(),
                                              flowStatus.status()});
    return;
  }

  std::unique_ptr<VideoStitch::Ptv::Value> warperValue(VideoStitch::Ptv::Value::emptyObject());
  warperValue->get("type")->asString() = warper.toStdString();
  auto warperStatus = VideoStitch::Core::ImageWarperFactory::createWarperFactory(warperValue.get());
  if (!warperStatus.ok()) {
    MsgBoxHandlerHelper::genericErrorMessage({VideoStitch::Origin::Stitcher, VideoStitch::ErrType::SetupFailure,
                                              tr("Could not create the image warper").toStdString(),
                                              warperStatus.status()});
    return;
  }

  setFlowFactory(flowStatus.release());
  setWarperFactory(warperStatus.release());
}

void ProjectDefinition::setDisplayOrientationGrid(bool display, bool reqRestitch) {
  VS_MUTEX_LOCK
  displayOrientationGrid = display;
  if (reqRestitch) {
    emit reqDisplayOrientationGrid(displayOrientationGrid);
  }
}

bool ProjectDefinition::getDisplayOrientationGrid() const {
  VS_MUTEX_LOCK
  return displayOrientationGrid;
}

void ProjectDefinition::getCroppedImageSize(unsigned& width, unsigned& height) const {
  VS_MUTEX_LOCK
  if (!getDelegate()) {
    width = 0;
    height = 0;
    return;
  } else {
    width = getPanoConst()->getWidth();
    height = getPanoConst()->getHeight();
  }
}

#define INTERACTIVE_FOV 360
#define MAX_STEREOGRAPHIC_FOV 360
#define DEFAULT_RECTILINEAR_FOV 160
#define DEFAULT_STEREOGRAPHIC_FOV 320
#define MAX_RECTILINEAR_FOV 180
void ProjectDefinition::changeProjection(const QString& projName, const double HFOV) {
  if (!isInit()) {
    return;
  }

  QString projection = projName;
  VideoStitch::Core::PanoProjection proj =
      (projName == "interactive")
          ? VideoStitch::Core::PanoProjection(VideoStitch::Core::PanoProjection::Equirectangular)
          : getPano()->getFormatFromName(projName.toStdString());
  double fov = HFOV;
  double currHFOV = getPano()->getHFOV();
  if (getPano()->getProjection() != proj || (currHFOV != HFOV)) {
    // particular cases
    currHFOV = HFOV;
    if (projName == "interactive") {
      //"interactive" is "equirectangular" displayed differently
      if ((getPano()->getHFOV() == INTERACTIVE_FOV) &&
          getPano()->getProjection() == VideoStitch::Core::PanoProjection::Equirectangular) {
        return;
      }
      projection = "equirectangular";
      fov = INTERACTIVE_FOV;

      // FIXME: we modify the user settings to avoid rejection of the PTV but we never set it back, see #232
      updateSize(getPano()->getWidth(), getPano()->getWidth() / 2);
    } else {
      getPano()->setProjection(getPano()->getFormatFromName(projName.toStdString()));
      if (proj == VideoStitch::Core::PanoProjection::Stereographic && (currHFOV >= MAX_STEREOGRAPHIC_FOV)) {
        fov = DEFAULT_STEREOGRAPHIC_FOV;
        getPano()->setHFOV(DEFAULT_STEREOGRAPHIC_FOV);
      } else if (proj == VideoStitch::Core::PanoProjection::Rectilinear && (currHFOV >= MAX_RECTILINEAR_FOV)) {
        fov = DEFAULT_RECTILINEAR_FOV;
        getPano()->setHFOV(DEFAULT_RECTILINEAR_FOV);
      }
    }
    setHFov(fov);
    setProjection(projection);
    emit reqReset(signalCompressor->add());
  }
}

void ProjectDefinition::updateMasks() {
  if (!isInit()) {
    Q_ASSERT(0);
    return;
  }
  bool userNeedsToBeWarned = false;
  for (int i = 0; i < (int)getPano()->numInputs(); i++) {
    VideoStitch::Core::InputDefinition& inputDef = getPano()->getInput(i);
    const VideoStitch::Core::InputDefinition::MaskPixelData& mask = inputDef.getMaskPixelData();

    int maskSize = mask.getWidth() * mask.getHeight();
    if (mask.getData() != nullptr) {
      if (mask.getWidth() == inputDef.getWidth() && mask.getHeight() == inputDef.getHeight()) {
        unsigned char* data = new unsigned char[maskSize];
        memcpy(data, mask.getData(), maskSize);
        emit reqUpdateMask(i, data, mask.getWidth(), mask.getHeight());
      } else {
        userNeedsToBeWarned = true;
        QImage* scaledImage =
            new QImage(QImage(mask.getData(), mask.getWidth(), mask.getHeight(), QImage::Format_Indexed8)
                           .scaled(inputDef.getWidth(), inputDef.getHeight()));
        inputDef.setMaskPixelData((char*)scaledImage->bits(), inputDef.getWidth(), inputDef.getHeight());
        emit reqUpdateMask(i, scaledImage);
      }
    }
  }
  emit maskUpdateDone();
  if (userNeedsToBeWarned) {
    emit reqDisplayWarning(tr("The size of your mask doesn't match the input's size.") + QString("<br>") +
                           tr("Your mask has been scaled to fit the input size."));
  }
}

bool ProjectDefinition::hasRigConfiguration() const {
  if (!isInit()) {
    return false;
  } else {
    return getDelegate()->getStereoRigDefinition() != nullptr;
  }
}

int ProjectDefinition::getAudioDelay() const {
  if (getAudioPipeConst().get() && getAudioPipeConst()->numAudioInputs() > 0) {
    return int(getAudioPipeConst()->getDelay(getAudioPipeConst()->getInput(0)->getName()).ref());
  } else {
    return 0;
  }
}

void ProjectDefinition::setInputCrop(const unsigned int currentInput, const Crop& crop,
                                     const InputLensClass::LensType lensType, const bool applyToAll) {
  const auto initIndex = applyToAll ? 0 : currentInput;
  const auto lastIndex = applyToAll ? getPano()->numInputs() : currentInput + 1;
  VideoStitch::Core::PanoDefinition* clonedPano = getPano()->clone();
  for (auto input = initIndex; input < lastIndex; ++input) {
    // Skip non-video inputs
    if (!clonedPano->getInput(input).getIsVideoEnabled()) {
      continue;
    }
    clonedPano->getInput(input).setFormat(InputLensClass::getInputDefinitionFormatFromLensType(
        lensType, getPanoConst()->getLensModelCategoryFromInputSources()));
    clonedPano->getInput(input).setCrop(crop.crop_left, crop.crop_right, crop.crop_top, crop.crop_bottom);
  }
  setPano(clonedPano);
}

void ProjectDefinition::setRigConfiguration(const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                                            const VideoStitch::Core::StereoRigDefinition::Geometry geometry,
                                            const double diameter, const double ipd, const QVector<int> leftInputs,
                                            const QVector<int> rightInputs) {
  VideoStitch::Core::StereoRigDefinition* clonedRig = nullptr;
  // There's no Rig configuration yet

  StereoRigDefinitionLocked rig = getStereoRigConst();
  if (rig.get() != nullptr) {
    VideoStitch::Ptv::Value* rigValue = VideoStitch::Ptv::Value::emptyObject();
    std::vector<VideoStitch::Ptv::Value*> leftVector;
    std::vector<VideoStitch::Ptv::Value*> rightVector;
    copyValuesIntoVector(leftVector, leftInputs);
    copyValuesIntoVector(rightVector, rightInputs);
    rigValue->get("orientation")->asString() = VideoStitch::Core::StereoRigDefinition::getOrientationName(orientation);
    rigValue->get("geometry")->asString() = VideoStitch::Core::StereoRigDefinition::getGeometryName(geometry);
    rigValue->get("diameter")->asDouble() = diameter;
    rigValue->get("ipd")->asDouble() = ipd;
    rigValue->get("left_inputs")->asList() = leftVector;
    rigValue->get("right_inputs")->asList() = rightVector;
    clonedRig = VideoStitch::Core::StereoRigDefinition::create(*rigValue);
    // There's an existing rig already
  } else {
    clonedRig = getStereoRig()->clone();
    clonedRig->setOrientation(orientation);
    clonedRig->setGeometry(geometry);
    clonedRig->setDiameter(diameter);
    clonedRig->setIPD(ipd);
    clonedRig->setLeftInputs(leftInputs.toStdVector());
    clonedRig->setRightInputs(rightInputs.toStdVector());
  }
  setStereoRigDefinition(clonedRig);
}

void ProjectDefinition::setInterPupillaryDistance(double ipd) {
  VideoStitch::Core::StereoRigDefinition* clonedRig = getStereoRig()->clone();
  clonedRig->setIPD(ipd);
  setStereoRigDefinition(clonedRig);
  emit reqResetRig(signalCompressor->add());
}

void ProjectDefinition::setAudioDelay(int delay_ms) {
  VS_MUTEX_LOCK
  emit reqSetAudioDelay(delay_ms);
}

void ProjectDefinition::lock() const { mutex.lock(); }

void ProjectDefinition::unlock() const { mutex.unlock(); }
