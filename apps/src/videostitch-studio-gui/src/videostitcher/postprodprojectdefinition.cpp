// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "postprodprojectdefinition.hpp"

#include "postprodmutableprojectdefinition.hpp"

#include "libvideostitch-gui/base/ptvMerger.hpp"
#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/videostitcher/presetsmanager.hpp"
#include "libvideostitch-gui/utils/audiohelpers.hpp"

#include "libvideostitch-base/file.hpp"

#include "libvideostitch/input.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/audioPipeDef.hpp"

#include <QFileInfo>
#include <QSet>
#include <QThread>

#define VS_MUTEX_LOCK QMutexLocker lock(&mutex);

static const QString DEFAULT_OUTPUTNAME("output");

PostProdProjectDefinition::PostProdProjectDefinition() : ProjectDefinition(), delegate(nullptr) {}

PostProdProjectDefinition::~PostProdProjectDefinition() { destroyDelegate(); }

void PostProdProjectDefinition::createDelegate(const VideoStitch::Ptv::Value& value) {
  delegate = PostProdMutableProjectDefinition::create(value);
}

void PostProdProjectDefinition::destroyDelegate() {
  delete delegate;
  delegate = nullptr;
}

PostProdMutableProjectDefinition* PostProdProjectDefinition::getDelegate() const { return delegate; }

VideoStitch::Ptv::Value* PostProdProjectDefinition::serialize() const {
  VS_MUTEX_LOCK
  return delegate->serialize();
}

frameid_t PostProdProjectDefinition::getFirstFrame() const {
  VS_MUTEX_LOCK
  return delegate->getFirstFrame();
}

frameid_t PostProdProjectDefinition::getLastFrame() const {
  VS_MUTEX_LOCK
  return delegate->getLastFrame();
}

bool PostProdProjectDefinition::isLastFrameAuto() const {
  VS_MUTEX_LOCK
  return getLastFrame() == -1;
}

PtvValueLocked PostProdProjectDefinition::getOutputConfig() const {
  VS_MUTEX_LOCK
  return PtvValueLocked(&delegate->getOutputConfig(), this);
}

std::vector<int> PostProdProjectDefinition::getOffsets() const {
  VS_MUTEX_LOCK
  std::vector<int> ret;
  for (int i = 0; i < int(getPanoConst()->numInputs()); ++i) {
    ret.push_back(getPanoConst()->getInput(i).getFrameOffset());
  }
  return ret;
}

void PostProdProjectDefinition::getImageSize(unsigned& width, unsigned& height) const {
  VS_MUTEX_LOCK
  width = getPanoConst()->getWidth();
  height = getPanoConst()->getHeight();
}

void PostProdProjectDefinition::getOptimalSize(unsigned& width, unsigned& height) const {
  VS_MUTEX_LOCK
  getPanoConst()->computeOptimalPanoSize(width, height);
}

QString PostProdProjectDefinition::getOutputVideoFormat() const {
  VS_MUTEX_LOCK
  QString outputFormat;
  if (getOutputConfig()->getType() != VideoStitch::Ptv::Value::OBJECT) {
    return outputFormat;
  }

  std::string outputType;
  if (VideoStitch::Parse::populateString("Ptv", *getOutputConfig().get(), "type", outputType, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    outputFormat = QString::fromStdString(outputType);
  }
  return outputFormat;
}

QString PostProdProjectDefinition::getOutputVideoCodec() const {
  VS_MUTEX_LOCK
  QString outputCodec;
  if (getOutputConfig()->getType() != VideoStitch::Ptv::Value::OBJECT) {
    return outputCodec;
  }
  std::string outputType;
  if (VideoStitch::Parse::populateString("Ptv", *getOutputConfig().get(), "video_codec", outputType, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    outputCodec = QString::fromStdString(outputType);
  }
  return outputCodec;
}

QString PostProdProjectDefinition::getOutputFilename() const {
  VS_MUTEX_LOCK
  QString ret = DEFAULT_OUTPUTNAME;
  if (getOutputConfig()->getType() != VideoStitch::Ptv::Value::OBJECT) {
    return ret;
  }

  std::string outputFilename;
  if (VideoStitch::Parse::populateString("Ptv", *getOutputConfig().get(), "filename", outputFilename, true) ==
      VideoStitch::Parse::PopulateResult_Ok) {
    ret = outputFilename.empty() ? DEFAULT_OUTPUTNAME : QString::fromStdString(outputFilename);
  }
  return ret;
}

bool PostProdProjectDefinition::getProcessSequence() const {
  VS_MUTEX_LOCK
  if (getOutputConfig()->has("process_sequence")) {
    return getOutputConfig()->get("process_sequence")->asBool();
  } else {
    return false;
  }
}

void PostProdProjectDefinition::setProcessSequence(const bool processSequence) {
  VS_MUTEX_LOCK
  if (processSequence) {
    getOutputConfig()->get("process_sequence")->asBool() = true;
  } else {
    if (getOutputConfig()->has("process_sequence")) {
      delete getOutputConfig()->remove("process_sequence");
    }
  }
  setModified();
}

bool PostProdProjectDefinition::hasAudioConfiguration() const {
  VS_MUTEX_LOCK
  return (getOutputConfig()->has("audio_bitrate") && getOutputConfig()->has("audio_codec") &&
          getOutputConfig()->has("sampling_rate") && getOutputConfig()->has("sample_format") &&
          getOutputConfig()->has("channel_layout"));
}

int PostProdProjectDefinition::getOutputAudioBitrate() const {
  VS_MUTEX_LOCK
  VideoStitch::Ptv::Value* outputConfig = getOutputConfig()->clone();
  if (outputConfig != nullptr && outputConfig->has("audio_bitrate")) {
    return outputConfig->get("audio_bitrate")->asInt();
  } else {
    return VideoStitch::AudioHelpers::DEFAULT_AUDIO_BITRATE;
  }
}

const QString PostProdProjectDefinition::getOutputAudioCodec() const {
  VS_MUTEX_LOCK
  VideoStitch::Ptv::Value* outputConfig = getOutputConfig()->clone();
  if (outputConfig != nullptr && outputConfig->has("audio_codec")) {
    return QString::fromStdString(outputConfig->get("audio_codec")->asString());
  } else {
    return QString();
  }
}

VideoStitch::Audio::SamplingDepth PostProdProjectDefinition::getOutputAudioSamplingFormat() const {
  VS_MUTEX_LOCK
  VideoStitch::Ptv::Value* outputConfig = getOutputConfig()->clone();
  if (outputConfig != nullptr && outputConfig->has("sample_format")) {
    return VideoStitch::Audio::getSamplingDepthFromString(outputConfig->get("sample_format")->asString().c_str());
  } else {
    return VideoStitch::Audio::SamplingDepth::SD_NONE;
  }
}

int PostProdProjectDefinition::getOutputAudioSamplingRate() const {
  if (getAudioPipeConst().get()) {
    return getAudioPipeConst()->getSamplingRate();
  } else {
    return 0;
  }
}

QString PostProdProjectDefinition::getOutputAudioChannels() const {
  // TODO: Implement the audio channel layout from the audio pipeline.
  return "Stereo";
}

void PostProdProjectDefinition::setOutputConfig(VideoStitch::Ptv::Value* outputConfig) {
  VS_MUTEX_LOCK
  if (outputConfig && *outputConfig != delegate->getOutputConfig()) {
    delegate->setOutputConfig(outputConfig);
    setModified();
  }
}

void PostProdProjectDefinition::updateAudioPipe(QString& oldSelectedAudio) {
  VS_MUTEX_LOCK
  if (getAudioPipeConst().get() != nullptr && getAudioPipeConst()->numAudioInputs() > 0) {
    // audio_selected
    // save selected audio
    oldSelectedAudio = QString::fromStdString(getAudioPipeConst()->getSelectedAudio());

    // resync audio pipe with inputs
    setAudioPipe(VideoStitch::Core::AudioPipeDefinition::createAudioPipeFromPanoInputs(getPanoConst().get()));
  }
}

void PostProdProjectDefinition::setOutputVideoConfig(VideoStitch::Ptv::Value* oldConfig,
                                                     VideoStitch::Ptv::Value* newConfig) {
  VS_MUTEX_LOCK
  VideoStitch::Ptv::Value* outputConfig = getOutputConfig()->clone();
  if (oldConfig != nullptr) {
    VideoStitch::Helper::PtvMerger::removeFrom(outputConfig, oldConfig);
  }
  VideoStitch::Helper::PtvMerger::mergeValue(outputConfig, newConfig);
  setOutputConfig(outputConfig);
}

void PostProdProjectDefinition::setFirstFrame(const frameid_t frame) {
  VS_MUTEX_LOCK
  if (getFirstFrame() != frame) {
    delegate->setFirstFrame(frame);

    if (frame > getLastFrame()) {
      delegate->setLastFrame(frame);
    }
    setModified();
    emit reqSetWorkingArea(delegate->getFirstFrame(), delegate->getLastFrame());
  }
}

void PostProdProjectDefinition::setLastFrame(const frameid_t frame) {
  VS_MUTEX_LOCK
  if (getLastFrame() != frame) {
    delegate->setLastFrame(frame);

    if (frame < getFirstFrame()) {
      delegate->setFirstFrame(frame);
    }
    setModified();
    emit reqSetWorkingArea(delegate->getFirstFrame(), delegate->getLastFrame());
  }
}

bool PostProdProjectDefinition::setDefaultValues(QList<VideoStitch::Ptv::Value*> userInputs) {
  PresetsManager* presetsManager = PresetsManager::getInstance();
  if (!presetsManager->hasPreset("project", "default_input") ||
      !presetsManager->hasPreset("project", "default_project")) {
    MsgBoxHandler::getInstance()->generic(tr("Cannot initialize the panorama: there might be something wrong with your "
                                             "%0 installation. Please reinstall %0 if the problem persists.")
                                              .arg(QCoreApplication::applicationName()),
                                          tr("Missing default presets"), CRITICAL_ERROR_ICON);
    return false;
  }

  // Configure inputs
  std::vector<VideoStitch::Ptv::Value*> inputVector;
  if (userInputs.isEmpty()) {
    inputVector.push_back(presetsManager->clonePresetContent("project", "default_input").release());
  } else {
    foreach (VideoStitch::Ptv::Value* input, userInputs) {
      std::unique_ptr<VideoStitch::Ptv::Value> defaultInput = defaultFromInput(input);
      if (!defaultInput) {
        return false;
      }
      inputVector.push_back(defaultInput.release());
    }
  }
  std::unique_ptr<VideoStitch::Ptv::Value> defaultProject =
      presetsManager->clonePresetContent("project", "default_project");
  defaultProject->get("pano")->asObject().get("inputs")->asList() = inputVector;

  // Configure default output
  bool videoOutput = false;
  for (const VideoStitch::Ptv::Value* input : inputVector) {
    if (input->has("reader_config")->getType() == VideoStitch::Ptv::Value::STRING) {
      const QString name(QString::fromStdString(input->has("reader_config")->asString()));
      if (File::getTypeFromFile(name) == File::VIDEO) {
        videoOutput = true;
        break;
      }
    }
  }

  VideoStitch::Ptv::Value* output = VideoStitch::Ptv::Value::emptyObject();
  output->get("type")->asString() = videoOutput ? "mp4" : "jpg";
  output->get("filename")->asString() = "vs-out";
  defaultProject->push("output", output);

  setDefaultPtvValues(*defaultProject);
  return true;
}

std::unique_ptr<VideoStitch::Ptv::Value> PostProdProjectDefinition::defaultFromInput(
    VideoStitch::Ptv::Value* input) const {
  QList<std::string> mandatoryFields = QList<std::string>() << "width"
                                                            << "height"
                                                            << "reader_config";
  foreach (std::string field, mandatoryFields) {
    if (!input->has(field)) {
      MsgBoxHandler::getInstance()->generic(tr("Missing mandatory field %0").arg(QString::fromStdString(field)),
                                            tr("Failed to initialize the default "), CRITICAL_ERROR_ICON);
      return std::unique_ptr<VideoStitch::Ptv::Value>();
    }
  }
  std::unique_ptr<VideoStitch::Ptv::Value> defaultInput =
      PresetsManager::getInstance()->clonePresetContent("project", "default_input");
  VideoStitch::Helper::PtvMerger::mergeValue(defaultInput.get(), input);
  return defaultInput;
}

void PostProdProjectDefinition::addInputs(QList<VideoStitch::Ptv::Value*> inputs) {
  std::unique_ptr<VideoStitch::Ptv::Value> projectSerialized(serialize());
  // When we change the number of inputs, the rig becomes invalid
  delete projectSerialized->get("pano")->remove("rig");
  delete projectSerialized->get("pano")->remove("cameras");

  foreach (VideoStitch::Ptv::Value* input, inputs) {
    std::unique_ptr<VideoStitch::Ptv::Value> defaultInput = defaultFromInput(input);
    if (!defaultInput) {
      emit reqReset(signalCompressor->add());
      return;
    }
    projectSerialized->get("pano")->asObject().get("inputs")->asList().push_back(defaultInput.release());
  }
  load(*projectSerialized);
  setModified();
  emit reqReset(signalCompressor->add());
}

int PostProdProjectDefinition::getOutputVideoBitrate() const {
  VS_MUTEX_LOCK
  if (getOutputConfig()->has("bitrate")) {
    return getOutputConfig()->get("bitrate")->asInt();
  } else {
    return 0;
  }
}

void PostProdProjectDefinition::setOutputAudioConfig(const QString codec, const int bitrate, const QString input) {
  VS_MUTEX_LOCK
  if (codec != getOutputAudioCodec() || bitrate != getOutputAudioBitrate()) {
    VideoStitch::Ptv::Value* outputConfig = getOutputConfig()->clone();
    const std::string& format =
        VideoStitch::Audio::getStringFromSamplingDepth(VideoStitch::Audio::SamplingDepth::FLT_P);
    const std::string& layout =
        VideoStitch::Audio::getStringFromChannelLayout(VideoStitch::Audio::ChannelLayout::STEREO);
    const int samplingRate =
        VideoStitch::AudioHelpers::getDefaultSamplingRate(VideoStitch::AudioHelpers::getCodecFromString(codec));
    outputConfig->get("audio_codec")->asString() = codec.toStdString();
    outputConfig->get("sample_format")->asString() = format;
    outputConfig->get("sampling_rate")->asInt() = samplingRate;
    outputConfig->get("channel_layout")->asString() = layout;
    outputConfig->get("audio_bitrate")->asInt() = bitrate;
    setOutputConfig(outputConfig);
  }
  if (input.toStdString() != getAudioPipeConst()->getSelectedAudio()) {
    getAudioPipe()->setSelectedAudio(input.toStdString());
    setModified(true);
  }  // No changes
}

void PostProdProjectDefinition::removeAudioSource() {
  VS_MUTEX_LOCK
  if (getOutputConfig()->has("audio_codec")) delete getOutputConfig()->remove("audio_codec");
  if (getOutputConfig()->has("sample_format")) delete getOutputConfig()->remove("sample_format");
  if (getOutputConfig()->has("sampling_rate")) delete getOutputConfig()->remove("sampling_rate");
  if (getOutputConfig()->has("channel_layout")) delete getOutputConfig()->remove("channel_layout");
  if (getOutputConfig()->has("audio_bitrate")) delete getOutputConfig()->remove("audio_bitrate");
  setModified();
}

void PostProdProjectDefinition::addOutputFileChunkSize(const int size) {
  VS_MUTEX_LOCK
  getOutputConfig()->get("max_video_file_chunk")->asInt() = size;
  setModified();
}

void PostProdProjectDefinition::removeOutputFileChunkSize() {
  VS_MUTEX_LOCK
  if (getOutputConfig()->has("max_video_file_chunk")) {
    getOutputConfig()->remove("max_video_file_chunk");
    setModified();
  }
}

void PostProdProjectDefinition::fixInputPaths() {
  VS_MUTEX_LOCK
  bool hasChanges = false;
  // inputs
  for (uint inputId = 0; inputId < (uint)getNumInputs(); ++inputId) {
    VideoStitch::Core::InputDefinition& iDef = getPanoConst()->getInput(inputId);
    if (iDef.getReaderConfig().getType() == VideoStitch::Ptv::Value::STRING) {
      const QString& fileName = QString::fromStdString(iDef.getReaderConfig().asString());
      const QString& normalizedPath = File::normalizePath(fileName);
      if (fileName != normalizedPath) {
        iDef.setFilename(normalizedPath.toStdString());
        hasChanges = true;
      }
    }
  }

  // audio_pipe
  if (hasChanges) {
    QString oldSelectedAudio;
    updateAudioPipe(oldSelectedAudio);
    if (oldSelectedAudio.isEmpty()) {
      const QString& normalizedPath = File::normalizePath(oldSelectedAudio);
      // set selected audio
      getAudioPipe()->setSelectedAudio(normalizedPath.toStdString());
    }
    setModified(true);
  }
}

void PostProdProjectDefinition::fixMissingInputs(QString& newFolder) {
  VS_MUTEX_LOCK
  bool hasChanges = false;

  QMap<int, QString> missingInputs;
  for (uint inputId = 0; inputId < (uint)getNumInputs(); ++inputId) {
    VideoStitch::Core::InputDefinition& iDef = getPanoConst()->getInput(inputId);
    if (iDef.getReaderConfig().getType() == VideoStitch::Ptv::Value::STRING) {
      const QString& filename = QString::fromStdString(iDef.getReaderConfig().asString());
      if (!QFile::exists(filename) && !filename.startsWith("procedural:")) {
        missingInputs.insert(inputId, filename);
      }
    }
  }

  // ask user to look for folder on the main thread (GUI dialog)
  if (!missingInputs.empty()) {
    hasChanges = true;
    emit reqMissingInputs(newFolder);
  }

  for (uint inputId = 0; inputId < (uint)getNumInputs(); ++inputId) {
    VideoStitch::Core::InputDefinition& iDef = getPanoConst()->getInput(inputId);
    const QString& newfilename = newFolder + QDir::separator() + QFileInfo(missingInputs.value(inputId)).fileName();
    if (missingInputs.contains(inputId)) {
      if (QFile::exists(newfilename)) {
        std::unique_ptr<VideoStitch::Input::DefaultReaderFactory> factory(
            new VideoStitch::Input::DefaultReaderFactory(0, NO_LAST_FRAME));
        std::unique_ptr<VideoStitch::Ptv::Value> val(VideoStitch::Ptv::Value::emptyObject());
        val->asString() = newfilename.toStdString();
        VideoStitch::Input::ProbeResult result = factory->probe(*val);
        if ((result.width != iDef.getWidth() || result.height != iDef.getHeight())) {
          emit reqWarnWrongInputSize(result.width, result.height, iDef.getWidth(), iDef.getHeight());
          for (uint inputId = 0; inputId < (uint)getNumInputs(); ++inputId) {
            VideoStitch::Core::InputDefinition& iDef = getPanoConst()->getInput(inputId);
            iDef.setFilename(
                QString(VS_DEFAULT_INPUT)
                    .arg(QColor::fromHsl((360 * inputId) / int(getNumInputs()), 255, 96).name().remove("#"))
                    .toStdString()
                    .c_str());
          }
          break;
        }
        iDef.setFilename(newfilename.toStdString());
      } else {
        iDef.setFilename(QString(VS_DEFAULT_INPUT)
                             .arg(QColor::fromHsl((360 * inputId) / int(getNumInputs()), 255, 96).name().remove("#"))
                             .toStdString()
                             .c_str());
      }
    }
  }

  // update audio_pipe
  if (hasChanges) {
    QString oldSelectedAudio;
    updateAudioPipe(oldSelectedAudio);
    if (oldSelectedAudio.isEmpty()) {
      const QString& selectedAudio = newFolder + QDir::separator() + QFileInfo(oldSelectedAudio).fileName();
      getAudioPipe()->setSelectedAudio(selectedAudio.toStdString());
    }
    setModified(true);
  }
}

void PostProdProjectDefinition::setOutputFilename(const QString filename) {
  VS_MUTEX_LOCK
  if (getOutputConfig()->get("filename")->asString() != filename.toStdString()) {
    getOutputConfig()->get("filename")->asString() = filename.toStdString();
    setOutputConfig(const_cast<VideoStitch::Ptv::Value*>(getOutputConfig().get()));
    setModified();
  }
}

void PostProdProjectDefinition::setPanoramaSize(unsigned width, unsigned height) {
  VS_MUTEX_LOCK
  const bool widthModified = getPanoConst()->getWidth() != width;
  const bool heightModified = getPanoConst()->getHeight() != height;
  if (widthModified) {
    getPano()->setWidth(width);
    setModified();
  }
  if (heightModified) {
    getPano()->setHeight(height);
    setModified();
  }
}

void PostProdProjectDefinition::checkRange(const frameid_t firstStitchableFrame, const frameid_t lastStitchableFrame) {
  if (getLastFrame() > lastStitchableFrame) {
    setLastFrame(lastStitchableFrame);
  }
  if (getFirstFrame() < firstStitchableFrame) {
    setFirstFrame(firstStitchableFrame);
  }
}

// ---------------------------- Curves ----------------------------------------

void PostProdProjectDefinition::resetOrientationCurve(bool resetController) {
  if (!isInit()) {
    return;
  }
  getPano()->replaceGlobalOrientation(new VideoStitch::Core::QuaternionCurve(
      VideoStitch::Core::SphericalSpline::point(0, VideoStitch::Quaternion<double>())));
  emit reqUpdateQuaternionCurve(getPano()->getGlobalOrientation().clone(), CurveGraphicsItem::GlobalOrientation);

  if (resetController) {
    emit reqReset(signalCompressor->add());
  }
}

void PostProdProjectDefinition::resetStabilizationCurve() {
  if (!isInit()) {
    return;
  }
  getPano()->resetStabilization();
  emit reqUpdateQuaternionCurve(getPano()->getStabilization().clone(), CurveGraphicsItem::Stabilization);
  setModified();
  emit reqReset(signalCompressor->add());
}

void PostProdProjectDefinition::resetPhotometricCalibration() {
  // we don't know which camera response was chosen before
  // let's keep an EMoR with zeros
  for (int i = 0; i < int(getNumInputs()); ++i) {
    getPano()->getInput(i).resetVignetting();
    getPano()->getInput(i).resetPhotoResponse();
  }

  setModified();
  emit reqReset(signalCompressor->add());
  emit reqRefreshPhotometry();
}

// FIXME: needs to be tested, needs to update the timeline, and cleaned from duplicates.
void PostProdProjectDefinition::resetCurves(bool resetController) {
  if (!isInit()) {
    return;
  }
  getPano()->resetExposureValue();
  emit reqUpdateCurve(getPano()->getExposureValue().clone(), CurveGraphicsItem::GlobalExposure);
  getPano()->resetRedCB();
  emit reqUpdateCurve(getPano()->getRedCB().clone(), CurveGraphicsItem::GlobalRedCorrection);
  getPano()->resetBlueCB();
  emit reqUpdateCurve(getPano()->getBlueCB().clone(), CurveGraphicsItem::GlobalBlueCorrection);

  resetOrientationCurve(false);
  resetStabilizationCurve();
  resetEvCurves();
  if (resetController) {
    emit reqReset(signalCompressor->add());
  }
}

void PostProdProjectDefinition::resetEvCurves(bool resetController) {
  for (int i = 0; i < int(getNumInputs()); ++i) {
    getPano()->getInput(i).resetGreenCB();
    getPano()->getInput(i).resetBlueCB();
    emit reqUpdateCurve(getPano()->getInput(i).getBlueCB().clone(), CurveGraphicsItem::BlueCorrection, i);
    getPano()->getInput(i).resetRedCB();
    emit reqUpdateCurve(getPano()->getInput(i).getRedCB().clone(), CurveGraphicsItem::RedCorrection, i);
    getPano()->getInput(i).resetExposureValue();
    emit reqUpdateCurve(getPano()->getInput(i).getExposureValue().clone(), CurveGraphicsItem::InputExposure, i);
  }
  if (resetController) {
    emit reqReset(signalCompressor->add());
  }
}

// FIXME: needs to be tested, needs to update the timeline, and cleaned from duplicates.
void PostProdProjectDefinition::resetCurve(CurveGraphicsItem::Type type, int inputId) {
  if (!isInit()) {
    return;
  }
  switch (type) {
    case CurveGraphicsItem::GlobalOrientation:
      getPano()->replaceGlobalOrientation(new VideoStitch::Core::QuaternionCurve(VideoStitch::Quaternion<double>()));
      emit reqUpdateQuaternionCurve(getPano()->getGlobalOrientation().clone(), type);
      break;
    case CurveGraphicsItem::Stabilization:
    case CurveGraphicsItem::StabilizationYaw:
    case CurveGraphicsItem::StabilizationPitch:
    case CurveGraphicsItem::StabilizationRoll:
      getPano()->resetStabilization();
      emit reqUpdateQuaternionCurve(getPano()->getStabilization().clone(), type);
      break;
    case CurveGraphicsItem::GlobalExposure:
      getPano()->resetExposureValue();
      emit reqUpdateCurve(getPano()->getExposureValue().clone(), type);
      break;
    case CurveGraphicsItem::GlobalBlueCorrection:
      getPano()->resetBlueCB();
      emit reqUpdateCurve(getPano()->getBlueCB().clone(), type);
      break;
    case CurveGraphicsItem::GlobalRedCorrection:
      getPano()->resetRedCB();
      emit reqUpdateCurve(getPano()->getRedCB().clone(), type);
      break;
    case CurveGraphicsItem::InputExposure:
      getPano()->getInput(inputId).resetExposureValue();
      emit reqUpdateCurve(getPano()->getInput(inputId).getExposureValue().clone(), type, inputId);
      break;
    case CurveGraphicsItem::BlueCorrection:
      getPano()->getInput(inputId).resetBlueCB();
      emit reqUpdateCurve(getPano()->getInput(inputId).getBlueCB().clone(), type, inputId);
      break;
    case CurveGraphicsItem::RedCorrection:
      getPano()->getInput(inputId).resetRedCB();
      emit reqUpdateCurve(getPano()->getInput(inputId).getRedCB().clone(), type, inputId);
      break;
    case CurveGraphicsItem::Unknown:
    default:
      Q_ASSERT(0);
      break;
  }
  setModified();
  emit reqReset(signalCompressor->add());
}

void PostProdProjectDefinition::curvesChanged(
    SignalCompressionCaps* comp, std::vector<std::pair<VideoStitch::Core::Curve*, CurveGraphicsItem::Type> > curves,
    std::vector<std::pair<VideoStitch::Core::QuaternionCurve*, CurveGraphicsItem::Type> > qcurves) {
  // Compress signals.
  if (comp && comp->pop() > 0) {
    for (std::vector<std::pair<VideoStitch::Core::Curve*, CurveGraphicsItem::Type> >::iterator i = curves.begin();
         i != curves.end(); ++i) {
      delete i->first;
    }
    for (std::vector<std::pair<VideoStitch::Core::QuaternionCurve*, CurveGraphicsItem::Type> >::iterator i =
             qcurves.begin();
         i != qcurves.end(); ++i) {
      delete i->first;
    }
    return;
  }
  for (std::vector<std::pair<VideoStitch::Core::Curve*, CurveGraphicsItem::Type> >::iterator i = curves.begin();
       i != curves.end(); ++i) {
    changeCurve(i->first, i->second, -1);
  }
  for (std::vector<std::pair<VideoStitch::Core::QuaternionCurve*, CurveGraphicsItem::Type> >::iterator i =
           qcurves.begin();
       i != qcurves.end(); ++i) {
    changeQuaternionCurve(i->first, i->second, -1);
  }
  applyCurvesChanges();
}

void PostProdProjectDefinition::curveChanged(SignalCompressionCaps* comp, VideoStitch::Core::Curve* curve,
                                             CurveGraphicsItem::Type type, int inputId) {
  // Compress signals.
  if (comp && comp->pop() > 0) {
    delete curve;
    return;
  }
  changeCurve(curve, type, inputId);
  applyCurvesChanges();
}

void PostProdProjectDefinition::quaternionCurveChanged(SignalCompressionCaps* comp,
                                                       VideoStitch::Core::QuaternionCurve* curve,
                                                       CurveGraphicsItem::Type type, int inputId) {
  // Compress signals.
  if (comp && comp->pop() > 0) {
    delete curve;
    return;
  }
  changeQuaternionCurve(curve, type, inputId);
  applyCurvesChanges();
}

void PostProdProjectDefinition::changeCurve(VideoStitch::Core::Curve* curve, CurveGraphicsItem::Type type,
                                            int inputId) {
  if (!isInit()) {
    delete curve;
    return;
  }
  switch (type) {
    case CurveGraphicsItem::GlobalExposure:
      getPano()->replaceExposureValue(curve);
      break;
    case CurveGraphicsItem::GlobalBlueCorrection:
      getPano()->replaceBlueCB(curve);
      break;
    case CurveGraphicsItem::GlobalRedCorrection:
      getPano()->replaceRedCB(curve);
      break;
    case CurveGraphicsItem::InputExposure:
      getPano()->getInput(inputId).replaceExposureValue(curve);
      break;
    case CurveGraphicsItem::RedCorrection:
      getPano()->getInput(inputId).replaceRedCB(curve);
      break;
    case CurveGraphicsItem::BlueCorrection:
      getPano()->getInput(inputId).replaceBlueCB(curve);
      break;
    case CurveGraphicsItem::GlobalOrientation: {
      VideoStitch::Core::QuaternionCurve* orientationCurve = getPano()->getGlobalOrientation().clone();

      VideoStitch::Core::Spline* spline = curve->splines();
      VideoStitch::Core::SphericalSpline* quaternionSplines = orientationCurve->splines();
      QSet<int> splineKF;
      QList<int> toRemove;

      if (spline) {
        for (spline = spline->next; spline; spline = spline->next) {
          splineKF << spline->end.t;
        }
      }
      if (quaternionSplines) {
        for (quaternionSplines = quaternionSplines->next; quaternionSplines;
             quaternionSplines = quaternionSplines->next) {
          if (!splineKF.contains(quaternionSplines->end.t)) {
            toRemove << quaternionSplines->end.t;
          }
        }
      }

      if (!toRemove.isEmpty()) {
        foreach (int kf, toRemove) { orientationCurve->mergeAt(kf); }
        changeQuaternionCurve(orientationCurve, CurveGraphicsItem::GlobalOrientation, -1);
      } else {
        delete orientationCurve;
      }
      delete curve;
      break;
    }
    case CurveGraphicsItem::Unknown:
    default:
      break;
  }
  setModified();
}

void PostProdProjectDefinition::changeQuaternionCurve(VideoStitch::Core::QuaternionCurve* curve,
                                                      CurveGraphicsItem::Type type, int inputId) {
  Q_UNUSED(inputId)
  if (!isInit()) {
    delete curve;
    return;
  }
  switch (type) {
    case CurveGraphicsItem::GlobalOrientation:
      getPano()->replaceGlobalOrientation(curve);
      break;
    case CurveGraphicsItem::Stabilization:
      getPano()->replaceStabilization(curve);
      break;
    case CurveGraphicsItem::Unknown:
    default:
      break;
  }
  setModified();
}

void PostProdProjectDefinition::applyCurvesChanges() {
  emit reqReset(signalCompressor->add());
  emit reqRefreshCurves();
}

void PostProdProjectDefinition::resetEvCurvesSequence(const frameid_t startPoint, const frameid_t endPoint) {
  if (!isInit()) {
    return;
  }
  for (int i = 0; i < int(getNumInputs()); ++i) {
    getPano()->getInput(i).replaceExposureValue(
        resetEvCurve(startPoint, endPoint, 0, getPano()->getInput(i).getExposureValue().clone()));
    emit reqUpdateCurve(getPano()->getInput(i).getExposureValue().clone(), CurveGraphicsItem::InputExposure, i);
    getPano()->getInput(i).replaceBlueCB(
        resetEvCurve(startPoint, endPoint, 1, getPano()->getInput(i).getBlueCB().clone()));
    emit reqUpdateCurve(getPano()->getInput(i).getBlueCB().clone(), CurveGraphicsItem::BlueCorrection, i);
    getPano()->getInput(i).replaceRedCB(
        resetEvCurve(startPoint, endPoint, 1, getPano()->getInput(i).getRedCB().clone()));
    emit reqUpdateCurve(getPano()->getInput(i).getRedCB().clone(), CurveGraphicsItem::RedCorrection, i);
  }
  setModified();
  emit reqReset(signalCompressor->add());
}

VideoStitch::Core::Curve* PostProdProjectDefinition::resetEvCurve(frameid_t startPoint, frameid_t endPoint,
                                                                  double value,
                                                                  const VideoStitch::Core::Curve* curveToReset) {
  if (!isInit()) {
    delete curveToReset;
    return nullptr;
  }

  VideoStitch::Core::Curve* ret;
  if (startPoint == 0 && endPoint == -1) {
    ret = new VideoStitch::Core::Curve(VideoStitch::Core::Spline::point(0, value));
  } else {
    ret = curveToReset->clone();
  }

  for (const VideoStitch::Core::Spline* spline = curveToReset->splines(); spline != nullptr; spline = spline->next) {
    if (spline->end.t >= startPoint && spline->end.t <= endPoint) {
      ret->mergeAt(spline->end.t);
    }
  }
  delete curveToReset;
  return ret;
}
