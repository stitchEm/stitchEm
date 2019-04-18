// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "liveprojectdefinition.hpp"
#include "liveoutputfile.hpp"
#include "liveoutputrtmp.hpp"
#include "liveoutputlist.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/base/ptvMerger.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include "libvideostitch-gui/videostitcher/presetsmanager.hpp"
#include "videostitcher/liveaudioprocessordelay.hpp"

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/rigDef.hpp"

#include <iostream>

static const unsigned int MAX_FACTOR(20u);
static const unsigned int FACTORS(4u);

LiveProjectDefinition::LiveProjectDefinition() : ProjectDefinition(), delegate(nullptr) {}

LiveProjectDefinition::~LiveProjectDefinition() {}

LiveOutputList* LiveProjectDefinition::getOutputConfigs() const { return delegate ? delegate->getOutputs() : nullptr; }

QStringList LiveProjectDefinition::getOutputNames() const {
  QStringList outputs;
  for (auto output : getOutputConfigs()->getValues()) {
    outputs << output->getIdentifier();
  }
  return outputs;
}

void LiveProjectDefinition::updateSize(int width, int height) {
  if (delegate) {
    ProjectDefinition::updateSize(width, height);
    for (LiveOutputFactory* output : delegate->getOutputs()->getValues()) {
      output->updateForPanoSizeChange(width, height);
    }
  }
}

LiveOutputFactory* LiveProjectDefinition::getOutputById(const QString& id) const {
  return getOutputConfigs()->getOutput(id);
}

bool LiveProjectDefinition::areActiveOutputs() const {
  return isInit() ? getOutputConfigs()->activeOutputs() > 0 : false;
}

void LiveProjectDefinition::updateOutputId(const QString& id) { getOutputConfigs()->replaceOutput(id); }

bool LiveProjectDefinition::canAddOutput(const LiveOutputFactory* output) const {
  // Repeated output names are not allowed
  return getOutputById(output->getIdentifier()) == nullptr;
}

void LiveProjectDefinition::clearOutputs() {
  if (delegate != nullptr) {
    getOutputConfigs()->clearOutput();
  }
}

LiveMutableProjectDefinition* LiveProjectDefinition::getDelegate() const { return delegate.data(); }

void LiveProjectDefinition::createDelegate(const VideoStitch::Ptv::Value& value) {
  delegate.reset(LiveMutableProjectDefinition::create(value));
}

void LiveProjectDefinition::destroyDelegate() { delegate.reset(); }

bool LiveProjectDefinition::updateInputs(LiveInputList inputs, const AudioConfiguration& audioConfiguration) {
  const PresetsManager* presetsManager = PresetsManager::getInstance();
  if (!presetsManager->hasPreset("project", "default_input") ||
      !presetsManager->hasPreset("project", "default_vah_project")) {
    return false;
  }

  // When we change the number of inputs, the rig becomes invalid
  if (getPanoConst().get() && inputs.count() != getPanoConst()->numVideoInputs()) {
    getPano()->setCalibrationRigPresets(new VideoStitch::Core::RigDefinition());
    getPano()->setCalibrationControlPointList(VideoStitch::Core::ControlPointList());
  }

  std::vector<VideoStitch::Ptv::Value*> inputValues = serializeAndInitializeInputs(inputs);
  if (audioConfiguration.isValid() &&
      !audioConfiguration.isAja()) {  // Aja is treated differently because it is not supported by PortAudio
    inputValues.push_back(audioConfiguration.serialize());
  }

  std::unique_ptr<VideoStitch::Ptv::Value> project;
  bool wasInit = isInit();
  if (wasInit) {
    project.reset(delegate->serialize());
  } else {
    project = presetsManager->clonePresetContent("project", "default_vah_project");
  }

  project->get("pano")->asObject().get("inputs")->asList() = inputValues;

  load(*project);
  if (wasInit) {
    setModified();
  }
  return true;
}

void LiveProjectDefinition::updateAudioInput(const AudioConfiguration& audioConfiguration) {
  Q_ASSERT(isInit());

  // Remove old audio configuration
  const audioreaderid_t nbAudioInputs = getPanoConst()->numAudioInputs();
  for (audioreaderid_t audioIndex = nbAudioInputs - 1; audioIndex >= 0; --audioIndex) {
    VideoStitch::Core::InputDefinition& input = getPano()->getAudioInput(audioIndex);
    if (input.getIsVideoEnabled()) {
      input.setIsAudioEnabled(false);
    } else {
      const readerid_t inputIndex = getPanoConst()->convertAudioInputIndexToInputIndex(audioIndex);
      getPano()->removeInput(inputIndex);
    }
  }

  // Add the new audio configuration
  if (audioConfiguration.isValid()) {
    bool audioConfigured = false;
    const videoreaderid_t nbVideoInputs = getPanoConst()->numVideoInputs();
    for (videoreaderid_t videoIndex = 0; videoIndex < nbVideoInputs; ++videoIndex) {
      VideoStitch::Core::InputDefinition& input = getPano()->getVideoInput(videoIndex);
      if (input.getDisplayName() == audioConfiguration.inputName) {
        input.setIsAudioEnabled(true);
        audioConfigured = true;
        break;
      }
    }

    if (!audioConfigured) {
      std::unique_ptr<VideoStitch::Ptv::Value> audioInput(audioConfiguration.serialize());
      getPano()->insertInput(VideoStitch::Core::InputDefinition::create(*audioInput), getPanoConst()->numInputs());
    }
  }

  setAudioPipe(VideoStitch::Core::AudioPipeDefinition::createAudioPipeFromPanoInputs(getPanoConst().get()));
  setModified();
}

std::vector<VideoStitch::Ptv::Value*> LiveProjectDefinition::serializeAndInitializeInputs(LiveInputList newList) const {
  const PresetsManager* presetsManager = PresetsManager::getInstance();
  if (!presetsManager->hasPreset("project", "default_input")) {
    return std::vector<VideoStitch::Ptv::Value*>();
  }

  std::vector<VideoStitch::Ptv::Value*> newInputList;
  int groupId = 0;

  for (auto newInput : newList) {
    std::unique_ptr<VideoStitch::Ptv::Value> newInputValue =
        presetsManager->clonePresetContent("project", "default_input");
    newInputValue->get("group")->asInt() = groupId;
    std::unique_ptr<VideoStitch::Ptv::Value> newConfig(newInput->serialize());

    // If we're modifying the project, we want to retrieve the previous input data
    if (isInit()) {
      const videoreaderid_t nbVideoInputs = getPanoConst()->numVideoInputs();
      for (videoreaderid_t videoInputIndex = 0; videoInputIndex < nbVideoInputs; ++videoInputIndex) {
        const VideoStitch::Ptv::Value& readerConfig = getPanoConst()->getVideoInput(videoInputIndex).getReaderConfig();
        if (*newConfig->has("reader_config") == readerConfig) {
          newInputValue.reset(getPanoConst()->getVideoInput(videoInputIndex).serialize());
          break;
        }
      }
    }

    VideoStitch::Helper::PtvMerger::mergeValue(newInputValue.get(), newConfig.get());
    newInputList.push_back(newInputValue.release());
    ++groupId;
  }
  return newInputList;
}

bool LiveProjectDefinition::addOutput(LiveOutputFactory* output) {
  if (!canAddOutput(output)) {
    return false;
  }
  delegate->getOutputs()->addOutput(output);
  setModified();
  return true;
}

void LiveProjectDefinition::deleteOutput(const QString& id) {
  VideoStitch::Ptv::Value* root = delegate->serialize();
  std::vector<VideoStitch::Ptv::Value*> outputsList = root->get("outputs")->asList();
  for (unsigned int i = 0; i < outputsList.size(); ++i) {
    if (outputsList.at(i)->get("filename")->asString() == id.toStdString()) {
      outputsList.erase(outputsList.begin() + i);
    }
  }
  delegate->removeOutput(id);
  VideoStitch::Ptv::Value* rootOriginal = delegate->serialize();
  rootOriginal->get("outputs")->asList() = outputsList;
  setModified();
}

LiveInputFactory* LiveProjectDefinition::retrieveConfigurationInput() const {
  for (int inputIndex = 0; inputIndex < (int)getNumInputs(); ++inputIndex) {
    const VideoStitch::Core::InputDefinition& inputDefinition = getPano()->getInput(inputIndex);
    if (inputDefinition.getIsVideoEnabled()) {
      std::unique_ptr<VideoStitch::Ptv::Value> input(inputDefinition.serialize());
      // We don't want audio parameter in the configuration input
      delete input->remove("audio_enabled");
      return LiveInputFactory::makeLiveInput(getVideoInputType(), input.get());
    }
  }
  return nullptr;
}

LiveInputList LiveProjectDefinition::retrieveVideoInputs() const {
  LiveInputList inputs;
  VideoStitch::InputFormat::InputFormatEnum videoType = getVideoInputType();
  size_t nbInputs = getNumInputs();
  for (size_t inputIndex = 0; inputIndex < nbInputs; ++inputIndex) {
    const VideoStitch::Core::InputDefinition& inputDefinition = getPanoConst()->getInput(readerid_t(inputIndex));
    if (inputDefinition.getIsVideoEnabled()) {
      std::unique_ptr<VideoStitch::Ptv::Value> inputValue(inputDefinition.serialize());
      std::shared_ptr<LiveInputFactory> input(LiveInputFactory::makeLiveInput(videoType, inputValue.get()));
      inputs.append(input);
    }
  }
  return inputs;
}

AudioConfiguration LiveProjectDefinition::getAudioConfiguration() const {
  const PanoDefinitionLocked& panoConst = getPanoConst();
  if (panoConst.get()) {
    for (readerid_t index = 0; index < panoConst->numInputs(); ++index) {
      const VideoStitch::Core::InputDefinition& inputDefinition = panoConst->getInput(index);
      if (inputDefinition.getIsAudioEnabled()) {
        return AudioConfiguration(inputDefinition.getReaderConfig());
      }
    }
  }

  return AudioConfiguration();
}

int LiveProjectDefinition::getInputIndexForAudio() const {
  const PanoDefinitionLocked& panoConst = getPanoConst();
  for (readerid_t index = 0; index < panoConst->numInputs(); ++index) {
    const VideoStitch::Core::InputDefinition& inputDef = panoConst->getInput(index);
    if (inputDef.getIsAudioEnabled()) {
      return int(index);
    }
  }
  return -1;
}

QVector<unsigned int> LiveProjectDefinition::getDownsampledFactors() const {
  QVector<unsigned int> factors;
  unsigned int factor = 1u;
  while (factors.size() != FACTORS && factor < MAX_FACTOR) {
    if ((getPanoConst()->getWidth() % (2 * factor) == 0) && (getPanoConst()->getHeight() % (2 * factor) == 0)) {
      factors.push_back(factor);
    }
    ++factor;
  }
  return factors;
}

QString LiveProjectDefinition::getOutputDisplayableString(unsigned int factor) const {
  if (factor == 1) {
    return QString("%0x%1").arg(getPanoConst()->getWidth()).arg(getPanoConst()->getHeight());
  } else {
    //: Display a reduced output size (%0 for width, %1 for height, %2 for factor)
    return tr("%0x%1 (reduced by %2)")
        .arg(getPanoConst()->getWidth() / factor)
        .arg(getPanoConst()->getHeight() / factor)
        .arg(factor);
  }
}

bool LiveProjectDefinition::isDeviceInUse(const QString& type, const QString& name) const {
  if (type != VideoStitch::InputFormat::getStringFromEnum(getVideoInputType())) {
    return false;
  }
  // Check in the inputs list
  if (getInputNames().indexOf(name) >= 0) {
    return true;
  }

  // Check in the outputs list
  if (getOutputNames().indexOf(name) >= 0) {
    return true;
  }
  return false;
}

bool LiveProjectDefinition::updateAudioProcessor(LiveAudioProcessFactory* liveProcessor) {
  const VideoStitch::AudioProcessors::ProcessorEnum type = liveProcessor->getType();
  const std::string& name = VideoStitch::AudioProcessors::getStringFromEnum(liveProcessor->getType()).toStdString();
  VideoStitch::PotentialValue<VideoStitch::Core::AudioProcessorDef*> processor = getAudioPipe()->getProcessor(name);
  const bool alreadyAdded = processor.ok();
  // No processor, adding a new one
  if (type == VideoStitch::AudioProcessors::ProcessorEnum::DELAY) {
    if (alreadyAdded) {
      LiveAudioProcessorDelay* delayProcessor = dynamic_cast<LiveAudioProcessorDelay*>(liveProcessor);
      setAudioDelay(delayProcessor->getDelay());
      return true;
    }
    if (getAudioPipeConst()->numAudioInputs() > 0) {
      return getAudioPipe()->addDelayProcessor(getAudioPipeConst()->getInput(0)->getName(), 0.0).ok();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool LiveProjectDefinition::removeAudioProcessor(const QString& name) {
  return getAudioPipe()->removeProcessor(name.toStdString()).ok();
}

bool LiveProjectDefinition::setAudioProcessorConfiguration(LiveAudioProcessFactory* liveProcessor) {
  const std::string& name = VideoStitch::AudioProcessors::getStringFromEnum(liveProcessor->getType()).toStdString();
  VideoStitch::PotentialValue<VideoStitch::Core::AudioProcessorDef*> processor = getAudioPipe()->getProcessor(name);
  if (processor.ok()) {
    liveProcessor->serializeParameters(processor.ref()->getParameters(liveProcessor->getInputName()));
    return true;
  }
  return false;
}

QList<LiveAudioProcessFactory*> LiveProjectDefinition::getAudioProcessors() const {
  QList<LiveAudioProcessFactory*> list;
  for (auto i = 0; i < int(getAudioPipeConst()->numProcessors()); ++i) {
    const VideoStitch::PotentialValue<VideoStitch::Core::AudioProcessorDef*> processor =
        getAudioPipeConst()->getProcessor(i);
    const VideoStitch::AudioProcessors::ProcessorEnum type =
        VideoStitch::AudioProcessors::getEnumFromString(QString::fromStdString(processor.ref()->getName()));
    if (type != VideoStitch::AudioProcessors::ProcessorEnum::UNKNOWN) {
      list.push_back(LiveAudioProcessFactory::createProcessor(processor.ref()->getParameters(), type));
    }
  }
  return list;
}
