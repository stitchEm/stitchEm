// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <memory>
#include "livemutableprojectdefinition.hpp"
#include "libvideostitch-base/logmanager.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "liveoutputfactory.hpp"
#include "liveoutputlist.hpp"
#include "exposure/liveexposure.hpp"

std::unique_ptr<LiveExposure> createExposure(const VideoStitch::Ptv::Value* config) {
  std::unique_ptr<LiveExposure> liveExposure(new LiveExposure());
  const VideoStitch::Ptv::Value* expoConfig = config->has("exposure");
  if (expoConfig == nullptr) {
    return liveExposure;
  }
  std::string algo;
  if (VideoStitch::Parse::populateString("LiveProjectDefinition", *expoConfig, "algorithm", algo, false) ==
      VideoStitch::Parse::PopulateResult_WrongType) {
    return liveExposure;
  }
  int anchor = -1;
  if (VideoStitch::Parse::populateInt("LiveProjectDefinition", *expoConfig, "anchor", anchor, false) ==
      VideoStitch::Parse::PopulateResult_WrongType) {
    return liveExposure;
  }
  bool isAuto = false;
  if (VideoStitch::Parse::populateBool("LiveProjectDefinition", *expoConfig, "auto", isAuto, false) ==
      VideoStitch::Parse::PopulateResult_WrongType) {
    return liveExposure;
  }
  liveExposure->setAlgorithm(QString::fromStdString(algo));
  liveExposure->setAnchor(anchor);
  liveExposure->setIsAutoStart(isAuto);
  return liveExposure;
}

LiveMutableProjectDefinition::LiveMutableProjectDefinition(std::unique_ptr<LiveOutputList> outputs,
                                                           std::unique_ptr<LiveExposure> exposure,
                                                           MutableProjectDefinition* parent)
    : MutableProjectDefinition(*parent), outputsList(std::move(outputs)), exposureConfig(std::move(exposure)) {}

LiveMutableProjectDefinition::~LiveMutableProjectDefinition() {
  // We can't use the default inline destructor because smart pointers must know the type of their object to delete them
}

LiveOutputList* LiveMutableProjectDefinition::getOutputs() const { return outputsList.get(); }

LiveExposure& LiveMutableProjectDefinition::getExposureConfiguration() const { return *exposureConfig; }

void LiveMutableProjectDefinition::removeOutput(const QString& output) { outputsList->removeOutput(output); }

VideoStitch::Ptv::Value* LiveMutableProjectDefinition::serialize() const {
  VideoStitch::Ptv::Value* root = MutableProjectDefinition::serialize();

  // Serialize the exposure
  root->push("exposure", exposureConfig->serialize());

  std::vector<VideoStitch::Ptv::Value*> outputs;
  // Serialize all outputs
  for (LiveOutputFactory* output : outputsList->getValues()) {
    outputs.push_back(output->serialize());
  }
  root->get("outputs")->asList() = outputs;
  return root;
}

LiveMutableProjectDefinition* LiveMutableProjectDefinition::create(const VideoStitch::Ptv::Value& value) {
  // Exposure configuration
  if (!value.has("exposure")) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        "Missing field 'exposure' for BaseProjectDefinition.");
  }
  std::unique_ptr<LiveExposure> exposure(createExposure(&value));

  // Rest of the common configuration
  QScopedPointer<MutableProjectDefinition> parent(MutableProjectDefinition::create(value));
  if (parent == nullptr) {
    return nullptr;
  }

  // Outputs configuration
  if (!value.has("outputs")) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        "Missing field 'outputs' for BaseProjectDefinition.");
    return nullptr;
  }
  std::vector<VideoStitch::Ptv::Value*> outputs = value.has("outputs")->asList();
  std::unique_ptr<LiveOutputList> list(new LiveOutputList());
  // Build the outputs that the ui will manage based on the PTV configs
  for (const VideoStitch::Ptv::Value* outputConfig : outputs) {
    list->addOutput(LiveOutputFactory::createOutput(outputConfig->clone(), &parent->getPanoDefinition()));
  }

  LiveMutableProjectDefinition* that =
      new LiveMutableProjectDefinition(std::move(list), std::move(exposure), parent.data());
  return that;
}
