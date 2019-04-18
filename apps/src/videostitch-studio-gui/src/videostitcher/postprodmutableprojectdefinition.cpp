// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "postprodmutableprojectdefinition.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"

#include <memory>

frameid_t PostProdMutableProjectDefinition::getFirstFrame() const { return firstFrame; }

frameid_t PostProdMutableProjectDefinition::getLastFrame() const { return lastFrame; }

void PostProdMutableProjectDefinition::setFirstFrame(frameid_t f) { firstFrame = f; }

void PostProdMutableProjectDefinition::setLastFrame(frameid_t f) { lastFrame = f; }

VideoStitch::Ptv::Value& PostProdMutableProjectDefinition::getOutputConfig() const { return *outputConfig; }

void PostProdMutableProjectDefinition::setOutputConfig(VideoStitch::Ptv::Value* o) { outputConfig.reset(o); }

VideoStitch::Ptv::Value* PostProdMutableProjectDefinition::serialize() const {
  VideoStitch::Ptv::Value* root = MutableProjectDefinition::serialize();
  root->get("first_frame")->asInt() = firstFrame;
  root->get("last_frame")->asInt() = lastFrame;
  delete root->push("output", outputConfig->clone());
  return root;
}

PostProdMutableProjectDefinition* PostProdMutableProjectDefinition::create(const VideoStitch::Ptv::Value& value) {
  int firstFrame, lastFrame = 1;
  if (VideoStitch::Parse::populateInt("PostProdMutableProjectDefinition", value, "first_frame", firstFrame, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    return nullptr;
  }
  if (VideoStitch::Parse::populateInt("PostProdMutableProjectDefinition", value, "last_frame", lastFrame, true) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    return nullptr;
  }
  if (lastFrame == -1) {  // We have this when loading the default preset
    lastFrame = NO_LAST_FRAME;
  }
  const VideoStitch::Ptv::Value* tmp = value.has("output");
  if (!tmp) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Missing field 'output' for BaseProjectDefinition.");
    return nullptr;
  }
  std::unique_ptr<VideoStitch::Ptv::Value> outputConfig(tmp->clone());

  MutableProjectDefinition* parent = MutableProjectDefinition::create(value);
  if (parent == nullptr) {
    return nullptr;
  }
  PostProdMutableProjectDefinition* that =
      new PostProdMutableProjectDefinition(firstFrame, lastFrame, std::move(outputConfig), parent);
  delete parent;
  return that;
}

PostProdMutableProjectDefinition::PostProdMutableProjectDefinition(
    int firstFrame, int lastFrame, std::unique_ptr<VideoStitch::Ptv::Value> outputConfig,
    MutableProjectDefinition* parent)
    : MutableProjectDefinition(*parent),
      firstFrame(firstFrame),
      lastFrame(lastFrame),
      outputConfig(std::move(outputConfig)) {}
