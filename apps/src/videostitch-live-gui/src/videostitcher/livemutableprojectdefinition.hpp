// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/videostitcher/mutableprojectdefinition.hpp"

#include <memory>

class LiveExposure;
class LiveOutputFactory;
class LiveOutputList;

std::unique_ptr<LiveExposure> createExposure(const VideoStitch::Ptv::Value* config);

class LiveMutableProjectDefinition : public MutableProjectDefinition {
 public:
  static LiveMutableProjectDefinition* create(const VideoStitch::Ptv::Value& value);
  ~LiveMutableProjectDefinition();
  virtual VideoStitch::Ptv::Value* serialize() const;
  LiveOutputList* getOutputs() const;
  LiveExposure& getExposureConfiguration() const;
  void removeOutput(const QString& output);

 private:
  LiveMutableProjectDefinition(const LiveMutableProjectDefinition&) = delete;
  LiveMutableProjectDefinition& operator=(const LiveMutableProjectDefinition&) = delete;
  LiveMutableProjectDefinition(std::unique_ptr<LiveOutputList> outputs, std::unique_ptr<LiveExposure> exposure,
                               MutableProjectDefinition* parent);

  std::unique_ptr<LiveOutputList> outputsList;
  std::unique_ptr<LiveExposure> exposureConfig;
};
