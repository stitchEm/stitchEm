// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef POSTPRODMUTABLEPROJECTDEFINITION_H
#define POSTPRODMUTABLEPROJECTDEFINITION_H

#include "libvideostitch-gui/videostitcher/mutableprojectdefinition.hpp"
#include <memory>

class PostProdMutableProjectDefinition : public MutableProjectDefinition {
 public:
  static PostProdMutableProjectDefinition* create(const VideoStitch::Ptv::Value& value);

  virtual VideoStitch::Ptv::Value* serialize() const;

  frameid_t getFirstFrame() const;
  frameid_t getLastFrame() const;

  void setFirstFrame(frameid_t);
  void setLastFrame(frameid_t);

  VideoStitch::Ptv::Value& getOutputConfig() const;
  void setOutputConfig(VideoStitch::Ptv::Value*);

 private:
  frameid_t firstFrame;
  frameid_t lastFrame;

  std::unique_ptr<VideoStitch::Ptv::Value> outputConfig;

  PostProdMutableProjectDefinition(int firstFrame, int lastFrame, std::unique_ptr<VideoStitch::Ptv::Value> outputConfig,
                                   MutableProjectDefinition* parent);
};

#endif  // POSTPRODMUTABLEPROJECTDEFINITION_H
