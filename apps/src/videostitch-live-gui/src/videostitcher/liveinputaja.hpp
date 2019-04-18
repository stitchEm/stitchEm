// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEINPUTAJA_HPP
#define LIVEINPUTAJA_HPP

#include "capturecardliveinput.hpp"

class LiveInputAJA : public CaptureCardLiveInput {
 public:
  explicit LiveInputAJA(const QString& name);
  ~LiveInputAJA();

  virtual VideoStitch::InputFormat::InputFormatEnum getType() const;
  virtual VideoStitch::Ptv::Value* serialize() const;
  virtual void initializeWith(const VideoStitch::Ptv::Value* initializationInput);

  void setAudioIsEnabled(bool audio);

 private:
  int nameToDevice(const QString name) const;
  int nameToChannel(const QString name) const;

  bool audioIsEnabled = false;
};

#endif  // LIVEINPUTAJA_HPP
