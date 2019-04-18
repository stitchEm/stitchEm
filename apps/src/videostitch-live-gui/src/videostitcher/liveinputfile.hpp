// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEINPUTFILE_H
#define LIVEINPUTFILE_H

#include "liveinputfactory.hpp"

class LiveInputFile : public LiveInputFactory {
 public:
  explicit LiveInputFile(const QString& name);
  ~LiveInputFile();

  virtual VideoStitch::Ptv::Value* serialize() const;
  virtual void initializeWith(const VideoStitch::Ptv::Value* initializationInput);
  virtual VideoStitch::InputFormat::InputFormatEnum getType() const;

  int getWidth() const;
  int getHeight() const;
  bool getHasAudio() const;

  void setWidth(int newWidth);
  void setHeight(int newHeight);
  void setHasAudio(bool audio);

 private:
  int width;
  int height;
  bool hasAudio;
};

#endif  // LIVEINPUTFILE_H
