// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LIVEINPUTFACTORY_HPP
#define LIVEINPUTFACTORY_HPP

#include "libvideostitch/object.hpp"
#include "libvideostitch-gui/utils/inputformat.hpp"
#include "utils/pixelformat.hpp"
#include <QVector>
#include <memory>

class LiveInputFactory : public VideoStitch::Ptv::Object {
 public:
  static LiveInputFactory* makeLiveInput(const VideoStitch::InputFormat::InputFormatEnum choice, const QString& name);
  static LiveInputFactory* makeLiveInput(const VideoStitch::InputFormat::InputFormatEnum choice,
                                         const VideoStitch::Ptv::Value* initializationInput);

 public:
  explicit LiveInputFactory(const QString& name);
  ~LiveInputFactory();

  virtual VideoStitch::InputFormat::InputFormatEnum getType() const = 0;
  virtual void initializeWith(const VideoStitch::Ptv::Value* initializationInput);
  const QString getName() const;

  void setName(QString newName);

 protected:
  QString name;
};

typedef QVector<std::shared_ptr<LiveInputFactory>> LiveInputList;

#endif  // LIVEINPUT_HPP
