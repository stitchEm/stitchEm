// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QMap>

class LiveOutputFactory;

class LiveOutputList {
 public:
  explicit LiveOutputList();

  void addOutput(LiveOutputFactory* output);

  void removeOutput(const QString& id);

  void replaceOutput(const QString& existingOne);

  LiveOutputFactory* getOutput(const QString& id) const;

  void clearOutput();

  bool isEmpty() const;

  unsigned int activeOutputs() const;

  QList<LiveOutputFactory*> getValues() const;

 private:
  QMap<QString, LiveOutputFactory*> map;
};
