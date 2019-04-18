// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "featurefilterer.hpp"

/**
 * @brief This class allows to disable features that are not compatible with projects that need
 * several visual inputs (videos, images or procedurals).
 *
 * To use it on 'object', just do this (in the main thread):
 * OneVisualInputFilterer::getInstance()->watch(object);
 */
class VS_GUI_EXPORT OneVisualInputFilterer : public FeatureFilterer {
  Q_OBJECT

 public:
  // Call this only from the main thread
  static OneVisualInputFilterer* getInstance();

 protected:
  virtual void connectProjectConditionToUpdate();
  virtual bool disableFeaturesCondition() const;

 private slots:
  void internalUpdateWatchedObjects(bool hasSeveralVisualInputs);

 private:
  OneVisualInputFilterer();
  ~OneVisualInputFilterer() = default;
  Q_DISABLE_COPY(OneVisualInputFilterer)
};
