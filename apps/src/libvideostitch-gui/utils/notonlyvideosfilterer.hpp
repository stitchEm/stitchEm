// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "featurefilterer.hpp"

/**
 * @brief This class allows to disable features that are not compatible with projects that have
 * only 1 video
 * or at least 1 image.
 *
 * To use it on 'object', just do this (in the main thread):
 * NotOnlyVideosFilterer::getInstance()->watch(object);
 */
class VS_GUI_EXPORT NotOnlyVideosFilterer : public FeatureFilterer {
  Q_OBJECT

 public:
  // Call this only from the main thread
  static NotOnlyVideosFilterer* getInstance();

 protected:
  virtual void connectProjectConditionToUpdate();
  virtual bool disableFeaturesCondition() const;

 private slots:
  void internalUpdateWatchedObjects(bool hasSeveralVideos);

 private:
  NotOnlyVideosFilterer();
  ~NotOnlyVideosFilterer() = default;
  Q_DISABLE_COPY(NotOnlyVideosFilterer)
};
