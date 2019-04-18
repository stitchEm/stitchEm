// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "featurefilterer.hpp"

/**
 * @brief This class allows to disable features that are not compatible with images|procedurals only projects.
 *
 * To use it on 'object', just do this (in the main thread):
 * ImagesOrProceduralsOnlyFilterer::getInstance()->watch(object);
 */
class VS_GUI_EXPORT ImagesOrProceduralsOnlyFilterer : public FeatureFilterer {
  Q_OBJECT

 public:
  // Call this only from the main thread
  static ImagesOrProceduralsOnlyFilterer* getInstance();

 protected:
  virtual void connectProjectConditionToUpdate();
  virtual bool disableFeaturesCondition() const;

 private:
  ImagesOrProceduralsOnlyFilterer();
  ~ImagesOrProceduralsOnlyFilterer() = default;
  Q_DISABLE_COPY(ImagesOrProceduralsOnlyFilterer)
};
