// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videostitcher/postprodprojectdefinition.hpp"

#include <QWidget>

/**
 * @brief Base class for an ouput configuration
 */
class IProcessWidget : public QWidget {
  Q_OBJECT
 public:
  explicit IProcessWidget(QWidget* const parent = nullptr);
  virtual ~IProcessWidget() = 0;

 public slots:
  void setProject(PostProdProjectDefinition* projectDef);
  void clearProject();

 signals:
  void notifyConfigurationChanged();

 protected:
  virtual void reactToChangedProject() {}
  virtual void reactToClearedProject() {}
  QPointer<PostProdProjectDefinition> project;
};
