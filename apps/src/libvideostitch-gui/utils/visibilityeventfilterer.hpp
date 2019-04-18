// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QObject>

#include <functional>

class VS_GUI_EXPORT VisibilityEventFilterer : public QObject {
  Q_OBJECT
 public:
  VisibilityEventFilterer(QObject* watchedObject, std::function<bool()> filterCondition, QObject* parent = nullptr);

  virtual bool eventFilter(QObject* watchedObject, QEvent* event);

 private:
  std::function<bool()> filterCondition;
};
