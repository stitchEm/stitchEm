// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef IAPPSETTINGS_HPP
#define IAPPSETTINGS_HPP

#include <QWidget>

class IAppSettings : public QWidget {
  Q_OBJECT
 public:
  explicit IAppSettings(QWidget* const parent = nullptr) : QWidget(parent) {}
  virtual void save() = 0;
  virtual void load() = 0;

 signals:
  void notifyNeedToSave(const bool needToRestart);
};

#endif  // IAPPSETTINGS_HPP
