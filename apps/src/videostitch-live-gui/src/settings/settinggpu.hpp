// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SETTINGGPU_HPP
#define SETTINGGPU_HPP

#include "iappsettings.hpp"
#include "ui_settinggpu.h"

class SettingGPU : public IAppSettings, public Ui::SettingGPUClass {
  Q_OBJECT

 public:
  explicit SettingGPU(QWidget* const parent = nullptr);
  virtual ~SettingGPU();

  virtual void load();
  virtual void save();

 private slots:
  void checkForChanges();
};
#endif  // SETTINGGPU_HPP
