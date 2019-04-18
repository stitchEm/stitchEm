// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "inputconfigurationwidget.hpp"

#include <memory>

class LiveInputFile;

namespace Ui {
class ConfigurationInputMedia;
}

class ConfigurationInputMedia : public InputConfigurationWidget {
  Q_OBJECT

 public:
  explicit ConfigurationInputMedia(std::shared_ptr<const LiveInputFile> liveInput, QWidget* parent = nullptr);
  ~ConfigurationInputMedia();

 protected:
  virtual void saveData() override;
  virtual void reactToChangedProject() override;
  virtual bool hasValidConfiguration() const override;
 private slots:
  void onSelectFilesClicked();
  void onFilesCleared();

 private:
  QScopedPointer<Ui::ConfigurationInputMedia> ui;
  std::shared_ptr<const LiveInputFile> templateInput;
};
