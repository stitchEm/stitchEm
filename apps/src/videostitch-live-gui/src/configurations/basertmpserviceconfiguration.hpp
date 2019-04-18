// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef BASERTMPSERVICECONFIGURATION_HPP
#define BASERTMPSERVICECONFIGURATION_HPP

#include <QWidget>
#include <QRegExpValidator>
#include "istreamingserviceconfiguration.hpp"

namespace Ui {
class BaseRTMPServiceConfiguration;
}

class BaseRTMPServiceConfiguration : public IStreamingServiceConfiguration {
  Q_OBJECT

 public:
  explicit BaseRTMPServiceConfiguration(QWidget* parent, LiveOutputRTMP* outputRef,
                                        LiveProjectDefinition* projectDefinition = nullptr);
  ~BaseRTMPServiceConfiguration();

  virtual bool loadConfiguration() override;
  virtual void saveConfiguration() override;
  virtual bool hasValidConfiguration() const override;

 private:
  QScopedPointer<Ui::BaseRTMPServiceConfiguration> ui;
  QRegExpValidator validator;
};

#endif  // BASERTMPSERVICECONFIGURATION_HPP
