// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "basertmpserviceconfiguration.hpp"
#include "videostitcher/liveoutputrtmp.hpp"
#include "ui_basertmpserviceconfiguration.h"

// Any url starting with rtmp://host etc
// Check rfc3986 for URL specification and special characters allowed
static const QString& rtmpRegExp("^(rtmp)://[a-z0-9A-Z:/?#\\[\\]@!$&\"()*+,;=\\-._~ %]+$");

BaseRTMPServiceConfiguration::BaseRTMPServiceConfiguration(QWidget* parent, LiveOutputRTMP* outputRef,
                                                           LiveProjectDefinition* projectDefinition)
    : IStreamingServiceConfiguration(parent, outputRef, projectDefinition),
      ui(new Ui::BaseRTMPServiceConfiguration),
      validator(QRegExp(rtmpRegExp)) {
  ui->setupUi(this);

  connect(ui->authenticationBox, &QGroupBox::toggled, this, &BaseRTMPServiceConfiguration::stateChanged);
  connect(ui->lineUsername, &QLineEdit::textChanged, this, &BaseRTMPServiceConfiguration::stateChanged);
  connect(ui->linePassword, &QLineEdit::textChanged, this, &BaseRTMPServiceConfiguration::stateChanged);
  connect(ui->lineStreamURL, &QLineEdit::textChanged, this, &BaseRTMPServiceConfiguration::stateChanged);

  ui->lineStreamURL->setValidator(&validator);
}

BaseRTMPServiceConfiguration::~BaseRTMPServiceConfiguration() {}

bool BaseRTMPServiceConfiguration::loadConfiguration() {
  ui->lineStreamURL->setText(outputRef->getUrl());
  ui->authenticationBox->setChecked(outputRef->needsAuthentication());
  if (outputRef->needsAuthentication()) {
    ui->lineUsername->setText(outputRef->getPubUser());
    ui->linePassword->setText(outputRef->getPubPasswd());
  }

  return true;
}

void BaseRTMPServiceConfiguration::saveConfiguration() {
  if (ui->authenticationBox->isChecked()) {
    outputRef->setPubUser(ui->lineUsername->text());
    outputRef->setPubPasswd(ui->linePassword->text());
  } else {
    outputRef->setPubUser(QString());
    outputRef->setPubPasswd(QString());
  }
  outputRef->setUrl(ui->lineStreamURL->text());
}

bool BaseRTMPServiceConfiguration::hasValidConfiguration() const { return ui->lineStreamURL->hasAcceptableInput(); }
