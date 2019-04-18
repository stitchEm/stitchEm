// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "oculussettingswidget.hpp"
#include "ui_oculussettingswidget.h"

#include "livesettings.hpp"

OculusSettingsWidget::OculusSettingsWidget(QWidget *parent) : IAppSettings(parent), ui(new Ui::OculusSettingsWidget) {
  ui->setupUi(this);
  connect(ui->mirrorModeEnabledBox, &QCheckBox::toggled, this, &OculusSettingsWidget::checkForChanges);
}

OculusSettingsWidget::~OculusSettingsWidget() {}

void OculusSettingsWidget::load() {
  ui->mirrorModeEnabledBox->setChecked(LiveSettings::getLiveSettings()->getMirrorModeEnabled());
}

void OculusSettingsWidget::save() {
  LiveSettings::getLiveSettings()->setMirrorModeEnabled(ui->mirrorModeEnabledBox->isChecked());
}

void OculusSettingsWidget::checkForChanges() { emit notifyNeedToSave(false); }
