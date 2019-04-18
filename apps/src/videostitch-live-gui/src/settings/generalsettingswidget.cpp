// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "generalsettingswidget.hpp"
#include "generic/backgroundcontainer.hpp"
#include "livesettings.hpp"
#include "settinggpu.hpp"
#include "widgetsmanager.hpp"

// -------------- Widget --------------
GeneralSettingsWidget::GeneralSettingsWidget(QWidget* const parent) : QFrame(parent) {
  setupUi(this);
  buttonSave->setProperty("vs-button-medium", true);
  labelWarningMessage->setText(labelWarningMessage->text().arg(QCoreApplication::applicationName()));
  showWarning(false);
  connect(buttonSave, &QPushButton::clicked, this, &GeneralSettingsWidget::onSettingsSaved);

  // Add all the desired application settings
  for (IAppSettings* settingWidget : findChildren<IAppSettings*>()) {
    settingWidget->load();
    connect(settingWidget, &IAppSettings::notifyNeedToSave, this, &GeneralSettingsWidget::showWarning);
  }
}

GeneralSettingsWidget::~GeneralSettingsWidget() {}

void GeneralSettingsWidget::onSettingsSaved() {
  for (IAppSettings* settingWidget : findChildren<IAppSettings*>()) {
    settingWidget->save();
  }
  emit notifySettingsSaved();
}

void GeneralSettingsWidget::showWarning(const bool show) {
  labelWarningIcon->setVisible(show);
  labelWarningMessage->setVisible(show);
}

// -------------- Dialog --------------
GeneralSettingsDialog::GeneralSettingsDialog(QWidget* const parent)
    : QFrame(parent),
      widget(new GeneralSettingsWidget(this)),
      background(new BackgroundContainer(widget, tr("Application settings"), parent)) {
  background->show();
  background->raise();
  setAttribute(Qt::WA_DeleteOnClose);
  connect(background, &BackgroundContainer::notifyWidgetClosed, this, &GeneralSettingsDialog::onWidgetClosed);
  connect(widget, &GeneralSettingsWidget::notifySettingsSaved, this, &GeneralSettingsDialog::onWidgetClosed);
}

GeneralSettingsDialog::~GeneralSettingsDialog() {}

void GeneralSettingsDialog::onWidgetClosed() {
  background->hide();
  background->deleteLater();
  close();
}
