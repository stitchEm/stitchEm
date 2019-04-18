// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "preferencesdialog.hpp"
#include "ui_preferencesdialog.h"

#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/vssettings.hpp"

#include <QFileDialog>

#define ICON_SIZE 23

QMap<QString, QString> PreferencesDialog::getLanguages() {
  static QMap<QString, QString> languages;
  if (languages.isEmpty()) {
    languages["English"] = "en";
  }
  return languages;
}

PreferencesDialog::PreferencesDialog(QWidget *parent, QVector<int> &deviceIds, QString &language)
    : QDialog(parent, Qt::WindowTitleHint),
      ui(new Ui::PreferencesDialog),
      m_deviceIds(deviceIds),
      m_language(language) {
  ui->setupUi(this);
  ui->deviceList->setSelectedGpus(m_deviceIds);

  ui->languageComboBox->addItems(getLanguages().keys());
  ui->languageComboBox->setCurrentIndex(getLanguages().keys().indexOf(getLanguages().key(m_language)));
  emptyImage.fill(QColor(0, 0, 0, 0));
  warning = QImage(WARNING_ICON).scaled(ICON_SIZE, ICON_SIZE, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  ui->warningIconLabel->setPixmap(QPixmap::fromImage(warning));
  ui->warningIconLabel->setFixedSize(ICON_SIZE, ICON_SIZE);
  ui->warningLabel->setWordWrap(true);
  bool displayLanguageConfiguration =
      VSSettings::getSettings()->getValue("debug/display-language-configuration", false).toBool();
  ui->languageComboBox->setVisible(displayLanguageConfiguration);
  ui->languageLabel->setVisible(displayLanguageConfiguration);
  connect(ui->deviceList, &QListWidget::itemChanged, this, &PreferencesDialog::onSelectedGpusChanged);
}

PreferencesDialog::~PreferencesDialog() { delete ui; }

void PreferencesDialog::onSelectedGpusChanged() {
  if (ui->deviceList->getSelectedGpus() != m_deviceIds) {
    setWarning(true);
  } else {
    setWarning(false);
  }
}

void PreferencesDialog::on_languageComboBox_currentIndexChanged(const QString &newText) {
  if (newText != getLanguages().key(m_language)) {
    setWarning(true);
  } else {
    setWarning(false);
  }
}

void PreferencesDialog::on_buttonBox_accepted() {
  m_language = getLanguages()[ui->languageComboBox->currentText()];
  m_deviceIds = ui->deviceList->getSelectedGpus();
}

void PreferencesDialog::setWarning(bool warning) {
  if (warning) {
    ui->warningLabel->setText(
        tr("You must restart %0 to apply this setting.").arg(QCoreApplication::applicationName()));
    ui->frameWarning->show();
  } else {
    ui->warningLabel->clear();
    ui->frameWarning->hide();
  }
}
