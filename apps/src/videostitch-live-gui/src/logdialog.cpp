// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "logdialog.hpp"
#include "guiconstants.hpp"
#include "libvideostitch-gui/widgets/logwidget.hpp"
#include "animations/animatedwidget.hpp"
#include "dialogbackground.hpp"
#include "livesettings.hpp"
#include "libvideostitch-base/logmanager.hpp"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpacerItem>
#include <QPushButton>

LogDialog::LogDialog(QWidget *const parent)
    : QFrame(parent),
      background(new QFrame(this)),
      layoutBackground(new QVBoxLayout()),
      layoutHorizontal(new QHBoxLayout()),
      buttonClose(new QPushButton(this)),
      labelTitle(new QLabel(tr("Application Log Window"), this)),
      logWidget(new LogWidget(this)) {
  connect(VideoStitch::Helper::LogManager::getInstance(), &VideoStitch::Helper::LogManager::reqWriteToLogFile,
          logWidget.data(), &LogWidget::logMessage);
  connect(buttonClose.data(), &QPushButton::clicked, this, &LogDialog::hide);
  connect(logWidget.data(), &LogWidget::notifyLogLevelChanged, this, &LogDialog::onLogLevelChanged);

  setFixedSize(parent->size());
  labelTitle->setFixedHeight(BUTTON_SIDE);
  buttonClose->setFixedSize(BUTTON_SIDE, BUTTON_SIDE);
  layoutBackground->setContentsMargins(0, 0, 0, 0);
  layoutHorizontal->setContentsMargins(0, 0, 0, 0);
  layoutHorizontal->setSpacing(0);
  setContentsMargins(0, 0, 0, 0);
  background->setObjectName("background");
  labelTitle->setObjectName("labelTitle");
  buttonClose->setObjectName("buttonClose");
  labelTitle->setProperty("vs-title2", true);
  layoutHorizontal->addWidget(labelTitle.data());
  layoutHorizontal->addWidget(buttonClose.data());
  layoutBackground->addLayout(layoutHorizontal.data());
  layoutBackground->addWidget(logWidget.data());
  background->setLayout(layoutBackground.data());
  logWidget->setLogLevel(LiveSettings::getLiveSettings()->getLogLevel());
}

LogDialog::~LogDialog() {}

void LogDialog::updateSize(const QSize &value) {
  setFixedSize(value);
  background->setGeometry(width() / 8, height() / 8, width() * 3 / 4, height() * 3 / 4);
}

void LogDialog::onLogLevelChanged(const int level) { LiveSettings::getLiveSettings()->setLogLevel(level); }
