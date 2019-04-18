// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "logwidget.hpp"
#include "ui_logwidget.h"

#include "libvideostitch/logging.hpp"

#include "libvideostitch-base/logmanager.hpp"
#include "libvideostitch-gui/mainwindow/objectutil.hpp"

#include <QMouseEvent>
#include <QMenu>
#include <QAction>

LogWidget::LogWidget(QWidget *parent, bool displayControls) : QWidget(parent), ui(new Ui::LogWidget) {
  ui->setupUi(this);
  ui->clearLogButton->setProperty("vs-button-medium", true);
  ui->logLevelBox->setVisible(displayControls);
  ui->clearLogButton->setVisible(displayControls);
  if (displayControls) {
    auto levels = VideoStitch::Helper::LogManager::getInstance()->getLevels();
    QString warning =
        VideoStitch::Helper::LogManager::getStringFromLevel(VideoStitch::Helper::LogManager::Level::Warning);
    for (VideoStitch::Helper::LogManager::Level level : levels) {
      ui->logLevelBox->addItem(VideoStitch::Helper::LogManager::getStringFromLevel(level), QVariant::fromValue(level));
    }
    ui->logLevelBox->setCurrentText(warning);
  }
  ui->logEdit->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(ui->logEdit, SIGNAL(customContextMenuRequested(QPoint)), this, SLOT(showTextAreaMenu(const QPoint &)));
  connect(ui->clearLogButton, SIGNAL(clicked()), ui->logEdit, SLOT(clear()));
}

void LogWidget::showControls() {
  ui->logLevelBox->show();
  ui->clearLogButton->show();
}

void LogWidget::hideControls() {
  ui->logLevelBox->hide();
  ui->clearLogButton->hide();
}

void LogWidget::setLogLevel(const int index) { ui->logLevelBox->setCurrentIndex(index); }

LogWidget::~LogWidget() { delete ui; }

void LogWidget::showTextAreaMenu(const QPoint &pt) {
  QMenu *menu = ui->logEdit->createStandardContextMenu();
  menu->addAction("Clear", ui->logEdit, SLOT(clear()));
  menu->exec(ui->logEdit->mapToGlobal(pt));
  delete menu;
}

void LogWidget::logMessage(const QString &message) { ui->logEdit->appendPlainText(message.simplified()); }

void LogWidget::toggleConnect(bool connectionState, const char *signal) {
  VideoStitch::Helper::toggleConnect(connectionState, VideoStitch::Helper::LogManager::getInstance(), signal, this,
                                     SLOT(logMessage(QString)), Qt::UniqueConnection);
}

void LogWidget::on_logLevelBox_currentIndexChanged(int index) {
  VideoStitch::Logger::setLevel((VideoStitch::Logger::LogLevel)index);
  emit notifyLogLevelChanged(index);
  for (int i = 0; i < ui->logLevelBox->count(); ++i) {
    VideoStitch::Helper::LogManager::Level level =
        ui->logLevelBox->itemData(i).value<VideoStitch::Helper::LogManager::Level>();
    toggleConnect(i <= index, VideoStitch::Helper::LogManager::getInstance()->getSignalForLevel(level));
  }
}
