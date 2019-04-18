// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <QUrl>
#include <QDate>
#include <QLabel>
#include <QPushButton>
#include <QDesktopServices>
#include "aboutwindow.hpp"
#include "version.hpp"
#include "common.hpp"

AboutWidget::AboutWidget(QString version, QWidget* const parent) : QWidget(parent) {
  setupUi(this);
  buttonWebSite->setProperty("vs-button-medium", true);

  labelVSName->setText(labelVSName->text()
                           .arg(QCoreApplication::applicationName())
                           .arg(QCoreApplication::organizationName())
                           .arg(QDate::currentDate().year()));
  connect(buttonWebSite, &QPushButton::clicked, this, &AboutWidget::onButtonWebSiteClicked);
  labelAppVersion->setText(version);
  QIcon webIcon(":/live/icons/assets/icon/live/web.png");
  buttonWebSite->setIcon(webIcon);
}

AboutWidget::~AboutWidget() {}

void AboutWidget::onButtonWebSiteClicked() { QDesktopServices::openUrl(QUrl(VIDEOSTITCH_URL)); }
