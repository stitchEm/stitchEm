// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "welcomedialog.hpp"
#include "ui_welcomedialog.h"

#include "libvideostitch/config.hpp"
#include "libvideostitch-gui/common.hpp"

WelcomeDialog::WelcomeDialog(QWidget* const parent) : QDialog(parent), ui(new Ui::WelcomeDialog) {
  ui->setupUi(this);
  setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
  ui->textBrowser->setHtml(QString("<html><head>") + QString(URL_STYLE) + QString("</head><body>") +
                           ui->textBrowser->toHtml()
                               .arg(QCoreApplication::applicationName())
                               .arg(VIDEOSTITCH_TUTORIAL_URL)
                               .arg(VIDEOSTITCH_YOUTUBE_STUDIO_URL)
                               .arg(VIDEOSTITCH_SUPPORT_URL) +
                           QString("</body></html>"));
}

WelcomeDialog::~WelcomeDialog() { delete ui; }
