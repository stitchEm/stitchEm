// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "softwarehelpwidget.hpp"
#include "ui_softwarehelpwidget.h"

#include "common.hpp"
#include "mainwindow/uniqueqapplication.hpp"

#include "libvideostitch-base/common-config.hpp"
#include "libvideostitch-base/linkhelpers.hpp"

#include <QHBoxLayout>

SoftwareHelpWidget::SoftwareHelpWidget(QWidget* parent) : QFrame(parent), ui(new Ui::SoftwareHelpWidget) {
  ui->setupUi(this);

  addHelpItem(tr("YouTube video tutorials"), UniqueQApplication::instance()->getYoutubeUrl(), "YoutubeTutorial");
}

SoftwareHelpWidget::~SoftwareHelpWidget() {}

void SoftwareHelpWidget::addHelpItem(const QString title, const QString url, const QString name) {
  QLabel* labelIcon = new QLabel(this);
  QLabel* formated = new QLabel(formatLink(url, title), this);
  QHBoxLayout* horizontalLayout = new QHBoxLayout();
  formated->setTextFormat(Qt::RichText);
  formated->setTextInteractionFlags(Qt::TextBrowserInteraction);
  formated->setOpenExternalLinks(true);
  labelIcon->setObjectName("label" + name);
  labelIcon->setScaledContents(true);
  labelIcon->setFixedSize(20, 20);
  labelIcon->setFocusPolicy(Qt::NoFocus);
  horizontalLayout->addWidget(labelIcon);
  horizontalLayout->addWidget(formated);
  ui->layoutItems->addItem(horizontalLayout);
}
