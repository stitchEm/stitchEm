// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "welcomescreenwidget.hpp"
#include "ui_welcomescreenwidget.h"

#include "softwarehelpwidget.hpp"
#include "projectselectionwidget.hpp"
#include "libvideostitch-gui/mainwindow/versionHelper.hpp"

static const int IMAGE_H_SCALE(2);
static const float IMAGE_V_SCALE(6);
static const int MIN_ADAPTED_WIDTH(950);
static const float DEFAULT_ASPECT_RATIO(2.0f);

WelcomeScreenWidget::WelcomeScreenWidget(QWidget *parent)
    : QFrame(parent),
      ui(new Ui::WelcomeScreenWidget),
      helpWidget(new SoftwareHelpWidget(this)),
      projectWidget(new ProjectSelectionWidget(this)) {
  ui->setupUi(this);
  const QString appInfo = VideoStitch::Helper::AppsInfo(QCoreApplication::applicationVersion()).toString();
  ui->labelProductName->setText(appInfo);
  addWidgets();
}

WelcomeScreenWidget::~WelcomeScreenWidget() { delete ui; }

void WelcomeScreenWidget::addWidgets() {
  connect(projectWidget, &ProjectSelectionWidget::notifyNewProject, this, &WelcomeScreenWidget::notifyNewProject);
  connect(projectWidget, &ProjectSelectionWidget::notifyProjectSelected, this,
          &WelcomeScreenWidget::notifyProjectSelected);
  connect(projectWidget, &ProjectSelectionWidget::notifyFilesDropped, this, &WelcomeScreenWidget::notifyFilesDropped);
  connect(projectWidget, &ProjectSelectionWidget::notifyProjectOpened, this, &WelcomeScreenWidget::notifyProjectOpened);
}

float WelcomeScreenWidget::getLogoRatio() const {
  const QPixmap *logo = ui->labelLogo->pixmap();
  if (logo) {
    return float(logo->width()) / float(logo->height());
  } else {
    return DEFAULT_ASPECT_RATIO;
  }
}

void WelcomeScreenWidget::updateContent() { projectWidget->onContentUpdated(); }

void WelcomeScreenWidget::resizeEvent(QResizeEvent *) {
  // Resize logo according to the screen resolution
  setLogoSize();
  if (ui->scrollArea->width() < MIN_ADAPTED_WIDTH) {
    setSmallSizeOrder();
  } else {
    setBigSizeOrder();
  }
}

void WelcomeScreenWidget::setSmallSizeOrder() {
  ui->bottomLayout->removeWidget(projectWidget);
  ui->leftcolumnlayout->insertWidget(0, projectWidget);
  ui->leftcolumnlayout->insertWidget(1, helpWidget);
}

void WelcomeScreenWidget::setBigSizeOrder() {
  ui->leftcolumnlayout->insertWidget(0, helpWidget);
  ui->leftcolumnlayout->removeWidget(projectWidget);
  ui->bottomLayout->addWidget(projectWidget, 0, 2, 1, 1);
}

void WelcomeScreenWidget::setLogoSize() {
  int newHeight;
  int newWidth;
  if (ui->scrollArea->width() < MIN_ADAPTED_WIDTH) {
    newWidth = width() / IMAGE_H_SCALE;
    newHeight = newWidth / getLogoRatio();
  } else {
    newHeight = height() / IMAGE_V_SCALE;
    newWidth = newHeight * getLogoRatio();
  }
  ui->labelLogo->setFixedSize(newWidth, newHeight);
}
