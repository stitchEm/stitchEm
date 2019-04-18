// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "centralstackedwidget.hpp"
#include <libvideostitch/logging.hpp>
#include "libvideostitch-gui/centralwidget/icentraltabwidget.hpp"
#include "libvideostitch-gui/centralwidget/ifreezablewidget.hpp"
#include <QHBoxLayout>

static const unsigned int MARGIN(9);

CentralStackedWidget::CentralStackedWidget(QWidget* parent)
    : QWidget(parent),
      _currentIndex(VSTabWidget::undefined),
      sourceWidget(true, this),
      outputWidget(this),
      interactiveWidget(this),
      welcomeWidget(this) {
  setLayout(new QHBoxLayout(this));
  sourceWidget.deactivate();
  this->layout()->addWidget(&sourceWidget);
  outputWidget.deactivate();
  this->layout()->addWidget(&outputWidget);
  interactiveWidget.deactivate();
  this->layout()->addWidget(&interactiveWidget);
  processTabWidget.hide();
  this->layout()->addWidget(&processTabWidget);
  welcomeWidget.hide();
  this->layout()->addWidget(&welcomeWidget);
}

CentralStackedWidget::VSTabWidget CentralStackedWidget::activeTab() const { return _currentIndex; }

bool CentralStackedWidget::allowsPlayback() {
  ICentralTabWidget* currentCentralTab = dynamic_cast<ICentralTabWidget*>(getTabWidget(_currentIndex));
  Q_ASSERT(currentCentralTab);
  return currentCentralTab->allowsPlayback();
}

void CentralStackedWidget::activate(CentralStackedWidget::VSTabWidget index) {
  deactivate(_currentIndex);
  _currentIndex = index;

  QWidget* currentWidget = getTabWidget(index);
  IFreezableWidget* currentFreezableWidget = dynamic_cast<IFreezableWidget*>(currentWidget);
  if (currentFreezableWidget) {
    currentFreezableWidget->activate();
  } else {
    currentWidget->show();
  }
}

void CentralStackedWidget::deactivate(CentralStackedWidget::VSTabWidget index) {
  QWidget* currentWidget = getTabWidget(index);
  IFreezableWidget* currentFreezableWidget = dynamic_cast<IFreezableWidget*>(currentWidget);
  if (currentFreezableWidget) {
    currentFreezableWidget->deactivate();
  } else {
    currentWidget->hide();
  }
}

QWidget* CentralStackedWidget::getTabWidget(const CentralStackedWidget::VSTabWidget index) {
  switch (index) {
    case VSTabWidget::source:
      return &sourceWidget;
    case VSTabWidget::output:
      return &outputWidget;
    case VSTabWidget::interactive:
      return &interactiveWidget;
    case VSTabWidget::process:
      return &processTabWidget;
    case VSTabWidget::welcome:
    default:
      return &welcomeWidget;
  }
}

SourceWidget* CentralStackedWidget::getSourceTabWidget() { return &sourceWidget; }

OutputTabWidget* CentralStackedWidget::getOutputTabWidget() { return &outputWidget; }

InteractiveTabWidget* CentralStackedWidget::getInteractiveTabWidget() { return &interactiveWidget; }

ProcessTabWidget* CentralStackedWidget::getProcessTabWidget() { return &processTabWidget; }

WelcomeScreenWidget* CentralStackedWidget::getWelcomeTabWidget() { return &welcomeWidget; }

void CentralStackedWidget::setPreviewFullScreen(bool activate) {
  const int layoutMargin = activate ? MARGIN : 0;
  layout()->setContentsMargins(layoutMargin, layoutMargin, layoutMargin, layoutMargin);
}
