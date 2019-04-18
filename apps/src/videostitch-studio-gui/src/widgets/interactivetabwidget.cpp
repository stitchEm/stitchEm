// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "interactivetabwidget.hpp"
#include "videostitcher/globalpostprodcontroller.hpp"

InteractiveTabWidget::InteractiveTabWidget(QWidget *const parent) : IFreezableWidget("InteractiveTab", parent) {
  setupUi(this);
  stackedWidget->setCurrentWidget(pageInteractiveView);
}

InteractiveTabWidget::~InteractiveTabWidget() {
  // clear label
  clearScreenshot();
}

DeviceInteractiveWidget *InteractiveTabWidget::getInteractiveWidget() { return interactiveWidget; }
