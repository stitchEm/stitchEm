// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "iprocesswidget.hpp"

IProcessWidget::IProcessWidget(QWidget* const parent) : QWidget(parent), project(nullptr) {}

IProcessWidget::~IProcessWidget() {}

void IProcessWidget::setProject(PostProdProjectDefinition* projectDef) {
  project = projectDef;
  reactToChangedProject();
}

void IProcessWidget::clearProject() {
  project = nullptr;
  reactToClearedProject();
}
