// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "workflowpage.hpp"

WorkflowPage::WorkflowPage(QWidget* parent) : QWidget(parent), workflowDialog(nullptr) {}

void WorkflowPage::setWorkflowDialog(WorkflowDialog* w) { workflowDialog = w; }
