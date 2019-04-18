// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "workflowdialog.hpp"
#include "ui_workflowdialog.h"

#include "generic/genericloader.hpp"
#include "workflowpage.hpp"

#include <QPushButton>

WorkflowDialog::WorkflowDialog(QWidget* parent)
    : QDialog(parent),
      ui(new Ui::WorkflowDialog),
      project(nullptr),
      buttonBack(new QPushButton()),
      buttonApply(new QPushButton()) {
  ui->setupUi(this);
  ui->stackedWidget->setCurrentWidget(ui->workflowPage);
  ui->errorLabel->hide();

  QPushButton* buttonCancel = new QPushButton(this);
  buttonBack->setObjectName("buttonBack");
  buttonApply->setObjectName("buttonApply");
  buttonCancel->setObjectName("buttonCancel");
  buttonBack->setProperty("vs-button-medium", true);
  buttonApply->setProperty("vs-button-medium", true);
  buttonCancel->setProperty("vs-button-medium", true);
  ui->buttonBox->addButton(buttonApply.data(), QDialogButtonBox::ActionRole);
  ui->buttonBox->addButton(buttonCancel, QDialogButtonBox::RejectRole);

  QPushButton* buttonFinalApply = new QPushButton(this);
  buttonFinalApply->setObjectName("buttonApply");
  buttonFinalApply->setProperty("vs-button-medium", true);
  ui->summaryButtonBox->addButton(buttonFinalApply, QDialogButtonBox::AcceptRole);

  connect(buttonBack.data(), &QPushButton::clicked, this, &WorkflowDialog::goBack);
  connect(buttonApply.data(), &QPushButton::clicked, this, &WorkflowDialog::saveCurrentPage);
  connect(ui->buttonBox, &QDialogButtonBox::rejected, this, &WorkflowDialog::reject);
  connect(ui->summaryButtonBox, &QDialogButtonBox::accepted, this, &WorkflowDialog::accept);

  setWindowFlags((windowFlags() | Qt::CustomizeWindowHint | Qt::MSWindowsFixedSizeDialogHint) &
                 ~Qt::WindowCloseButtonHint & ~Qt::WindowContextHelpButtonHint);
}

WorkflowDialog::~WorkflowDialog() {}

ProjectDefinition* WorkflowDialog::getProject() const { return project; }

void WorkflowDialog::setProject(ProjectDefinition* p) {
  project = p;
  for (auto page : pages) {
    page->setProject(project);
  }
}

void WorkflowDialog::addPage(WorkflowPage* page) {
  pages.append(QSharedPointer<WorkflowPage>(page));
  page->setWorkflowDialog(this);
  page->setProject(project);

  if (pages.count() == 1) {
    page->initializePage();
    ui->workflowPageLayout->insertWidget(0, page);
  }
}

void WorkflowDialog::completeCurrentPage(QString message) {
  hideErrorMessage();
  if (!ui->summaryLabel->text().isEmpty()) {
    message = ui->summaryLabel->text() + "\n" + message;
  }
  ui->summaryLabel->setText(message);

  int index = getCurrentPageIndex();
  ui->workflowPageLayout->takeAt(0)->widget()->setParent(nullptr);
  pages.at(index)->deinitializePage();

  if (index == pages.count() - 1) {
    ui->stackedWidget->setCurrentWidget(ui->summaryPage);
  } else {
    ui->buttonBox->addButton(buttonBack.data(), QDialogButtonBox::ActionRole);
    ui->buttonBox->addButton(buttonApply.data(),
                             QDialogButtonBox::ActionRole);  // Re-add the apply button to keep the order
    pages.at(index + 1)->initializePage();
    ui->workflowPageLayout->insertWidget(0, pages.at(index + 1).data());
  }
}

void WorkflowDialog::showErrorMessage(QString errorMessage) {
  ui->errorLabel->setText(errorMessage);
  ui->errorLabel->show();
}

void WorkflowDialog::hideErrorMessage() { ui->errorLabel->hide(); }

void WorkflowDialog::showWaitingPage(QString message) {
  GenericLoader* loader = new GenericLoader(message, ui->waitingPage);
  loader->show();
  loader->raise();
  loader->updateSize(ui->waitingPage->size().width(), ui->waitingPage->size().height());
  ui->stackedWidget->setCurrentWidget(ui->waitingPage);
}

void WorkflowDialog::closeWaitingPage() {
  if (ui->stackedWidget->currentWidget() == ui->waitingPage) {
    ui->stackedWidget->setCurrentWidget(ui->workflowPage);
  }
  GenericLoader* loader = ui->waitingPage->findChild<GenericLoader*>();
  delete loader;
}

void WorkflowDialog::blockNextStep(bool block) { buttonApply->setDisabled(block); }

WorkflowPage* WorkflowDialog::getCurrentPage() const {
  return qobject_cast<WorkflowPage*>(ui->workflowPageLayout->itemAt(0)->widget());
}

int WorkflowDialog::getCurrentPageIndex() const {
  auto currentPage = getCurrentPage();
  for (int index = 0; index < pages.count(); ++index) {
    if (pages.at(index) == currentPage) {
      return index;
    }
  }
  return -1;
}

void WorkflowDialog::saveCurrentPage() { getCurrentPage()->save(); }

void WorkflowDialog::goBack() {
  hideErrorMessage();

  int index = getCurrentPageIndex();
  if (index == 1) {
    ui->buttonBox->removeButton(buttonBack.data());
  }

  pages.at(index - 1)->initializePage();
  ui->workflowPageLayout->takeAt(0)->widget()->setParent(nullptr);
  ui->workflowPageLayout->insertWidget(0, pages.at(index - 1).data());
  pages.at(index)->deinitializePage();
}
