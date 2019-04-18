// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputdetailwidget.hpp"

#include "guiconstants.hpp"
#include "generic/genericdialog.hpp"
#include "videostitcher/liveoutputfactory.hpp"
#include "widgetsmanager.hpp"

OutputDetailWidget::OutputDetailWidget(LiveOutputFactory* output, QWidget* parent)
    : QWidget(parent), output(output), deleteDialog(nullptr) {
  setupUi(this);
  frame->setProperty("vs-section-contained-widget", true);
  buttonDelete->setProperty("vs-section-delete-button", true);
  labelName->setText(output->getOutputTypeDisplayName());
  updateDetails();
  labelIcon->setPixmap(output->getIcon());
  labelIcon->setScaledContents(true);

  connect(buttonDelete, &QPushButton::clicked, this, &OutputDetailWidget::onDeleteClicked);
  connect(output, &LiveOutputFactory::outputDisplayNameChanged, this, &OutputDetailWidget::updateDetails);
}

OutputDetailWidget::~OutputDetailWidget() {}

LiveOutputFactory* OutputDetailWidget::getOutput() const { return output; }

void OutputDetailWidget::updateDetails() {
  QString details = output->getOutputDisplayName();
  labelDetailedName->setText(details.isEmpty() ? QString() : QString("(%1)").arg(details));
}

void OutputDetailWidget::allowsRemoving(bool allow) { buttonDelete->setVisible(allow); }

void OutputDetailWidget::onDeleteClicked() {
  if (deleteDialog == nullptr) {
    QString message =
        output->getOutputDisplayName().isEmpty()
            ? tr("Are you sure that you want to delete the output: %0?").arg(output->getOutputTypeDisplayName())
            : tr("Are you sure that you want to delete the output: %0 named %1?")
                  .arg(output->getOutputTypeDisplayName())
                  .arg(output->getOutputDisplayName());
    deleteDialog = new GenericDialog(tr("Delete Output"), message, GenericDialog::DialogMode::ACCEPT_CANCEL,
                                     WidgetsManager::getInstance()->getMainWindowRef());
    connect(deleteDialog, &GenericDialog::notifyAcceptClicked, this, &OutputDetailWidget::onDeleteAccepted);
    connect(deleteDialog, &GenericDialog::notifyCancelClicked, this, &OutputDetailWidget::onDeleteRejected);
    deleteDialog->show();
    deleteDialog->raise();
  }
}

void OutputDetailWidget::onDeleteAccepted() {
  emit notifyDeleteOutput(output->getIdentifier());
  deleteDialog->close();
  delete deleteDialog;
  deleteDialog = nullptr;
}

void OutputDetailWidget::onDeleteRejected() {
  deleteDialog->close();
  delete deleteDialog;
  deleteDialog = nullptr;
}
