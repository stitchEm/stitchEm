// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "authenticationwidget.hpp"
#include "ui_authenticationwidget.h"

#include "credentialfactory.hpp"
#include "credentialmodel.hpp"
#include "generic/backgroundcontainer.hpp"

AuthenticationWidget::AuthenticationWidget(QWidget* parent)
    : QFrame(parent),
      ui(new Ui::AuthenticationWidget),
      refOnCredentialModel(CredentialFactory::getCredentialModel(OutputCredential::YouTube)) {
  ui->setupUi(this);
  ui->createButton->setProperty("vs-section-add-button", true);
  connect(ui->credentialView, &SectionView::askToDeleteRow, this, &AuthenticationWidget::deleteCredential);
  connect(ui->createButton, &QPushButton::clicked, this, &AuthenticationWidget::createCredential);
  connect(refOnCredentialModel.get(), &CredentialModel::modelReset, this, &AuthenticationWidget::updateStackPage);

  ui->credentialView->setSectionModel(refOnCredentialModel.get());
  ui->credentialView->horizontalHeader()->setSectionResizeMode(int(CredentialModelColumn::UserName),
                                                               QHeaderView::Stretch);

  updateStackPage();
}

AuthenticationWidget::~AuthenticationWidget() {}

void AuthenticationWidget::createCredential() { refOnCredentialModel->authorizeAndSetCurrentCredential(); }

void AuthenticationWidget::deleteCredential(const QString& userName) {
  refOnCredentialModel->revokeCredential(userName);
}

void AuthenticationWidget::updateStackPage() {
  auto index = refOnCredentialModel->rowCount() > 0 ? 0 : 1;
  ui->stackedWidget->setCurrentIndex(index);
}

AuthenticationDialog::AuthenticationDialog(QWidget* parent)
    : QFrame(parent),
      widget(new AuthenticationWidget(this)),
      background(new BackgroundContainer(widget, tr("YouTube credentials"), parent)) {
  setAttribute(Qt::WA_DeleteOnClose);
  connect(background, &BackgroundContainer::notifyWidgetClosed, this, &AuthenticationDialog::closeDialog);
}

AuthenticationDialog::~AuthenticationDialog() {}

void AuthenticationDialog::show() { background->show(); }

void AuthenticationDialog::closeDialog() {
  background->hide();
  background->deleteLater();
  close();
}
