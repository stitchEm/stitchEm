// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "youtubeconfiguration.hpp"
#include "youtubebroadcastmodel.hpp"
#include "googleauthenticationmanager.hpp"
#include "googlecredentialmodel.hpp"

#include "guiconstants.hpp"

#include "ui_youtubeconfiguration.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

#include "googleapis/client/auth/oauth2_authorization.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <array>

YoutubeConfiguration::YoutubeConfiguration(QWidget* parent)
    : QFrame(parent),
      ui(new Ui::YoutubeConfiguration),
      youtubeBroadcastModel(new YoutubeBroadcastModel()),
      googleCredentialModel(GoogleAuthenticationManager::getInstance().getCredentialModel()) {
  ui->setupUi(this);
  ui->updateButton->setProperty("vs-button-medium", true);
  ui->saveButton->setProperty("vs-button-medium", true);
  ui->authButton->setProperty("vs-section-add-button", true);

  ui->broadcastsView->setModel(youtubeBroadcastModel.get());
  ui->accountsView->setSectionModel(googleCredentialModel.get());
  ui->accountsView->horizontalHeader()->setSectionResizeMode(int(CredentialModelColumn::UserName),
                                                             QHeaderView::Stretch);
  ui->accountsView->setColumnHidden(int(CredentialModelColumn::Delete), true);
  ui->accountsView->setVisible(googleCredentialModel->rowCount());

  connect(ui->accountsView, &QTableView::clicked, this, &YoutubeConfiguration::accountSelected);

  connect(ui->broadcastsView->selectionModel(), &QItemSelectionModel::selectionChanged, this,
          &YoutubeConfiguration::onSelectionChanged);
  connect(youtubeBroadcastModel.get(), &YoutubeBroadcastModel::modelReset, this,
          &YoutubeConfiguration::handleStateChanged);

  connect(ui->updateButton, SIGNAL(clicked()), this, SLOT(updateYoutubeData()));
  connect(ui->authButton, &QPushButton::clicked, this, &YoutubeConfiguration::authenticate);

  connect(ui->changeAccountButton, &QPushButton::clicked, this, &YoutubeConfiguration::changeAccount);
  connect(ui->stackedWidget, &QStackedWidget::currentChanged, this, &YoutubeConfiguration::onStackedWidgetPageChanged);
  connect(ui->saveButton, &QPushButton::clicked, this, [this]() {
    emit notifySettingsSaved(
        youtubeBroadcastModel->getBroadcastIdForIndex(ui->broadcastsView->selectionModel()->currentIndex()));
  });

  ui->infoLabel->setTextInteractionFlags(Qt::TextBrowserInteraction);
  ui->infoLabel->setOpenExternalLinks(true);

  toggleAccountInfoVisible(false);
  ui->stackedWidget->setCurrentIndex(Ui::PAGE_SELECT_ACCOUNT);
}

YoutubeConfiguration::~YoutubeConfiguration() {}

void YoutubeConfiguration::handleStateChanged() {
  auto authorized = GoogleAuthenticationManager::getInstance().authorized();

  if (!authorized) {
    ui->stackedWidget->setCurrentIndex(Ui::PAGE_SELECT_ACCOUNT);
    return;
  }

  ui->stackedWidget->setCurrentIndex(Ui::PAGE_BROADCASTS);
  ui->selectEventBox->setVisible(youtubeBroadcastModel->rowCount());
}

void YoutubeConfiguration::updateYoutubeData() {
  if (GoogleAuthenticationManager::getInstance().authorized()) {
    ui->accountNameLabel->setText(
        QString::fromStdString(GoogleAuthenticationManager::getInstance().getCredential()->email()));

    youtubeBroadcastModel->updateUserData(GoogleAuthenticationManager::getInstance().getCredential());
  }
}

void YoutubeConfiguration::authenticate() {
  if (googleCredentialModel->authorizeAndSetCurrentCredential()) {
    updateYoutubeData();
  }
}

void YoutubeConfiguration::accountSelected(const QModelIndex& selected) {
  googleCredentialModel->authorizeAndSetCurrentCredential(selected.data().toString());

  updateYoutubeData();
  handleStateChanged();
}

void YoutubeConfiguration::changeAccount() {
  ui->stackedWidget->setCurrentIndex(Ui::PAGE_SELECT_ACCOUNT);
  ui->broadcastsView->clearSelection();
}

void YoutubeConfiguration::onStackedWidgetPageChanged(int pageId) {
  toggleAccountInfoVisible(pageId != Ui::PAGE_SELECT_ACCOUNT);
}

void YoutubeConfiguration::onSelectionChanged(const QItemSelection& selected, const QItemSelection& deselected) {
  Q_UNUSED(deselected);
  ui->saveButton->setEnabled(!selected.empty());
}

void YoutubeConfiguration::toggleAccountInfoVisible(bool visible) { ui->accountInfoFrame->setVisible(visible); }
