// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "googlecredentialmodel.hpp"

#include "googleauthenticationmanager.hpp"

#include <QDir>

GoogleCredentialModel::GoogleCredentialModel(QObject* parent) : CredentialModel(parent) {}

GoogleCredentialModel::~GoogleCredentialModel() {}

void GoogleCredentialModel::initialize(QString newCredentialsPath, QString newCredentialStoreName) {
  beginResetModel();
  storePath = newCredentialsPath;
  storeName = newCredentialStoreName;
  update();
  endResetModel();
}

void GoogleCredentialModel::update() {
  credentials.clear();
  for (QFileInfo credential : QDir(storePath).entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name)) {
    if (QDir(credential.absoluteFilePath()).exists(storeName)) {
      credentials.append(credential.fileName());
    }
  }
}

void GoogleCredentialModel::internalSetCurrentAndUpdate(QString newCurrentCredential) {
  beginResetModel();
  update();
  currentCredential = newCurrentCredential;
  endResetModel();
}

void GoogleCredentialModel::internalRevokeCredential(QString credentialToRevoke) {
  beginResetModel();
  update();
  if (currentCredential == credentialToRevoke) {
    currentCredential = QString();
  }
  endResetModel();
}

bool GoogleCredentialModel::authorizeAndSetCurrentCredential(QString newCredential) {
  return GoogleAuthenticationManager::getInstance().authorizeClient(newCredential);
}

void GoogleCredentialModel::revokeCredential(QString credentialToRevoke) {
  GoogleAuthenticationManager::getInstance().revokeClient(credentialToRevoke);
}
