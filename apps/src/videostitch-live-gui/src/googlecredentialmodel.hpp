// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "credentialmodel.hpp"

class GoogleCredentialModel : public CredentialModel {
  Q_OBJECT

 public:
  explicit GoogleCredentialModel(QObject* parent = nullptr);
  virtual ~GoogleCredentialModel();

  void initialize(QString newCredentialsPath, QString newCredentialStoreName);
  void internalSetCurrentAndUpdate(QString newCurrentCredential);
  void internalRevokeCredential(QString credentialToRevoke);

  using ::CredentialModel::authorizeAndSetCurrentCredential;

  virtual bool authorizeAndSetCurrentCredential(QString newCredential) override;
  virtual void revokeCredential(QString credentialToRevoke) override;

 private:
  Q_DISABLE_COPY(GoogleCredentialModel)
  void update();

  QString storePath;
  QString storeName;
};
