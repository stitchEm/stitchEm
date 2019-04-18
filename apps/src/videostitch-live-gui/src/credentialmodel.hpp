// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "sectionmodel.hpp"

enum class CredentialModelColumn { UserName = 0, Delete, NbrColumns };

class CredentialModel : public SectionModel {
  Q_OBJECT

 public:
  explicit CredentialModel(QObject* parent = nullptr);
  virtual ~CredentialModel();

  virtual bool authorizeAndSetCurrentCredential(QString newCredential) = 0;
  virtual bool authorizeAndSetCurrentCredential();
  virtual void revokeCredential(QString credentialToRevoke) = 0;

  // reimplemented methods
  virtual int columnCount(const QModelIndex& parent = QModelIndex()) const;
  virtual int rowCount(const QModelIndex& parent = QModelIndex()) const;
  virtual QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const;
  virtual QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;

 private:
  Q_DISABLE_COPY(CredentialModel)
  bool credentialIsCurrent(int index) const;

 protected:
  QStringList credentials;
  QString currentCredential;
};
