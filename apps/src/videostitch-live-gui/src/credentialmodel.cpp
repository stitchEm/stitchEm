// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "credentialmodel.hpp"

#include <QInputDialog>

CredentialModel::CredentialModel(QObject* parent) : SectionModel(parent) {
  idColumnIndex = int(CredentialModelColumn::UserName);
  deleteColumnIndex = int(CredentialModelColumn::Delete);
  deleteButtonTooltip = tr("Remove account");
}

CredentialModel::~CredentialModel() {}

bool CredentialModel::authorizeAndSetCurrentCredential() {
  QInputDialog dialog(nullptr, Qt::WindowTitleHint | Qt::WindowSystemMenuHint);
  dialog.setWindowTitle(tr("Email address"));
  dialog.setLabelText(tr("Please enter your email address"));
  QString userName;
  if (dialog.exec()) {
    userName = dialog.textValue();
  }
  if (!userName.isEmpty()) {
    return authorizeAndSetCurrentCredential(userName);
  }
  return false;
}

int CredentialModel::columnCount(const QModelIndex& parent) const {
  return parent.isValid() ? 0 : int(CredentialModelColumn::NbrColumns);
}

int CredentialModel::rowCount(const QModelIndex& parent) const { return parent.isValid() ? 0 : credentials.count(); }

QVariant CredentialModel::data(const QModelIndex& index, int role) const {
  if (role == Qt::DisplayRole && index.column() == int(CredentialModelColumn::UserName)) {
    return credentials.value(index.row());
  } else {
    return SectionModel::data(index, role);
  }
}

QVariant CredentialModel::headerData(int section, Qt::Orientation orientation, int role) const {
  if (orientation == Qt::Horizontal && role == Qt::DisplayRole) {
    static QStringList headers = {tr("Email address"), tr("Delete")};
    return headers.value(section);
  }
  return SectionModel::headerData(section, orientation, role);
}

bool CredentialModel::credentialIsCurrent(int index) const { return credentials.value(index) == currentCredential; }
