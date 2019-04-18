// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <memory>

#include <QFrame>

namespace Ui {
class AuthenticationWidget;
}

class BackgroundContainer;
class CredentialModel;

class AuthenticationWidget : public QFrame {
  Q_OBJECT

 public:
  explicit AuthenticationWidget(QWidget* parent = nullptr);
  virtual ~AuthenticationWidget();

 private slots:
  void createCredential();
  void deleteCredential(const QString& userName);
  void updateStackPage();

 private:
  Q_DISABLE_COPY(AuthenticationWidget)

  QScopedPointer<Ui::AuthenticationWidget> ui;
  std::shared_ptr<CredentialModel> refOnCredentialModel;
};

// AuthenticationDialog manages itself its lifecycle when we show it
class AuthenticationDialog : public QFrame {
  Q_OBJECT

 public:
  explicit AuthenticationDialog(QWidget* parent = nullptr);
  virtual ~AuthenticationDialog();

  void show();

 private slots:
  void closeDialog();

 private:
  AuthenticationWidget* widget;
  BackgroundContainer* background;
};
