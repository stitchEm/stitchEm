// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/mainwindow/packet.hpp"
#include "libvideostitch-gui/mainwindow/stitchingwindow.hpp"

namespace Ui {
class MainWindowClass;
}

class LogDialog;
class AboutWidget;
class ProjectWorkWidget;
class GenericDialog;
class WelcomeScreenWidget;
class QShortcut;
class QDropEvent;

class MainWindow : public StitchingWindow {
  Q_OBJECT

 public:
  /**
   * @brief Constructor
   * @param parent A parent Widget
   */
  explicit MainWindow(QWidget* const parent = nullptr);
  ~MainWindow();

  void updateMainTitle();
  ProjectWorkWidget* getProjectWorkWidget() const { return currentWorkWidget; }

 protected:
  virtual void resizeEvent(QResizeEvent* event) override;
  virtual void closeEvent(QCloseEvent* event) override;

 public slots:
  void processMessage(const Packet& packet);
  void onFileOpened(const QStringList files);
  void onEnableWindow();
  void onDisableWindow();

 private:
  void initializeMainWindow();
  void initalizeWelcomePage();
  void initializeConnections();
  void initializeUserFolders();
  bool checkForCudaDevices();

  QScopedPointer<Ui::MainWindowClass> ui;
  ProjectWorkWidget* currentWorkWidget = nullptr;
  WelcomeScreenWidget* welcomeWidget = nullptr;
  LogDialog* logDialog = nullptr;
  AboutWidget* aboutDialog = nullptr;
  GenericDialog* exitDialog = nullptr;
  bool activeProject;
  bool audioPlayback;

 private slots:
  void onActivateAudioPlayback(bool);
  void onButtonStartNewProjectClicked();
  void onButtonOpenProject();
  void onShowLogClicked();
  void onButtonAboutClicked();
  void onButtonSettingsClicked();
  void showCredentialWindow();
  void onFileSelected(const QString& file);
  void onProjectLoaded();
  void onProjectClosed();
  void onProjectNameAccepted(const QString& name);
  void onExitApplicationAccepted();
  void onExitApplicationRejected();
  void onResetDimensions(const unsigned panoWidth, const unsigned panoHeight);
  void onNewPanoSizeSet(const int width, const int height);
  void onDropInputs(QDropEvent* e);
  void showIncompatibleFileDialog();

 signals:
  void reqSendMessage(Packet packet, QString host, bool andDie = false);
  void reqRestartApplication();
  void notifySizeChanged(const QSize& size);
  void notifySaveProject();
};
