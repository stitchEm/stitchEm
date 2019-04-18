// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mainwindow.hpp"
#include "ui_mainwindow.h"

#include "guiconstants.hpp"
#include "logdialog.hpp"
#include "libvideostitch-gui/widgets/aboutwindow.hpp"
#include "projectworkwidget.hpp"
#include "newprojectnamedialog.hpp"
#include "resizepanoramadialog.hpp"
#include "livesettings.hpp"
#include "settings/generalsettingswidget.hpp"
#include "authenticationwidget.hpp"
#include "generic/genericdialog.hpp"
#include "widgetsmanager.hpp"
#include "generic/backgroundcontainer.hpp"
#include "videostitcher/globallivecontroller.hpp"

#include "libvideostitch-gui/common.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/videostitcher/presetsmanager.hpp"
#include "libvideostitch-gui/utils/widgetshelper.hpp"
#include "libvideostitch-gui/widgets/welcome/welcomescreenwidget.hpp"

#include "libvideostitch-base/logmanager.hpp"

#include "libvideostitch/logging.hpp"

#include <QFileDialog>

MainWindow::MainWindow(QWidget* const parent)
    : StitchingWindow(parent, nullptr),
      ui(new Ui::MainWindowClass()),
      exitDialog(nullptr),
      activeProject(false),
      audioPlayback(false) {
  MsgBoxHandler::getInstance();
  PresetsManager::getInstance();
  GlobalLiveController::getInstance();

  currentWorkWidget = new ProjectWorkWidget(this);
  welcomeWidget = new WelcomeScreenWidget(this);
  logDialog = new LogDialog(this);
  aboutDialog = new AboutWidget(VideoStitch::Helper::AppsInfo(QCoreApplication::applicationVersion()).toString(), this);

  initializeMainWindow();
  initializeUserFolders();
  initalizeWelcomePage();
  showMaximized();
}

MainWindow::~MainWindow() {
  MsgBoxHandler::getInstance()->destroy();
  ProjectFileHandler::getInstance()->destroy();
  WidgetsManager::getInstance()->destroy();
}

void MainWindow::resizeEvent(QResizeEvent* event) {
  Q_UNUSED(event)
  notifySizeChanged(size());
  logDialog->updateSize(size());
}

void MainWindow::closeEvent(QCloseEvent* event) {
  if (activeProject) {
    if (exitDialog == nullptr) {
      const QString& title(tr("Exit application"));
      const QString& message(tr("Would you like to close the current camera configuration and exit the application?"));
      exitDialog = new GenericDialog(title, message, GenericDialog::DialogMode::ACCEPT_CANCEL, this);
      connect(exitDialog, &GenericDialog::notifyAcceptClicked, this, &MainWindow::onExitApplicationAccepted);
      connect(exitDialog, &GenericDialog::notifyCancelClicked, this, &MainWindow::onExitApplicationRejected);
      exitDialog->show();
      event->ignore();
    } else {
      event->ignore();
    }
  } else {
    QApplication::exit();
  }
}

void MainWindow::updateMainTitle() { setWindowTitle(QCoreApplication::applicationName()); }

void MainWindow::initializeUserFolders() {
  // Create user directories if needed
  QDir dir;
  dir.mkpath(getRecordingsPath());
  dir.mkpath(getProjectsPath());
  dir.mkpath(getSnapshotsPath());
}

void MainWindow::initializeMainWindow() {
  ui->setupUi(this);
  logDialog->hide();
  aboutDialog->hide();
  ui->buttonAbout->setToolTip(tr("About %0").arg(QCoreApplication::applicationName()));
  ui->stackedWidget->addWidget(welcomeWidget);
  ui->stackedWidget->addWidget(currentWorkWidget);
  ui->stackedWidget->setCurrentWidget(welcomeWidget);
  ui->buttonPlayback->setVisible(VSSettings::getSettings()->getShowExperimentalFeatures());
  ui->buttonPlayback->setEnabled(false);

  // VSA-6594
#if !ENABLE_YOUTUBE_OUTPUT
  ui->credentialButton->hide();
#endif

  updateMainTitle();

  connect(currentWorkWidget, &ProjectWorkWidget::notifyProjectOpened, this, &MainWindow::onProjectLoaded);
  connect(currentWorkWidget, &ProjectWorkWidget::notifyPanoResized, this, &MainWindow::onResetDimensions);
  connect(currentWorkWidget, &ProjectWorkWidget::notifyProjectClosed, this, &MainWindow::onProjectClosed);
  connect(currentWorkWidget, &ProjectWorkWidget::notifyBackendCompileProgress, this,
          &MainWindow::onKernelProgressChanged);
  connect(currentWorkWidget, &ProjectWorkWidget::notifyBackendCompileDone, this, &MainWindow::onKernelCompileDone);
  connect(currentWorkWidget, &ProjectWorkWidget::reqDisableWindow, this, &MainWindow::onDisableWindow);
  connect(currentWorkWidget, &ProjectWorkWidget::reqEnableWindow, this, &MainWindow::onEnableWindow);
  connect(ui->buttonLog, &QPushButton::clicked, this, &MainWindow::onShowLogClicked);
  connect(ui->buttonAbout, &QPushButton::clicked, this, &MainWindow::onButtonAboutClicked);
  connect(ui->buttonSettings, &QPushButton::clicked, this, &MainWindow::onButtonSettingsClicked);
  connect(ui->credentialButton, &QPushButton::clicked, this, &MainWindow::showCredentialWindow);
  connect(this, &MainWindow::notifySaveProject, currentWorkWidget, &ProjectWorkWidget::saveProject,
          Qt::QueuedConnection);
  connect(this, &MainWindow::reqCancelKernelCompile, currentWorkWidget, &ProjectWorkWidget::reqCancelKernelCompile);
  connect(ui->buttonPlayback, &QPushButton::toggled, this, &MainWindow::onActivateAudioPlayback);
}

void MainWindow::initalizeWelcomePage() {
  connect(welcomeWidget, &WelcomeScreenWidget::notifyNewProject, this, &MainWindow::onButtonStartNewProjectClicked);
  connect(welcomeWidget, &WelcomeScreenWidget::notifyProjectSelected, this, &MainWindow::onFileSelected);
  connect(welcomeWidget, &WelcomeScreenWidget::notifyProjectOpened, this, &MainWindow::onButtonOpenProject);
  connect(welcomeWidget, &WelcomeScreenWidget::notifyFilesDropped, this, &MainWindow::onDropInputs);
}

bool MainWindow::checkForCudaDevices() {
  // XXX TODO FIXME move to the lib? initVideoStitch function?
  if (VideoStitch::Discovery::getNumberOfCudaDevices() == 0) {
    GenericDialog* errorDialog(
        new GenericDialog("Critical error", "No usable GPU", GenericDialog::DialogMode::ACCEPT, this));
    errorDialog->show();
    return false;
  }
  return true;
}

void MainWindow::onActivateAudioPlayback(bool b) {
  audioPlayback = b;
  emit currentWorkWidget->notifyAudioPlaybackActivated(b);
}

void MainWindow::processMessage(const Packet& packet) {
  Q_UNUSED(packet)
#ifdef Q_OS_WIN
  switch (packet.getType()) {
    case Packet::WAKEUP: {
      VideoStitch::WidgetsHelpers::bringToForeground(this);
      break;
    }
    case Packet::OPEN_FILES: {
      VideoStitch::WidgetsHelpers::bringToForeground(this);
      const QString argString = QString::fromLatin1(packet.getPayload());
      const QStringList args = QStringList() << argString;
      onFileOpened(args);
      break;
    }
    default:
      break;
  }
#endif
}

void MainWindow::onFileSelected(const QString& file) {
  ui->stackedWidget->setCurrentWidget(currentWorkWidget);
  currentWorkWidget->openFile(getCurrentDevices(), QFileInfo(file));
}

void MainWindow::onFileOpened(const QStringList files) {
  if (files.isEmpty() || files.first().isEmpty()) {
    return;
  }
  if (GlobalLiveController::getInstance().getController()) {  // Only if a project is already opened
    if (GlobalController::getInstance().getController()->isProjectOpening()) {
      return;
    }
    currentWorkWidget->onCloseProject();
  }
  ui->stackedWidget->setCurrentWidget(ui->pageProject);
  onFileSelected(files.first());
}

void MainWindow::onProjectLoaded() {
  activeProject = true;
  ui->buttonPlayback->setEnabled(true);
  emit currentWorkWidget->notifyAudioPlaybackActivated(audioPlayback);
}

void MainWindow::onProjectClosed() {
  ui->stackedWidget->setCurrentWidget(welcomeWidget);
  welcomeWidget->updateContent();
  activeProject = false;
  ui->buttonPlayback->setEnabled(false);
}

void MainWindow::onProjectNameAccepted(const QString& name) {
  ui->stackedWidget->setCurrentWidget(currentWorkWidget);
  currentWorkWidget->startNewProject(getCurrentDevices(), name);
}

void MainWindow::onExitApplicationAccepted() {
  currentWorkWidget->onCloseProject();
  exitDialog->close();
  exitDialog = nullptr;
  QApplication::exit();
}

void MainWindow::onExitApplicationRejected() {
  exitDialog->close();
  exitDialog = nullptr;
}

void MainWindow::onButtonStartNewProjectClicked() {
  NewProjectNameDialog* dialogName = new NewProjectNameDialog(this);
  connect(dialogName, &GenericDialog::notifyCancelClicked, dialogName, &GenericDialog::close);
  connect(dialogName, &NewProjectNameDialog::notifySetProjectName, this, &MainWindow::onProjectNameAccepted);
  dialogName->show();
}

void MainWindow::onButtonOpenProject() {
  const QString& path = QFileDialog::getOpenFileName(
      this, tr("Open %0 project").arg(QCoreApplication::applicationName()), getProjectsPath(),
      tr("%0 project (*.vah);;").arg(QCoreApplication::applicationName()));
  if (!path.isEmpty()) {
    onFileSelected(path);
  }
}

void MainWindow::onShowLogClicked() {
  logDialog->show();
  logDialog->raise();
}

void MainWindow::onButtonAboutClicked() {
  BackgroundContainer* container =
      new BackgroundContainer(aboutDialog, tr("About %0").arg(QCoreApplication::applicationName()), this, true);

  connect(container, &BackgroundContainer::notifyWidgetClosed, this, [=]() {
    container->hide();
    container->deleteLater();
    aboutDialog->hide();
    aboutDialog->setParent(this);
  });
  container->show();
  aboutDialog->show();
  aboutDialog->raise();
}

void MainWindow::onButtonSettingsClicked() {
  // GeneralSettingsDialog manages itself its lifecycle when we show it
  GeneralSettingsDialog* settingsDialog = new GeneralSettingsDialog(this);
  settingsDialog->show();
}

void MainWindow::showCredentialWindow() {
  // AuthenticationDialog manages itself its lifecycle when we show it
  AuthenticationDialog* dialog = new AuthenticationDialog(this);
  dialog->show();
}

void MainWindow::onResetDimensions(const unsigned panoWidth, const unsigned panoHeight) {
  currentWorkWidget->onCloseProject();
  ResizePanoramaDialog* dialog = new ResizePanoramaDialog(panoWidth, panoHeight, this);
  connect(dialog, &ResizePanoramaDialog::notifyPanoValuesSet, this, &MainWindow::onNewPanoSizeSet);
  dialog->show();
}

void MainWindow::onNewPanoSizeSet(const int width, const int height) {
  ui->stackedWidget->setCurrentWidget(currentWorkWidget);
  currentWorkWidget->openFile(getCurrentDevices(), QFileInfo(ProjectFileHandler::getInstance()->getFilename()), width,
                              height);
  emit notifySaveProject();
}

void MainWindow::onDropInputs(QDropEvent* e) {
  if (!e->mimeData()->hasUrls()) {
    e->setAccepted(false);
    return;
  }
  const QList<QUrl> urlList = e->mimeData()->urls();
  if (!urlList.size()) {
    e->setAccepted(false);
    return;
  }
  const QString file = urlList.first().toLocalFile();
  if (File::getTypeFromFile(file) != File::VAH) {
    e->setAccepted(false);
    showIncompatibleFileDialog();
    return;
  }
  onFileSelected(file);
  activateWindow();
}

void MainWindow::onEnableWindow() { setEnabled(true); }

void MainWindow::onDisableWindow() { setEnabled(false); }

void MainWindow::showIncompatibleFileDialog() {
  GenericDialog* cancelDialog = new GenericDialog(tr("Opening file error"), tr("This file format is not supported"),
                                                  GenericDialog::DialogMode::ACCEPT, this);
  cancelDialog->show();
}
