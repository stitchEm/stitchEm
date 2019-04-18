// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "processtabwidget.hpp"
#include "ui_processtabwidget.h"

#include "processtab/outputfileprocess.hpp"
#include "processtab/videoprocess.hpp"
#include "processtab/audioprocess.hpp"
#include "processtab/iprocesswidget.hpp"
#include "dialogs/processprogressdialog.hpp"
#include "videostitcher/postprodprojectdefinition.hpp"
#include "videostitcher/globalpostprodcontroller.hpp"

#include "libvideostitch-gui/mainwindow/vssettings.hpp"
#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"
#include "libvideostitch-gui/mainwindow/statemanager.hpp"
#include "libvideostitch-gui/utils/pluginshelpers.hpp"
#include "libgpudiscovery/genericDeviceInfo.hpp"

#include <QSpacerItem>

static const unsigned int LOG_LEVEL(3);

ProcessTabWidget::ProcessTabWidget(QWidget* const parent)
    : QFrame(parent),
      ui(new Ui::ProcessTabWidget),
      outputFileProcessWidget(new OutputFileProcess(this)),
      videoProcessWidget(new VideoProcess(this)),
      audioProcessWidget(new AudioProcess(this)) {
  ui->setupUi(this);
  // File output and process
  addProcessWidget(outputFileProcessWidget);
  // Video format and encoding
  addProcessWidget(videoProcessWidget);
  // Audio input and encoding
  addProcessWidget(audioProcessWidget);
  // Add a spacer at the end
  ui->content->layout()->addItem(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Expanding));
  // Connect speciphic signals
  connect(outputFileProcessWidget, &OutputFileProcess::reqSendToBatch, this, &ProcessTabWidget::reqSendToBatch);
  connect(outputFileProcessWidget, &OutputFileProcess::reqSavePtv, this, &ProcessTabWidget::reqSavePtv);
  connect(outputFileProcessWidget, &OutputFileProcess::reqProcessOutput, this, &ProcessTabWidget::processOutput);
  connect(outputFileProcessWidget, &OutputFileProcess::panoSizeChanged, this, &ProcessTabWidget::panoSizeChanged);
  connect(outputFileProcessWidget, &OutputFileProcess::reqReset, this, &ProcessTabWidget::reqReset);
  connect(videoProcessWidget, &VideoProcess::reqChangeFormat, outputFileProcessWidget,
          &OutputFileProcess::onFileFormatChanged);
  connect(videoProcessWidget, &VideoProcess::reqChangeFormat, audioProcessWidget, &AudioProcess::onFileFormatChanged);
  connect(audioProcessWidget, &AudioProcess::reqChangeAudioInput, this, &ProcessTabWidget::reqChangeAudioInput);
  connect(this, &ProcessTabWidget::reqUpdateSequence, outputFileProcessWidget, &OutputFileProcess::updateSequence);
  StateManager::getInstance()->registerObject(this);
}

ProcessTabWidget::~ProcessTabWidget() { delete ui; }

void ProcessTabWidget::onProjectOpened(ProjectDefinition* project) {
  emit reqProjectOpened(qobject_cast<PostProdProjectDefinition*>(project));
}

void ProcessTabWidget::clearProject() { emit reqProjectCleared(); }

void ProcessTabWidget::addProcessWidget(IProcessWidget* const processWidget) {
  ui->content->layout()->addWidget(processWidget);
  connect(this, &ProcessTabWidget::reqProjectOpened, processWidget, &IProcessWidget::setProject);
  connect(this, &ProcessTabWidget::reqProjectCleared, processWidget, &IProcessWidget::clearProject);
}

QStringList ProcessTabWidget::getGPUs() const {
  QStringList devices;
  for (int deviceId : VSSettings::getSettings()->getDevices()) {
    VideoStitch::Discovery::DeviceProperties prop;
    if (VideoStitch::Discovery::getDeviceProperties(deviceId, prop)) {
      devices.append(QString::number(deviceId));
    }
  }
  return devices;
}

void ProcessTabWidget::processOutput(const frameid_t first, const frameid_t last) {
  emit reqChangeState(GUIStateCaps::frozen);
  QStringList args;
  args << "-i" << ProjectFileHandler::getInstance()->getFilename();
  args << "-d" << getGPUs().join(",");
  args << "-p" << VideoStitch::Plugin::getCorePluginFolderPath();
  auto path = VideoStitch::Plugin::getGpuCorePluginFolderPath();
  if (!path.isEmpty()) {
    args << "-p" << path;
  }
  args << "-f" << QString::number(first);
  args << "-l" << QString::number(last);
  args << "-v" << QString::number(LOG_LEVEL);
  ProcessProgressDialog pdial(args, first, last, this);
#ifdef Q_OS_WIN
  emit reqStartProgress();
  connect(&pdial, &ProcessProgressDialog::reqChangeProgressState, this, &ProcessTabWidget::reqChangeProgressState);
  connect(&pdial, &ProcessProgressDialog::reqChangeProgressValue, this, &ProcessTabWidget::reqChangeProgressValue);
#endif
  pdial.exec();
#ifdef Q_OS_WIN
  emit reqStopProgress();
#endif
  emit reqChangeState(GUIStateCaps::stitch);
}
