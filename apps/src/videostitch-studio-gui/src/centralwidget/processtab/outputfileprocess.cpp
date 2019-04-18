// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "outputfileprocess.hpp"
#include "ui_outputfileprocess.h"

#include <QFileDialog>
#include <QMessageBox>

#include "videostitcher/postprodprojectdefinition.hpp"
#include "videostitcher/globalpostprodcontroller.hpp"
#include "libvideostitch-gui/centralwidget/formats/extensionhandlers/extensionhandler.hpp"
#include "libvideostitch-gui/centralwidget/formats/codecs/basicmpegcodec.hpp"
#include "libvideostitch-gui/mainwindow/msgboxhandlerhelper.hpp"
#include "libvideostitch-gui/mainwindow/outputfilehandler.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"
#include "libvideostitch-gui/utils/imagesorproceduralsonlyfilterer.hpp"
#include "libvideostitch-gui/utils/panoutilities.hpp"
#include "commands/outputsizechangedcommand.hpp"

#include "libvideostitch-base/file.hpp"

#include "libvideostitch/ptv.hpp"

#include <QStorageInfo>

static const unsigned int BITS_IN_BYTE(8);
static const unsigned int BYTES(1024);
static const unsigned int BYTES_IN_MB(BYTES* BYTES);
static const unsigned int BYTES_IN_GB(BYTES_IN_MB* BYTES);
static const unsigned int SIZE_SPINBOX_STEP(16);

OutputFileProcess::OutputFileProcess(QWidget* const parent)
    : IProcessWidget(parent),
      ui(new Ui::OutputFileProcess),
      extensionHandler(new ExtensionHandler()),
      currentFormat(LIBAV_WRITER_DEFAULT_CONTAINER),
      compressor(SignalCompressionCaps::createOwned()) {
  ui->setupUi(this);
  extensionHandler->init();
  ui->radioProcessAll->setAttribute(Qt::WA_LayoutUsesWidgetRect);
  ui->radioProcessSequence->setAttribute(Qt::WA_LayoutUsesWidgetRect);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->radioProcessSequence,
                                                        FeatureFilterer::PropertyToWatch::visible);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->radioProcessAll, FeatureFilterer::PropertyToWatch::visible);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->labelLength, FeatureFilterer::PropertyToWatch::visible);
  ImagesOrProceduralsOnlyFilterer::getInstance()->watch(ui->sequenceTime, FeatureFilterer::PropertyToWatch::visible);
  ui->spinWidth->setMinimum(VideoStitch::Util::MIN_PANO_WIDTH);
  ui->spinHeight->setMinimum(VideoStitch::Util::MIN_PANO_HEIGHT);
  ui->spinWidth->setSingleStep(SIZE_SPINBOX_STEP);
  ui->spinHeight->setSingleStep(SIZE_SPINBOX_STEP);
  ui->spinWidth->setKeyboardTracking(false);
  ui->spinHeight->setKeyboardTracking(false);
  ui->buttonSetOptimal->setToolTip(tr("Set the highest panorama size without distorsion."));
  connect(ui->buttonBrowse, &QPushButton::clicked, this, &OutputFileProcess::onBrowseFileClicked);
  connect(ui->buttonProcessNow, &QPushButton::clicked, this, &OutputFileProcess::onProcessClicked);
  connect(ui->buttonSendToBatch, &QPushButton::clicked, this, &OutputFileProcess::onSendToBatchClicked);
  connect(ui->lineFileName, &QLineEdit::textChanged, this, &OutputFileProcess::onFilenameChanged);
  connect(ui->buttonSetOptimal, &QPushButton::clicked, this, &OutputFileProcess::onOptimalSizeClicked);
  // Since these call reset on the stitcher controller, we want to emit very few signals (so editingFinished is better
  // than valueChanged)
  connect(ui->spinWidth, &QSpinBox::editingFinished, this, &OutputFileProcess::onWidthChanged, Qt::UniqueConnection);
  connect(ui->spinHeight, &QSpinBox::editingFinished, this, &OutputFileProcess::onHeightChanged, Qt::UniqueConnection);
  connect(ui->radioProcessAll, &QRadioButton::clicked, this, &OutputFileProcess::onProcessOptionChanged);
  connect(ui->radioProcessSequence, &QRadioButton::clicked, this, &OutputFileProcess::onProcessOptionChanged);
}

OutputFileProcess::~OutputFileProcess() {}

bool OutputFileProcess::checkProjectFileExists() {
  if (!ProjectFileHandler::getInstance()->getFilename().isEmpty()) {
    return true;
  }

  const QFlags<QMessageBox::StandardButton> buttons = QMessageBox::Save | QMessageBox::Cancel;
  const auto button = MsgBoxHandler::getInstance()->genericSync(
      tr("Your project is not saved.\nDo you want to save it before processing?"), QCoreApplication::applicationName(),
      QUESTION_ICON, buttons);
  if (button == QMessageBox::Save) {
    emit reqSavePtv();
  }
  // The user can still cancel
  return !ProjectFileHandler::getInstance()->getFilename().isEmpty();
}

void OutputFileProcess::reactToChangedProject() {
  currentFormat = project->getOutputVideoFormat();
  ui->lineFileName->blockSignals(true);
  ui->lineFileName->setText(
      QDir::toNativeSeparators(extensionHandler->handle(project->getOutputFilename(), currentFormat)));
  ui->lineFileName->blockSignals(false);
  const bool processSequence = project->getProcessSequence();
  ui->radioProcessAll->setChecked(!processSequence);
  ui->radioProcessSequence->setChecked(processSequence);
  // Size
  unsigned width = 0;
  unsigned height = 0;
  project->getImageSize(width, height);
  setValue(ui->spinWidth, width);
  setValue(ui->spinHeight, height);
}

void OutputFileProcess::onFileFormatChanged(const QString format) {
  const QString baseName = extensionHandler->stripBasename(ui->lineFileName->text(), currentFormat);
  currentFormat = format;
  updateFileName(baseName);
}

void OutputFileProcess::updateSequence(const QString start, const QString stop) {
  ui->sequenceTime->sequenceUpdated(start, stop);
}

void OutputFileProcess::onBrowseFileClicked() {
  QString videoPath = QFileDialog::getSaveFileName(this, tr("Select the output path"), QDir::currentPath(),
                                                   QString("*.%0").arg(currentFormat));
  if (videoPath.isEmpty()) {
    return;
  }
  const QString outputDirectory = QFileInfo(videoPath).absolutePath();
  while (!outputDirectory.isEmpty() && !QFileInfo(outputDirectory).isWritable() && QFileInfo(videoPath).isWritable()) {
    QString text = tr("You do not have the right to write in %0. Do you want to select another directory?");
    if (MsgBoxHandler::getInstance()->genericSync(text.arg(videoPath), tr("Wrong permissions"), WARNING_ICON,
                                                  QMessageBox::Retry | QMessageBox::No) == QMessageBox::No) {
      return;
    }
  }
#ifdef Q_OS_MAC
  // TODO: Workaround for bug QTBUG-44227
  const QFileInfo info(videoPath);
  videoPath = info.absolutePath() + QDir::separator() + info.completeBaseName();
#endif
  updateFileName(videoPath);
}

void OutputFileProcess::onShowFolderClicked() {
  QString file = ui->lineFileName->text();
  if (file.contains("-" + FRAME_EXTENSION)) {
    file.replace(file.lastIndexOf(FRAME_EXTENSION), FRAME_EXTENSION.size(), "0");
  }
  File::showInShellExporer(file);
}

void OutputFileProcess::onProcessClicked() {
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("Process video now");
  // Are there any pending modifications in the project?
  if (!checkPendingModifications()) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        "Process cancelled because of pending modifications");
    return;
  }

  // Is the project saved into a file?
  if (!checkProjectFileExists()) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        "Process cancelled because of non existing project file");
    return;
  }

  // Does file exists?
  if (!checkFileExists()) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        "Process cancelled because of the user doesn't want to replace his output file");
    return;
  }

  // Can it be written?
  if (!checkFileIsWritable()) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        "Process cancelled because of the output file is not writable");
    return;
  }

  // The file system can handle the file size?
  if (!checkFileSystem()) {
    VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
        "Process cancelled because of file system limitation");
    return;
  }

  frameid_t firstFrame = project->getFirstFrame();
  frameid_t lastFrame = project->getLastFrame();
  if (ui->radioProcessAll->isChecked()) {
    firstFrame = 0;
    lastFrame = GlobalPostProdController::getInstance().getController()->getLastStitchableFrame();
  }
  emit reqProcessOutput(firstFrame, lastFrame);
}

bool OutputFileProcess::checkFileExists() {
  if (QFile::exists(ui->lineFileName->text())) {
    const QFlags<QMessageBox::StandardButton> buttons = QMessageBox::Yes | QMessageBox::Cancel;
    return QMessageBox::Yes == MsgBoxHandler::getInstance()->genericSync(
                                   tr("The output file already exists.\nDo you want to overwrite it?"),
                                   tr("File already exists"), QUESTION_ICON, buttons);
  }
  return true;
}

bool OutputFileProcess::checkFileIsWritable() {
  const QFileInfo fileInfo(ui->lineFileName->text());
  const QDir directory = fileInfo.dir();
  if (!directory.exists() || (fileInfo.exists() && !fileInfo.isWritable())) {
    MsgBoxHandler::getInstance()->genericSync(
        tr("The output file is not accessible at %0.\nPlease select an existing path.").arg(directory.absolutePath()),
        tr("File not writeable"), CRITICAL_ERROR_ICON, QMessageBox::Ok);
    return false;
  }
  return true;
}

bool OutputFileProcess::checkPendingModifications() {
  if (project->hasLocalModifications()) {
    const QFlags<QMessageBox::StandardButton> buttons = QMessageBox::Save | QMessageBox::Cancel;
    const auto button = MsgBoxHandler::getInstance()->genericSync(
        tr("Your project contains modifications.\nDo you want to save your changes before processing?"),
        QCoreApplication::applicationName(), QUESTION_ICON, buttons);
    if (button == QMessageBox::Cancel) {
      return false;
    }
    emit reqSavePtv();
    return true;
  }
  return true;
}

bool OutputFileProcess::checkFileSystem() {
  bool returnValue = true;

  if (VideoStitch::OutputFormat::isVideoFormat(
          VideoStitch::OutputFormat::getEnumFromString(project->getOutputVideoFormat()))) {
    const QStorageInfo storageInfo(ui->lineFileName->text());
    // If the information cannot be retrieved from the drive, we should not check the available size.
    if (!storageInfo.isValid() || !storageInfo.isReady()) {
      VideoStitch::Helper::LogManager::getInstance()->writeToLogFile(
          "Warning: The information for the target drive cannot be accessed.");
      return returnValue;
    }

    // Get the available space
    const qint64 fileSize = getEstimatedFileSize();
    const qint64 freeSpace = storageInfo.bytesAvailable();
    const qint64 maxSpace = storageInfo.bytesTotal();
    if (fileSize >= freeSpace) {
      const QFlags<QMessageBox::StandardButton> buttons = QMessageBox::Yes | QMessageBox::No;
      returnValue &= (QMessageBox::Yes == MsgBoxHandler::getInstance()->genericSync(
                                              tr("The output file exceeds the available disk space (estimated file "
                                                 "size %0 MB versus %1 MB of free disk space).\n"
                                                 "Do you want to proceed anyway?")
                                                  .arg(double(fileSize) / BYTES_IN_MB)
                                                  .arg(double(freeSpace) / BYTES_IN_MB),
                                              tr("Low disk space"), WARNING_ICON, buttons));
    }
    // Do not display the message box if returnValue is already False
    if (fileSize >= maxSpace && returnValue) {
      const QFlags<QMessageBox::StandardButton> buttons = QMessageBox::Yes | QMessageBox::No;
      returnValue &=
          (QMessageBox::Yes ==
           MsgBoxHandler::getInstance()->genericSync(
               tr("The output file will be split into multiple files of %0 GB, due to file system limitations.\n\
                                                                     Do you want to proceed anyway?")
                   .arg(double(maxSpace) / BYTES_IN_GB),
               tr("File system limitation"), WARNING_ICON, buttons));

      if (returnValue) {
        project->addOutputFileChunkSize((maxSpace / 100) * 95);
      }
    } else {
      project->removeOutputFileChunkSize();
    }
  }
  return returnValue;
}

qint64 OutputFileProcess::getEstimatedFileSize() const {
  int bitRate = 0;
  if (project->hasAudioConfiguration()) {
    bitRate += project->getOutputAudioBitrate();
  }
  if (project->getOutputVideoBitrate() != 0) {
    bitRate += project->getOutputVideoBitrate();
  } else {
    bitRate += LIBAV_WRITER_DEFAULT_BITRATE;
  }

  frameid_t frameNumber = 0;
  if (ui->radioProcessAll->isChecked()) {
    frameNumber = GlobalPostProdController::getInstance().getController()->getLastStitchableFrame();
  } else {
    frameNumber = project->getLastFrame() - project->getFirstFrame();
  }
  const VideoStitch::FrameRate frameRate = GlobalController::getInstance().getController()->getFrameRate();
  qint64 fileSize = frameNumber * frameRate.den / frameRate.num * bitRate / BITS_IN_BYTE;
  fileSize += fileSize / 20;  // Add 5% to the file size
  return fileSize;
}

QString OutputFileProcess::getBaseFileName() const {
  const QString base = extensionHandler->stripBasename(ui->lineFileName->text(), currentFormat);
  return QDir::fromNativeSeparators(base);
}

void OutputFileProcess::updateFileName(const QString path) {
  ui->lineFileName->setText(extensionHandler->handle(path, currentFormat));
  project->setOutputFilename(QDir::fromNativeSeparators(path));
}

void OutputFileProcess::onSendToBatchClicked() { emit reqSendToBatch(true); }

void OutputFileProcess::onFilenameChanged(const QString path) {
  ui->buttonProcessNow->setEnabled(!path.isEmpty());
  if (!path.isEmpty()) {
    project->setOutputFilename(getBaseFileName());
  }
}

void OutputFileProcess::setPanoramaSize(const unsigned width, const unsigned height) {
  setValue(ui->spinWidth, width);
  setValue(ui->spinHeight, height);
  project->setPanoramaSize(width, height);
  emit panoSizeChanged(width, height);
  emit reqReset(compressor->add());
}

void OutputFileProcess::onOptimalSizeClicked() {
  unsigned height = 0;
  unsigned width = 0;
  project->getOptimalSize(width, height);
  saveUndoPanoValue(width, height);
}

void OutputFileProcess::onHeightChanged() {
  const VideoStitch::Util::PanoSize size = VideoStitch::Util::calculateSizeFromHeight(ui->spinHeight->value());
  saveUndoPanoValue(size.width, size.height);
}

void OutputFileProcess::onWidthChanged() {
  const VideoStitch::Util::PanoSize size = VideoStitch::Util::calculateSizeFromWidth(ui->spinWidth->value());
  saveUndoPanoValue(size.width, size.height);
}

void OutputFileProcess::onProcessOptionChanged() { project->setProcessSequence(ui->radioProcessSequence->isChecked()); }

void OutputFileProcess::setValue(QSpinBox* box, const int value) {
  box->blockSignals(true);
  box->setValue(value);
  box->blockSignals(false);
}

void OutputFileProcess::saveUndoPanoValue(const unsigned newWidth, const unsigned newHeight) {
  if (project != nullptr) {
    unsigned oldWidth = 0;
    unsigned oldHeight = 0;
    project->getImageSize(oldWidth, oldHeight);
    OutputSizeChangedCommand* command = new OutputSizeChangedCommand(oldWidth, oldHeight, newWidth, newHeight, this);
    qApp->findChild<QUndoStack*>()->push(command);
  }
}
