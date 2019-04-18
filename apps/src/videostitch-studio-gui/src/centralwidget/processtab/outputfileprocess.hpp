// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QWidget>
#include "iprocesswidget.hpp"
#include "libvideostitch/config.hpp"

namespace Ui {
class OutputFileProcess;
}

class ExtensionHandler;
class QSpinBox;

/**
 * @brief Class for file output configurations (name, path, length, output size, etc)
 */
class OutputFileProcess : public IProcessWidget {
  Q_OBJECT

 public:
  explicit OutputFileProcess(QWidget* const parent = nullptr);
  ~OutputFileProcess();
  void setPanoramaSize(const unsigned width, const unsigned height);

 public slots:
  void onFileFormatChanged(const QString format);
  void updateSequence(const QString start, const QString stop);

 signals:
  void reqSendToBatch(bool saveACopy);
  void reqSavePtv();
  void reqProcessOutput(const frameid_t first, const frameid_t last);
  void panoSizeChanged(unsigned int width, unsigned int height);
  void reqReset(SignalCompressionCaps* compressor);

 protected:
  virtual void reactToChangedProject() override;

 private:
  bool checkFileSystem();
  bool checkFileExists();
  bool checkFileIsWritable();
  bool checkPendingModifications();
  bool checkProjectFileExists();
  qint64 getEstimatedFileSize() const;
  QString getBaseFileName() const;
  void updateFileName(const QString path);
  void saveUndoPanoValue(const unsigned newWidth, const unsigned newHeight);
  void setValue(QSpinBox* box, const int value);

 private slots:
  void onBrowseFileClicked();
  void onShowFolderClicked();
  void onProcessClicked();
  void onSendToBatchClicked();
  void onFilenameChanged(const QString path);
  void onOptimalSizeClicked();
  void onHeightChanged();
  void onWidthChanged();
  void onProcessOptionChanged();

 private:
  QScopedPointer<Ui::OutputFileProcess> ui;
  QScopedPointer<ExtensionHandler> extensionHandler;
  QString currentFormat;
  std::shared_ptr<SignalCompressionCaps> compressor;
};
