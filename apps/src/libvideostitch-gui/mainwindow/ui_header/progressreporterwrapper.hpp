// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PROGRESSREPORTERWRAPPER_HPP
#define PROGRESSREPORTERWRAPPER_HPP

#include "libvideostitch/algorithm.hpp"

#include <QProgressBar>

#include <atomic>

class VS_GUI_EXPORT ProgressReporterWrapper : public QProgressBar,
                                              public VideoStitch::Util::Algorithm::ProgressReporter {
  Q_OBJECT

 public:
  explicit ProgressReporterWrapper(QWidget *parent = 0)
      : QProgressBar(parent), VideoStitch::Util::Algorithm::ProgressReporter(), isCanceled(false), valueToDisplay(0) {
    setRange(0, 100);
    connect(this, &ProgressReporterWrapper::reqSetValue, this, &ProgressReporterWrapper::setValue);
  }

  bool hasBeenCanceled() { return isCanceled; }

 signals:
  void reqSetValue(int);
  void reqProgressMessage(QString);

 public slots:
  /**
   * @brief notify thread-safe function to update the progress bar.
   */
  bool notify(const std::string &message, double percent) {
    // If we have several running algorithms linked to the progress bar,
    // valueToDisplay assures us that the progress bar will not go back
    valueToDisplay = qMax(valueToDisplay, qRound(percent));
    emit reqSetValue(valueToDisplay);
    emit reqProgressMessage(QString::fromStdString(message));
    return isCanceled;
  }

  void cancel() { isCanceled = true; }

  void reset() {
    isCanceled = false;
    valueToDisplay = 0;
  }

 private:
  std::atomic<bool> isCanceled;
  int valueToDisplay;
};

#endif  // PROGRESSREPORTERWRAPPER_HPP
