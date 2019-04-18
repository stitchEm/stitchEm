// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef EXTRACTDIALOG_HPP
#define EXTRACTDIALOG_HPP

#include <QDialog>
#include <QToolTip>

namespace Ui {
class ExtractDialog;
}

/**
 * @brief Class used to display the extract dialog.
 */
class ExtractDialog : public QDialog {
  Q_OBJECT

 public:
  explicit ExtractDialog(QWidget *parent, QString &dir);
  /**
   * @brief Sets the last directory used by the last calibration request. This calibration directory is shared accross
   * the Extract Dialog instances.
   * @return Last used directory.
   */
  static QString getLastCalibrationDirectory();
  /**
   * @brief Resets the last directory used by the last calibration request.
   *        This calibration directory is shared accross the Extract Dialog instances.
   *        The value used for reset is an empty QString().
   */
  static void resetLastCalibrationDirectory();
  ~ExtractDialog();

 private slots:
  /**
   * @brief Displays a QFileDialog to select the output directory.
   */
  void on_browseButton_clicked();

  void on_buttonBox_accepted();

 private:
  /**
   * @brief Sets the last directory used by the last calibration request. This calibration directory is shared across
   * the Extract Dialog instances.
   * @param directory Last used directory.
   */
  static void setLastCalibrationDirectory(const QString &directory);

  static QString lastCalibrationDirectory;

  Ui::ExtractDialog *ui;
  QString &m_dir;
};

#endif  // EXTRACTDIALOG_HPP
