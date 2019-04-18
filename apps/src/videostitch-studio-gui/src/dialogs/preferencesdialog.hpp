// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef PREFERENCESDIALOG_H
#define PREFERENCESDIALOG_H

#include <QDialog>

#include "libvideostitch-gui/widgets/vspathedit.hpp"

namespace Ui {
class PreferencesDialog;
}

/**
 * @brief Dialog used to display the preference settings to the user.
 */
class PreferencesDialog : public QDialog {
  Q_OBJECT

 public:
  /**
   * @brief Get the available languages
   * @return QStringList with the languages and their keyword
   */
  static QMap<QString, QString> getLanguages();

 public:
  explicit PreferencesDialog(QWidget *parent, QVector<int> &deviceIds, QString &language);
  ~PreferencesDialog();

 private slots:
  void onSelectedGpusChanged();
  /**
   * @brief Slot called when the language has been changed
   * @param newText New language keyword
   */
  void on_languageComboBox_currentIndexChanged(const QString &newText);
  /**
   * @brief Slot called when the user pushes the button "ok"
   */
  void on_buttonBox_accepted();

 private:
  /**
   * @brief show or hide the global warning icon asking to restart the application
   */
  void setWarning(bool warning);

  Ui::PreferencesDialog *ui;
  QVector<int> &m_deviceIds;
  QString &m_language;

  QImage warning;
  QImage emptyImage;
};

#endif  // PREFERENCESDIALOG_H
