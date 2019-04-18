// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef OUTPUTDIALOG_HPP
#define OUTPUTDIALOG_HPP

#include <QDialog>

namespace Ui {
class OutputDialog;
}

/**
 * @brief Class used to display the output dialog.
 */
class OutputDialog : public QDialog {
  Q_OBJECT

 public:
  explicit OutputDialog(QWidget *parent, const bool init, unsigned &width, unsigned &height,
                        const QStringList &outputFormatList);
  ~OutputDialog();

 private slots:
  /**
   * @brief on_lineEditWidth_textChanged
   * @param newText
   */
  void on_lineEditWidth_textChanged(const QString &newText);
  /**
   * @brief on_lineEditHeight_textChanged
   * @param newText
   */
  void on_lineEditHeight_textChanged(const QString &newText);

 private:
  Ui::OutputDialog *ui;
  unsigned &m_width, &m_height;
};

#endif  // OUTPUTDIALOG_HPP
