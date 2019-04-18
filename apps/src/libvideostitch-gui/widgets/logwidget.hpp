// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef LOGWIDGET_HPP
#define LOGWIDGET_HPP

#include <QWidget>
#include <QEvent>

namespace Ui {
class LogWidget;
}

class VS_GUI_EXPORT LogWidget : public QWidget {
  Q_OBJECT

 public:
  explicit LogWidget(QWidget *parent = 0, bool displayControls = true);
  ~LogWidget();
 public slots:
  void showControls();
  void hideControls();
  void setLogLevel(const int index);
  /**
   * @brief logMessage Writes an error passed as a QString into the log.
   * @param message Message to write into the log.
   */
  void logMessage(const QString &message);
 private slots:
  void on_logLevelBox_currentIndexChanged(int index);
  /**
   * @brief Shows the contextual menu
   * @param pt position of the contextual menu.
   */
  void showTextAreaMenu(const QPoint &pt);
 signals:
  void notifyLogLevelChanged(const int level);

 private:
  void toggleConnect(bool connectionState, const char *signal);
  Ui::LogWidget *ui;
};

#endif  // LOGWIDGET_HPP
