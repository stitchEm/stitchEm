// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef INPUTTYPEWIDGET_HPP
#define INPUTTYPEWIDGET_HPP

#include "libvideostitch-gui/utils/inputformat.hpp"
#include "libvideostitch-gui/widgets/stylablewidget.hpp"
#include <QWidget>

namespace Ui {
class InputTypeWidget;
}
class QListWidgetItem;

class InputTypeWidget : public QWidget {
  Q_OBJECT
  Q_MAKE_STYLABLE

 public:
  explicit InputTypeWidget(QWidget* parent = nullptr);
  ~InputTypeWidget();

 signals:
  void inputTypeSelected(VideoStitch::InputFormat::InputFormatEnum inputType);

 private:
  void addInputType(VideoStitch::InputFormat::InputFormatEnum inputType);
  static QVector<VideoStitch::InputFormat::InputFormatEnum> getAvailableInputs();

 private slots:
  void selectInputType(QListWidgetItem* item);

 private:
  QScopedPointer<Ui::InputTypeWidget> ui;
};

#endif  // INPUTTYPEWIDGET_HPP
