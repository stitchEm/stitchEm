// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef BACKGROUNDCONTAINER_HPP
#define BACKGROUNDCONTAINER_HPP

#include <QFrame>

namespace Ui {
class BackgroundContainer;
}

class BackgroundContainer : public QFrame {
  Q_OBJECT

 public:
  explicit BackgroundContainer(QWidget* contained, const QString& title, QWidget* const parent = nullptr,
                               const bool closeButton = true);
  ~BackgroundContainer();

  QWidget* getContainedWidget() const;

 protected:
  virtual bool eventFilter(QObject*, QEvent* event);

 private:
  QScopedPointer<Ui::BackgroundContainer> ui;
  QWidget* containedWidget;

 signals:
  void notifyWidgetClosed();
};

#endif  // BACKGROUNDCONTAINER_HPP
