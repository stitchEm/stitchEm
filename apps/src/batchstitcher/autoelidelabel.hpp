// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef AUTOELIDELABEL_HPP
#define AUTOELIDELABEL_HPP

#include <QFontMetrics>
#include <QLabel>

class AutoElideLabel : public QLabel {
  Q_OBJECT
 public:
  explicit AutoElideLabel(Qt::TextElideMode elideMode = Qt::ElideMiddle, QWidget *parent = 0) : QLabel(parent) {
    this->elideMode = elideMode;
  }

  QString text() const { return content; }
 public slots:
  void setText(const QString &newText) {
    content = newText;
    updateText();
  }

 private:
  void showEvent(QShowEvent *event) {
    QLabel::showEvent(event);
    updateText();
  }

  void resizeEvent(QResizeEvent *event) {
    QLabel::resizeEvent(event);
    updateText();
  }

  void updateText() { QLabel::setText(fontMetrics().elidedText(content, elideMode, width())); }

  QString content;
  Qt::TextElideMode elideMode;
};

#endif  // AUTOELIDELABEL_HPP
