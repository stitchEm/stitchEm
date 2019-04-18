// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef EMORVIEW_H
#define EMORVIEW_H

#include <QWidget>

class VS_GUI_EXPORT EmorView : public QWidget {
  Q_OBJECT
 public:
  explicit EmorView(QWidget *parent = 0);
  void paintEvent(QPaintEvent *event);

  void setEmorParams(double eA, double eB, double eC, double eD, double eE) {
    emorA = eA;
    emorB = eB;
    emorC = eC;
    emorD = eD;
    emorE = eE;
    update();
  }

 private:
  double emorA, emorB, emorC, emorD, emorE;
};

#endif  // EMORVIEW_H
