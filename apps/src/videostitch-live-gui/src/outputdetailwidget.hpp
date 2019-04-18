// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_outputdetailwidget.h"

#include <QWidget>

class GenericDialog;
class LiveOutputFactory;

class OutputDetailWidget : public QWidget, public Ui::OutputDetailWidgetClass {
  Q_OBJECT

 public:
  explicit OutputDetailWidget(LiveOutputFactory* output, QWidget* parent = nullptr);
  ~OutputDetailWidget();

  LiveOutputFactory* getOutput() const;
  void allowsRemoving(bool allow);

 signals:
  void notifyDeleteOutput(const QString& id);

 private slots:
  void updateDetails();
  void onDeleteClicked();
  void onDeleteAccepted();
  void onDeleteRejected();

 private:
  LiveOutputFactory* output;
  GenericDialog* deleteDialog;
};
