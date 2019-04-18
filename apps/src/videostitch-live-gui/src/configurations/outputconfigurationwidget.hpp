// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "iconfigurationvalue.hpp"

class LiveOutputFactory;

class OutputConfigurationWidget : public IConfigurationCategory {
  Q_OBJECT

 public:
  explicit OutputConfigurationWidget(QWidget* parent = nullptr);
  virtual ~OutputConfigurationWidget() {}

  virtual LiveOutputFactory* getOutput() const = 0;

  virtual void toggleWidgetState() = 0;

 signals:
  void reqChangeOutputId(const QString& oldId, const QString& newId);
  void reqChangeOutputConfig(const QString& Id);

 protected:
  virtual void fillWidgetWithValue();
};
