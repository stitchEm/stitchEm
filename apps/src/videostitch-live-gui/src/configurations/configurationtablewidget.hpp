// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef CONFIGURATIONTABLEWIDGET_H
#define CONFIGURATIONTABLEWIDGET_H

#include <QWidget>
#include "ui_configurationtablewidget.h"

enum class ConfigIdentifier { CONFIG_MAIN, CONFIG_OUTPUT };

class ConfigurationTableWidget : public QWidget, public Ui::ConfigurationTableWidgetClass {
  Q_OBJECT

 public:
  explicit ConfigurationTableWidget(ConfigIdentifier identifier, const QString& title, QWidget* const parent = nullptr);

  ~ConfigurationTableWidget();

  ConfigIdentifier getConfigIdentifier() const;

 private:
  ConfigIdentifier configIdentifier;
};

#endif  // CONFIGURATIONTABLEWIDGET_H
