// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videostitcher/liveinputstream.hpp"
#include "configurations/inputconfigurationwidget.hpp"

namespace Ui {
class MultiInputConfiguration;
}
class NewInputItemWidget;

class MultiInputConfiguration : public InputConfigurationWidget {
  Q_OBJECT

 public:
  explicit MultiInputConfiguration(std::shared_ptr<const LiveInputStream> liveInput, QWidget* const parent = nullptr);
  ~MultiInputConfiguration();

  bool save() override;

 public slots:
  void onInputTestResult(const int id, const bool result, qint64 width, qint64 height);

 protected:
  void saveData() override;
  void reactToChangedProject() override;

 private:
  void addInputWidget(QString inputName, int index);

 private slots:
  void onUrlsValidityChanged();
  void onTestAllInputs();
  void changeNbOfInputs(int newNbOfInputs);

 signals:
  void notifyTestActivated(const int id, const QString name);
  void notifyInputResult(const bool success);

 private:
  bool streamsAreValid();
  void adjustListSize();
  QVector<NewInputItemWidget*> getItemWidgets() const;
  QScopedPointer<Ui::MultiInputConfiguration> ui;
  int inputsChecked;
  std::shared_ptr<const LiveInputStream> templateInput;  // This input is usefull only to display its parameters
};
