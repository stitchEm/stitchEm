// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "ui_configoutputswidget.h"
#include "iconfigurationvalue.hpp"

#include <QPointer>

class QPushButton;
class LiveOutputFactory;
class NewOutputWidget;
class GenericTableWidget;
class LiveVideoStitcher;
class OutputDetailWidget;
class OutputConfigurationWidget;

class ConfigOutputsWidget : public IConfigurationCategory, public Ui::ConfigOutputsWidgetClass {
  Q_OBJECT

 public:
  explicit ConfigOutputsWidget(QWidget* const parent = nullptr);
  ~ConfigOutputsWidget();

  void showOutputList();
  void toggleOutput(const QString& id);
  void removeOutputDevices();
  void restoreMainPage();

  NewOutputWidget* getNewOutputWidget() const;

  /**
   * @brief Enables or disables actions on the output list
   */
  void updateEditability(bool outputIsActivated, bool algorithmIsActivated);

 private:
  void addOutputDevices(QList<LiveOutputFactory*> liveOutputs);
  /**
   * @brief Adds a configuration value in the list
   * @param type Output type
   * @param id A unique output id
   */
  void addSingleConfigurationValue(LiveOutputFactory* output);

  /**
   * @brief Creates and setups a button for adding new outputs
   */
  void createAddButton();

  void updateConfiguration(LiveOutputFactory* output);

  OutputDetailWidget* getListItemById(const QString& id) const;

  GenericTableWidget* outputList;
  NewOutputWidget* newOutputWidget;
  QPointer<QPushButton> buttonAddOutput;
  bool hasEmptyOutputs;

 public slots:
  void configureOutputById(const QString& id);
  void onButtonAddOutputClicked();

 protected slots:
  void onItemClicked(int row, int column);
  void onOutputIdChanged(const QString& oldId, const QString& newId);

 protected:
  virtual void reactToChangedProject();
  virtual void reactToClearedProject();
  OutputConfigurationWidget* getConfigurationWidgetForId(const QString& outputId);

 signals:
  void injectProject(LiveProjectDefinition* project);
  void projectCleared();
  void injectStitcher(LiveVideoStitcher* stitcher);
  void notifyStartAddingOutput();
  void notifyOutputIdChanged(const QString& oldName, const QString& newName);
  void reqChangeOutputConfig(const QString& Name);
  void reqRemoveOutput(const QString& id);
};
