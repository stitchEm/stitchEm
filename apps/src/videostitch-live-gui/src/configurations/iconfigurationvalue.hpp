// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/widgets/stylablewidget.hpp"

class GenericDialog;
class LiveProjectDefinition;
class QHBoxLayout;
class QPushButton;
class QSpacerItem;

class IConfigurationCategory : public QWidget {
  Q_OBJECT
  Q_MAKE_STYLABLE

 public:
  enum class Mode {
    Undefined,
    CreationInPopup,  // Button save
    CreationInStack,  // Button back and button save (enabled if hasValidConfiguration)
    Edition,          // Button back and button save (disabled when starting)
    View              // Button back
  };

 public:
  explicit IConfigurationCategory(QWidget* const parent = nullptr);
  virtual ~IConfigurationCategory() {}

  /**
   * @brief Set the GUI layout for the mode
   */
  void changeMode(Mode newMode);
  /**
   * @brief Revert the modified data and update the GUI
   */
  void restore();
  /**
   * @brief Save the modified data and update the GUI
   * @return Returns true if the data were saved synchronously, false otherwise.
   */
  virtual bool save();

 public slots:
  void setProject(LiveProjectDefinition* project);
  void clearProject();

 signals:
  void saved();
  void reqBack();

 protected:
  virtual void reactToChangedProject() {}
  virtual void reactToClearedProject() {}
  virtual void updateAfterChangedMode() {}
  virtual void saveData() {}     // Implement this for Creation or Edition modes
  virtual void restoreData() {}  // Implement this for Edition mode if the widget is persistent
  virtual bool hasValidConfiguration() const { return true; }
  // TODO: hack for ConfigOutputsWidget, find a way to remove this
  void displayConfigInTheView(bool display);

 protected slots:
  void onConfigurationChanged();

 private slots:
  void onButtonBackClicked();
  void onSaveCancelled();
  void onSaveAccepted();

 protected:
  LiveProjectDefinition* projectDefinition;
  QHBoxLayout* buttonsLayout;
  Mode mode;

 private:
  void cleanDialog();
  QPushButton* buttonBack;
  QPushButton* buttonSave;
  QSpacerItem* spacer;
  GenericDialog* savingDialog;
};
