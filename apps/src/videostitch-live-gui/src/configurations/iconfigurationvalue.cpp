// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "iconfigurationvalue.hpp"
#include "widgetsmanager.hpp"
#include "generic/genericdialog.hpp"
#include "guiconstants.hpp"

IConfigurationCategory::IConfigurationCategory(QWidget* const parent)
    : QWidget(parent),
      projectDefinition(nullptr),
      buttonsLayout(new QHBoxLayout()),
      mode(Mode::Undefined),
      buttonBack(new QPushButton(this)),
      buttonSave(new QPushButton(this)),
      spacer(new QSpacerItem(0, 0, QSizePolicy::Expanding, QSizePolicy::Fixed)),
      savingDialog(nullptr) {
  connect(buttonBack, &QPushButton::clicked, this, &IConfigurationCategory::onButtonBackClicked);
  connect(buttonSave, &QPushButton::clicked, this, &IConfigurationCategory::save);
  buttonBack->setObjectName("buttonBack");
  buttonSave->setObjectName("buttonSave");
  buttonBack->setProperty("vs-button-medium", true);
  buttonSave->setProperty("vs-button-medium", true);
  buttonsLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);
  buttonsLayout->setContentsMargins(0, 0, 0, 0);
  buttonsLayout->setSpacing(BUTTON_SPACING);
  buttonsLayout->addSpacerItem(spacer);
  buttonsLayout->addWidget(buttonBack);
  buttonsLayout->addWidget(buttonSave);
  changeMode(Mode::Undefined);
}

void IConfigurationCategory::changeMode(IConfigurationCategory::Mode newMode) {
  mode = newMode;
  buttonBack->setVisible(mode == Mode::Edition || mode == Mode::View || mode == Mode::CreationInStack);
  buttonSave->setVisible(mode == Mode::Edition || mode == Mode::CreationInPopup || mode == Mode::CreationInStack);
  buttonSave->setEnabled((mode == Mode::CreationInPopup || mode == Mode::CreationInStack) && hasValidConfiguration());
  updateAfterChangedMode();
}

void IConfigurationCategory::restore() {
  restoreData();
  buttonSave->setEnabled(false);
  emit reqBack();
}

bool IConfigurationCategory::save() {
  saveData();
  buttonSave->setEnabled(false);
  emit saved();
  return true;
}

void IConfigurationCategory::setProject(LiveProjectDefinition* project) {
  projectDefinition = project;
  reactToChangedProject();
  buttonSave->setEnabled((mode == Mode::CreationInPopup || mode == Mode::CreationInStack) && hasValidConfiguration());
}

void IConfigurationCategory::clearProject() {
  projectDefinition = nullptr;
  reactToClearedProject();
}

void IConfigurationCategory::displayConfigInTheView(bool display) { buttonBack->setVisible(!display); }

void IConfigurationCategory::onButtonBackClicked() {
  if (buttonSave->isEnabled()) {
    if (savingDialog == nullptr) {
      savingDialog = new GenericDialog(tr("Save"), tr("Do you want to save the changes?"),
                                       GenericDialog::DialogMode::ACCEPT_CANCEL,
                                       WidgetsManager::getInstance()->getMainWindowRef());
      connect(savingDialog, &GenericDialog::notifyAcceptClicked, this, &IConfigurationCategory::onSaveAccepted);
      connect(savingDialog, &GenericDialog::notifyCancelClicked, this, &IConfigurationCategory::onSaveCancelled);
      savingDialog->show();
    }
  } else {
    emit reqBack();
  }
}

void IConfigurationCategory::onConfigurationChanged() { buttonSave->setEnabled(hasValidConfiguration()); }

void IConfigurationCategory::cleanDialog() {
  savingDialog->close();
  delete savingDialog;
  savingDialog = nullptr;
}

void IConfigurationCategory::onSaveCancelled() {
  cleanDialog();
  restore();
}

void IConfigurationCategory::onSaveAccepted() {
  cleanDialog();
  if (save()) {
    emit reqBack();
  }
}
