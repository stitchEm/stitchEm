// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "capturecardinputconfiguration.hpp"
#include "ui_capturecardinputconfiguration.h"

#include "guiconstants.hpp"
#include "plugin/pluginscontroller.hpp"
#include "utils/displaymode.hpp"
#include "videostitcher/capturecardliveinput.hpp"
#include "videostitcher/liveprojectdefinition.hpp"

CaptureCardInputConfiguration::CaptureCardInputConfiguration(std::shared_ptr<const CaptureCardLiveInput> liveInput,
                                                             QWidget* parent)
    : InputConfigurationWidget(parent), ui(new Ui::CaptureCardInputConfiguration), templateInput(liveInput) {
  ui->setupUi(this);
  ui->label->setProperty("vs-title1", true);
  ui->deviceListWidget->setProperty("vs-section-container", true);
  ui->noDeviceLabel->setProperty("vs-message", true);
  ui->mainLayout->addLayout(buttonsLayout);
  ui->deviceListWidget->setIconSize(QSize(ICON_SIZE, ICON_SIZE));

  connect(ui->deviceListWidget, &QListWidget::itemClicked,
          [](QListWidgetItem* item) { item->setIcon(item->isSelected() ? QIcon(CHECKED_ICON) : QIcon()); });

  connect(ui->deviceListWidget, &QListWidget::itemSelectionChanged, this,
          &CaptureCardInputConfiguration::onConfigurationChanged);
}

CaptureCardInputConfiguration::~CaptureCardInputConfiguration() {}

void CaptureCardInputConfiguration::reactToChangedProject() {
  Q_ASSERT(pluginsController != nullptr);
  Q_ASSERT(templateInput != nullptr);
  disconnect(ui->displayModeBox, &QComboBox::currentTextChanged, this,
             &CaptureCardInputConfiguration::updateAvailableDevices);

  // Retrieve input device names
  const QString cardName = VideoStitch::InputFormat::getStringFromEnum(templateInput->getType());
  DeviceList availableDevices;
  pluginsController->listInputDevicesFromPlugin(cardName, availableDevices);

  // Update display mode box
  ui->displayModeBox->clear();
  std::vector<VideoStitch::Plugin::DisplayMode> supportedDisplayModes =
      pluginsController->listDisplayModes(cardName, availableDevices);
  for (VideoStitch::Plugin::DisplayMode displayMode : supportedDisplayModes) {
    ui->displayModeBox->addItem(displayModeToString(displayMode), QVariant::fromValue(displayMode));
  }

  ui->displayModeBox->setCurrentText(displayModeToString(templateInput->getDisplayMode()));

  // Update device list
  QStringList selectedDevices = projectDefinition->getInputNames();
  ui->stackedWidget->setCurrentWidget(availableDevices.empty() ? ui->pageNoDevice : ui->pageDevice);

  for (VideoStitch::Plugin::DiscoveryDevice device : availableDevices) {
    QString deviceName = QString::fromStdString(device.name);
    QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(device.displayName), ui->deviceListWidget);
    item->setSizeHint(QSize(ui->deviceListWidget->width(), ITEM_HEIGHT));
    item->setData(Qt::UserRole, deviceName);
    item->setSelected(selectedDevices.contains(deviceName));
    item->setIcon(item->isSelected() ? QIcon(CHECKED_ICON) : QIcon());
    ui->deviceListWidget->addItem(item);
  }
  connect(ui->displayModeBox, &QComboBox::currentTextChanged, this,
          &CaptureCardInputConfiguration::updateAvailableDevices);
  updateAvailableDevices();
}

bool CaptureCardInputConfiguration::hasValidConfiguration() const {
  return !ui->deviceListWidget->selectedItems().isEmpty();
}

void CaptureCardInputConfiguration::updateAvailableDevices() {
  const QString cardName = VideoStitch::InputFormat::getStringFromEnum(templateInput->getType());
  auto selectedDisplayMode = ui->displayModeBox->currentData().value<VideoStitch::Plugin::DisplayMode>();
  for (int index = 0; index < ui->deviceListWidget->count(); ++index) {
    auto item = ui->deviceListWidget->item(index);
    QString deviceName = item->data(Qt::UserRole).toString();
    auto supportedDisplayModes = pluginsController->listDisplayModes(cardName, QStringList() << deviceName);
    if (std::find(supportedDisplayModes.cbegin(), supportedDisplayModes.cend(), selectedDisplayMode) ==
        supportedDisplayModes.cend()) {
      item->setFlags(Qt::NoItemFlags);
      item->setToolTip(tr("This device is not compatible with this display mode"));
      item->setSelected(false);
      item->setIcon(QIcon());
    } else {
      item->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      item->setToolTip(QString());
    }
  }

  // The selected items can change
  onConfigurationChanged();
}

void CaptureCardInputConfiguration::saveData() {
  editedInputs.clear();
  VideoStitch::InputFormat::InputFormatEnum type = templateInput->getType();
  auto displayMode = ui->displayModeBox->currentData().value<VideoStitch::Plugin::DisplayMode>();
  for (QListWidgetItem* item : ui->deviceListWidget->selectedItems()) {
    QString deviceName = item->data(Qt::UserRole).toString();
    CaptureCardLiveInput* input = static_cast<CaptureCardLiveInput*>(LiveInputFactory::makeLiveInput(type, deviceName));
    input->setDisplayMode(displayMode);
    editedInputs.append(std::shared_ptr<LiveInputFactory>(input));
  }
}
