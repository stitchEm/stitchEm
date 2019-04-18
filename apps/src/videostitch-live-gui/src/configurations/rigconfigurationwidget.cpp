// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "rigconfigurationwidget.hpp"
#include "inputeyeselector.hpp"
#include <QListWidgetItem>

RigConfigurationWidget::RigConfigurationWidget(QWidget* const parent) : QFrame(parent) {
  setupUi(this);
  buttonAccept->setProperty("vs-button-medium", true);
  connect(buttonAccept, &QPushButton::clicked, this, &RigConfigurationWidget::onButtonAcceptClicked);
  connect(buttonCircular, &QPushButton::clicked, this, &RigConfigurationWidget::onButtonCircularChecked);
  connect(buttonPolygonal, &QPushButton::clicked, this, &RigConfigurationWidget::onButtonPolygonalChecked);
  connect(comboOrientation, &QComboBox::currentTextChanged, this, &RigConfigurationWidget::onOrientationChanged);

  comboOrientation->addItem(QString::fromStdString(VideoStitch::Core::StereoRigDefinition::getOrientationName(
      VideoStitch::Core::StereoRigDefinition::Orientation::Portrait)));
  comboOrientation->addItem(QString::fromStdString(VideoStitch::Core::StereoRigDefinition::getOrientationName(
      VideoStitch::Core::StereoRigDefinition::Orientation::Landscape)));
  comboOrientation->addItem(QString::fromStdString(VideoStitch::Core::StereoRigDefinition::getOrientationName(
      VideoStitch::Core::StereoRigDefinition::Orientation::Portrait_flipped)));
  comboOrientation->addItem(QString::fromStdString(VideoStitch::Core::StereoRigDefinition::getOrientationName(
      VideoStitch::Core::StereoRigDefinition::Orientation::Landscape_flipped)));
}

RigConfigurationWidget::~RigConfigurationWidget() {}

void RigConfigurationWidget::loadConfiguration(const QStringList inputs,
                                               const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                                               const VideoStitch::Core::StereoRigDefinition::Geometry geometry,
                                               const double diameter, const double ipd, const QVector<int> leftInputs,
                                               const QVector<int> rightInputs) {
  inputNames = inputs;
  buttonCircular->setChecked(geometry == VideoStitch::Core::StereoRigDefinition::Geometry::Circular);
  buttonPolygonal->setChecked(geometry == VideoStitch::Core::StereoRigDefinition::Geometry::Polygonal);
  spinDiameter->setValue(diameter);
  spinIPD->setValue(ipd);
  cameraListWidget->clear();
  comboOrientation->setCurrentText(
      QString::fromStdString(VideoStitch::Core::StereoRigDefinition::getOrientationName(orientation)));
  addInputsToList(leftInputs, rightInputs);
}

void RigConfigurationWidget::onButtonAcceptClicked() {
  const VideoStitch::Core::StereoRigDefinition::Geometry geometry =
      buttonCircular->isChecked() ? VideoStitch::Core::StereoRigDefinition::Geometry::Circular
                                  : VideoStitch::Core::StereoRigDefinition::Geometry::Polygonal;

  VideoStitch::Core::StereoRigDefinition::Orientation orientation;
  VideoStitch::Core::StereoRigDefinition::getOrientationFromName(comboOrientation->currentText().toStdString(),
                                                                 orientation);

  emit notifyRigConfigured(orientation, geometry, spinDiameter->value(), spinIPD->value(), getLeftInputs(),
                           getRightInputs());
  close();
}

void RigConfigurationWidget::onButtonCircularChecked() { buttonPolygonal->setChecked(false); }

void RigConfigurationWidget::onButtonPolygonalChecked() { buttonCircular->setChecked(false); }

void RigConfigurationWidget::onOrientationChanged(const QString& name) {
  QString orientationImage = ":/live/icons/assets/icon/live/%0.png";
  VideoStitch::Core::StereoRigDefinition::Orientation orientation;
  VideoStitch::Core::StereoRigDefinition::getOrientationFromName(name.toStdString(), orientation);

  switch (orientation) {
    case VideoStitch::Core::StereoRigDefinition::Orientation::Portrait:
      orientationImage = orientationImage.arg("portrait");
      break;
    case VideoStitch::Core::StereoRigDefinition::Orientation::Landscape:
      orientationImage = orientationImage.arg("landscape");
      break;
    case VideoStitch::Core::StereoRigDefinition::Orientation::Portrait_flipped:
      orientationImage = orientationImage.arg("portrait-flipped");
      break;
    case VideoStitch::Core::StereoRigDefinition::Orientation::Landscape_flipped:
      orientationImage = orientationImage.arg("landscape-flipped");
      break;
    default:
      break;
  }
  labelOrientationImage->setPixmap(QPixmap(orientationImage));
}

void RigConfigurationWidget::addInputsToList(const QVector<int> left, const QVector<int> right) {
  for (auto index = 0; index < inputNames.count(); ++index) {
    const bool isLeft = left.count(index) > 0;
    const bool isRight = right.count(index) > 0;
    InputEyeSelector* inputItem =
        new InputEyeSelector(QString::number(index) + " - " + inputNames.at(index), isLeft, isRight, this);
    QListWidgetItem* item = new QListWidgetItem;
    item->setSizeHint(inputItem->sizeHint());
    cameraListWidget->addItem(item);
    cameraListWidget->setItemWidget(item, inputItem);
  }
}

QVector<int> RigConfigurationWidget::getLeftInputs() const {
  QVector<int> leftVector;
  for (auto itemIndex = 0; itemIndex < cameraListWidget->count(); ++itemIndex) {
    InputEyeSelector* widget =
        qobject_cast<InputEyeSelector*>(cameraListWidget->itemWidget(cameraListWidget->item(itemIndex)));
    if (widget != nullptr && widget->isLeftEye()) {
      leftVector.push_back(itemIndex);
    }
  }
  return leftVector;
}

QVector<int> RigConfigurationWidget::getRightInputs() const {
  QVector<int> rightVector;
  for (auto itemIndex = 0; itemIndex < cameraListWidget->count(); ++itemIndex) {
    InputEyeSelector* widget =
        qobject_cast<InputEyeSelector*>(cameraListWidget->itemWidget(cameraListWidget->item(itemIndex)));
    if (widget != nullptr && widget->isRightEye()) {
      rightVector.push_back(itemIndex);
    }
  }
  return rightVector;
}
