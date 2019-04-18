// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "codec.hpp"
#include "libvideostitch/parse.hpp"
#include <QGridLayout>
#include <QComboBox>
#include <QLabel>
#include <QApplication>

/**
 * @brief The ProResCodec class represents a widget holding the properties of the Pro Res codec
 */
class ProResCodec : public Codec {
  Q_OBJECT

  enum class ProfileEnum { PROXY, LT, STANDARD, HQ };

 public:
  explicit ProResCodec(QWidget* const parent = nullptr)
      : Codec(parent),
        mainLayout(new QGridLayout(this)),
        profileComboBox(new QComboBox(this)),
        labelProfile(new QLabel(tr("Profile:"), this)) {
    addProfile(ProfileEnum::PROXY);
    addProfile(ProfileEnum::LT);
    addProfile(ProfileEnum::STANDARD);
    addProfile(ProfileEnum::HQ);
    connect(profileComboBox, SIGNAL(currentIndexChanged(int)), this, SIGNAL(valueChanged()));
  }

  virtual QString getKey() const override { return QStringLiteral("prores"); }

  virtual void setup() override {
    setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(CONTROLS_SPACING);
    mainLayout->setContentsMargins(0, 0, 0, 1);
    labelProfile->setFixedSize(LABEL_WIDTH, CONTROL_HEIGHT);
    profileComboBox->setFixedSize(CONTROL_WIDTH, CONTROL_HEIGHT);
    mainLayout->addWidget(labelProfile, 0, 0);
    mainLayout->addWidget(profileComboBox, 0, 1);
    setLayout(mainLayout);
  }

  virtual bool hasConfiguration() const override { return true; }

  virtual VideoStitch::Ptv::Value* getOutputConfig() const override {
    VideoStitch::Ptv::Value* outputConfig = VideoStitch::Ptv::Value::emptyObject();
    outputConfig->get("profile")->asString() = profileComboBox->currentData().toString().toStdString();
    return outputConfig;
  }

  virtual bool setFromOutputConfig(const VideoStitch::Ptv::Value* config) override {
    std::string profile;
    if (VideoStitch::Parse::populateString("Ptv", *config, "profile", profile, false) !=
        VideoStitch::Parse::PopulateResult_Ok) {
      return false;
    }

    const int index = profileComboBox->findData(QString::fromStdString(profile));
    if (index < 0) {
      profileComboBox->setCurrentIndex(0);
    } else {
      profileComboBox->setCurrentIndex(index);
    }
    return true;
  }

 private:
  QGridLayout* mainLayout;
  QComboBox* profileComboBox;
  QLabel* labelProfile;

  QString getDisplayNameFromEnum(const ProfileEnum& value) const {
    switch (value) {
      case ProfileEnum::PROXY:
        return tr("Proxy");
      case ProfileEnum::LT:
        return tr("LT");
      case ProfileEnum::STANDARD:
        return tr("Standard");
      case ProfileEnum::HQ:
        return tr("High Quality");
      default:
        return QString();
    }
  }

  QString getStringFromEnum(const ProfileEnum& value) const {
    switch (value) {
      case ProfileEnum::PROXY:
        return QStringLiteral("proxy");
      case ProfileEnum::LT:
        return QStringLiteral("lt");
      case ProfileEnum::STANDARD:
        return QStringLiteral("standard");
      case ProfileEnum::HQ:
        return QStringLiteral("hq");
      default:
        return QString();
    }
  }

  void addProfile(const ProfileEnum& profile) {
    const QString name = getDisplayNameFromEnum(profile);
    const QString data = getStringFromEnum(profile);
    profileComboBox->addItem(name, data);
  }
};
