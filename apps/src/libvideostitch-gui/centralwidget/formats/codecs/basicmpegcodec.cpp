// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "basicmpegcodec.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch-gui/utils/bitratemodeenum.hpp"

static const unsigned int BITRATE_UNIT(1000);
static const unsigned int BASE(100);
static const unsigned int STEP_PAGE(10);  // Every 1000kb using pageUp / pageDown
static const unsigned int STEP_UP(1);     // Every 100kb using arrow up / arrow down

BasicMpegCodec::BasicMpegCodec(QWidget* parent)
    : Codec(parent),
      mainLayout(new QGridLayout(this)),
      bitrateModeComboBox(new QComboBox(this)),
      bitrateSlider(new QSlider(Qt::Horizontal, this)),
      bitrateLabel(new QLabel(this)),
      bitrateModeLabel(new QLabel(this)),
      bitrateLineEdit(new QLineEdit(this)) {}

void BasicMpegCodec::setup() {
  mainLayout->setContentsMargins(0, 0, 0, 0);
  mainLayout->setSpacing(CONTROLS_SPACING);
  setContentsMargins(0, 0, 0, 0);
  bitrateModeComboBox->addItems(BitRateModeEnum::getDescriptorsList());
  bitrateSlider->setRange(LIBAV_WRITER_MIN_MP4_BITRATE / BITRATE_UNIT / BASE,
                          LIBAV_WRITER_MAX_MP4_BITRATE / BITRATE_UNIT / BASE);
  bitrateSlider->setSliderPosition(LIBAV_WRITER_DEFAULT_BITRATE / BITRATE_UNIT / BASE);
  bitrateSlider->setPageStep(STEP_PAGE);
  bitrateSlider->setSingleStep(STEP_UP);
  bitrateLineEdit->setValidator(new QIntValidator(LIBAV_WRITER_MIN_MP4_BITRATE / BITRATE_UNIT,
                                                  LIBAV_WRITER_MAX_MP4_BITRATE / BITRATE_UNIT, this));
  bitrateLineEdit->setPlaceholderText(QString::number(LIBAV_WRITER_DEFAULT_BITRATE / BITRATE_UNIT));
  bitrateLineEdit->setObjectName("bitrateLineEdit");
  bitrateSlider->setObjectName("bitrateSlider");
  bitrateLineEdit->setFixedSize(CONTROL_WIDTH, CONTROL_HEIGHT);
  bitrateLabel->setText(tr("Bitrate (kbps):"));
  bitrateModeLabel->setText(tr("Bitrate mode:"));
  bitrateLabel->setFixedSize(LABEL_WIDTH, CONTROL_HEIGHT);
  bitrateModeLabel->setFixedSize(LABEL_WIDTH, CONTROL_HEIGHT);
  bitrateModeComboBox->setFixedWidth(CONTROL_WIDTH);
  bitrateModeComboBox->setFocusPolicy(Qt::StrongFocus);
  bitrateSlider->setFixedSize(CONTROL_WIDTH, CONTROL_HEIGHT);
  mainLayout->addWidget(bitrateModeLabel, 0, 0);
  mainLayout->addWidget(bitrateModeComboBox, 0, 1);
  mainLayout->addWidget(bitrateLabel, 1, 0);
  mainLayout->addWidget(bitrateSlider, 1, 1);
  mainLayout->addWidget(bitrateLineEdit, 1, 2);
  setLayout(mainLayout);
  connect(bitrateModeComboBox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this,
          &BasicMpegCodec::valueChanged);
  connect(bitrateSlider, static_cast<void (QSlider::*)(int)>(&QSlider::valueChanged), this,
          &BasicMpegCodec::updateTextEdit);
  connect(bitrateSlider, &QSlider::sliderReleased, this, [=]() { emit valueChanged(); });
  connect(bitrateLineEdit, &QLineEdit::textChanged, this, &BasicMpegCodec::updatePosition);
}

bool BasicMpegCodec::hasConfiguration() const { return true; }

VideoStitch::Ptv::Value* BasicMpegCodec::getOutputConfig() const {
  VideoStitch::Ptv::Value* outputConfig = VideoStitch::Ptv::Value::emptyObject();
  outputConfig->get("bitrate")->asInt() = bitrateSlider->value() * BITRATE_UNIT * BASE;
  outputConfig->get("bitrate_mode")->asString() = bitrateModeComboBox->currentText().toStdString();
  return outputConfig;
}

bool BasicMpegCodec::setFromOutputConfig(const VideoStitch::Ptv::Value* config) {
  std::string bitratemode;
  if (VideoStitch::Parse::populateString("Ptv", *config, "bitrate_mode", bitratemode, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    bitratemode = LIBAV_WRITER_DEFAULT_BITRATE_MODE;
  }

  int bitrate = 0;
  if (VideoStitch::Parse::populateInt("Ptv", *config, "bitrate", bitrate, false) !=
      VideoStitch::Parse::PopulateResult_Ok) {
    bitrate = LIBAV_WRITER_DEFAULT_BITRATE;
  }

  if (!BitRateModeEnum::getDescriptorsList().contains(QString::fromStdString(bitratemode))) {
    bitratemode = LIBAV_WRITER_DEFAULT_BITRATE_MODE;
  }

  bitrateSlider->setValue(bitrate / BITRATE_UNIT / BASE);
  bitrateModeComboBox->setCurrentText(QString::fromStdString(bitratemode));
  return true;
}

void BasicMpegCodec::updateTextEdit(int value) {
  BasicMpegCodec::bitrateLineEdit->setText(QString::number(value * BASE));
}
void BasicMpegCodec::updatePosition(const QString str) { BasicMpegCodec::bitrateSlider->setValue(str.toInt() / BASE); }
