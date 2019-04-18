// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/object.hpp"
#include "libvideostitch-gui/utils/audioprocessors.hpp"

#include <QObject>

#include <memory>

struct AudioConfiguration;
class AudioProcessorConfigurationWidget;
class QPixmap;

namespace VideoStitch {
namespace Ptv {
class Value;
}
}  // namespace VideoStitch

class LiveAudioProcessFactory : public QObject, public VideoStitch::Ptv::Object {
  Q_OBJECT
 public:
  // Don't take ownership of 'config'
  static LiveAudioProcessFactory* createProcessor(const VideoStitch::Ptv::Value* config,
                                                  const VideoStitch::AudioProcessors::ProcessorEnum& type);
  static QVector<QPair<QString, QString>> getAvailableProcessors();
  static std::unique_ptr<VideoStitch::Ptv::Value> createDefaultConfig(
      const QString& inputName, const VideoStitch::AudioProcessors::ProcessorEnum& type);

  VideoStitch::AudioProcessors::ProcessorEnum getType() const;
  const std::string& getInputName() const;
  virtual QPixmap getIcon() const = 0;
  virtual QString getDescription() const = 0;

  virtual AudioProcessorConfigurationWidget* createConfigurationWidget(QWidget* parent) = 0;
  virtual void serializeParameters(VideoStitch::Ptv::Value* value) = 0;

 protected:
  VideoStitch::AudioProcessors::ProcessorEnum type = VideoStitch::AudioProcessors::ProcessorEnum::UNKNOWN;
  std::string inputName;
};
