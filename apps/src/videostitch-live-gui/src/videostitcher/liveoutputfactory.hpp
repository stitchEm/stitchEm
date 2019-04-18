// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch-gui/utils/audiohelpers.hpp"
#include "libvideostitch-gui/utils/outputformat.hpp"

#include "libvideostitch/allocator.hpp"
#include "libvideostitch/audio.hpp"
#include "libvideostitch/object.hpp"
#include "libvideostitch/frame.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/panoDef.hpp"

#include <QObject>
#include <memory>
#include <stdint.h>

namespace VideoStitch {
class Status;
namespace Output {
class Output;
class VideoWriter;
class StereoWriter;
}  // namespace Output
namespace Ptv {
class Value;
}
}  // namespace VideoStitch
class LiveProjectDefinition;
class LiveAudio;
class IConfigurationCategory;
class QLabel;
class OutputConfigurationWidget;
class LiveMutableProjectDefinition;

/**
 * @brief Class for wrapping the VAH data and writer activation for an output
 */
class LiveOutputFactory : public QObject, public VideoStitch::Ptv::Object {
  Q_OBJECT

 public:
  // Take the ownership of 'config'
  static LiveOutputFactory* createOutput(VideoStitch::Ptv::Value* config,
                                         const VideoStitch::Core::PanoDefinition* panoDefinition);
  static LiveOutputFactory* createOutput(VideoStitch::Ptv::Value* config,
                                         const VideoStitch::OutputFormat::OutputFormatEnum& type,
                                         const VideoStitch::Core::PanoDefinition* panoDefinition);

 public:
  enum class OutputState { DISABLED, INITIALIZATION, CONNECTING, ENABLED };
  enum class PanoSizeChange { NotSupported, SupportedWithUpdate, Supported };

  explicit LiveOutputFactory(VideoStitch::OutputFormat::OutputFormatEnum type);
  virtual ~LiveOutputFactory();

  /**
   * Returns an unique identifier of the output. Never display this in the GUI, it's an internal data
   */
  virtual const QString getIdentifier() const = 0;
  virtual const QString getOutputTypeDisplayName() const;
  virtual const QString getOutputDisplayName() const = 0;
  /**
   * Returns anonymous output name, to be tracked
   */
  virtual const QString getDataToTrack() const { return getOutputTypeDisplayName(); }
  /**
   * @brief returns true if the output needs to be closes from a high level, before the project is fully closed
   *        false by default
   */
  virtual bool earlyClosingRequired() const { return false; }

  /**
   * Returns true if the type is supported
   */
  virtual bool isSupported() const { return true; }
  /**
   * Returns true if configuration widget can be created
   */
  virtual bool isConfigurable() const { return false; }
  /*
   * @brief Returns a configuration widget for this live output. If there is nothing to configure, return nullptr
   */
  virtual OutputConfigurationWidget* createConfigurationWidget(QWidget* const parent);
  virtual QPixmap getIcon() const = 0;
  virtual QWidget* createStatusWidget(QWidget* const parent) = 0;
  virtual bool checkIfIsActivable(const VideoStitch::Core::PanoDefinition* panoDefinition, QString& message) const;

  LiveAudio* getAudioConfig() const { return audioConfig.data(); }

  /*
   * @brief This will return all the sampling depths, ordering them. First sampling depths are closer to the reference
   * sampling depth.
   */
  static QList<VideoStitch::Audio::SamplingDepth> getOrderedSamplingDepths();
  virtual QList<VideoStitch::Audio::SamplingDepth> getSupportedSamplingDepths(
      const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const;
  VideoStitch::Audio::SamplingDepth getPreferredSamplingDepth(
      const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodecType) const;
  virtual QList<VideoStitch::Audio::SamplingRate> getSupportedSamplingRates(
      const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodec) const;
  virtual QList<VideoStitch::Audio::ChannelLayout> getSupportedChannelLayouts(
      const VideoStitch::AudioHelpers::AudioCodecEnum& audioCodecType) const;
  virtual PanoSizeChange supportPanoSizeChange(int newWidth, int newHeight) const;
  virtual QString getPanoSizeChangeDescription(int newWidth, int newHeight) const;
  virtual void updateForPanoSizeChange(int newWidth, int newHeight);

  OutputState getOutputState() const { return state; }
  void setOutputState(const OutputState& newState) { state = newState; }

  VideoStitch::OutputFormat::OutputFormatEnum getType() const { return type; }

 signals:
  void outputDisplayNameChanged(QString outputDisplayName);

 protected:
  QLabel* createStatusIcon(QWidget* const parent) const;
  virtual void initializeAudioOutput(const VideoStitch::Ptv::Value* config) const;

  QScopedPointer<LiveAudio> audioConfig;
  bool hasLog;
  OutputState state;
  VideoStitch::OutputFormat::OutputFormatEnum type;

 private:
};

class LiveWriterFactory : public LiveOutputFactory {
 public:
  explicit LiveWriterFactory(VideoStitch::OutputFormat::OutputFormatEnum type);
  /**
   * Factory methods.
   */
  virtual VideoStitch::Potential<VideoStitch::Output::Output> createWriter(LiveProjectDefinition* project,
                                                                           VideoStitch::FrameRate framerate);
  virtual VideoStitch::Potential<VideoStitch::Output::StereoWriter> createStereoWriter(
      LiveProjectDefinition* project, VideoStitch::FrameRate framerate);
  virtual void destroyWriter() {}
};

class LiveRendererFactory : public LiveOutputFactory {
 public:
  explicit LiveRendererFactory(VideoStitch::OutputFormat::OutputFormatEnum type);
  /**
   * Factory methods.
   */
  virtual VideoStitch::PotentialValue<std::shared_ptr<VideoStitch::Core::PanoRenderer>> createRenderer() = 0;
  virtual void destroyRenderer(bool /*wait*/) {}
};
