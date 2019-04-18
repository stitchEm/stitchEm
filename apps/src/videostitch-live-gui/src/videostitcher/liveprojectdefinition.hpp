// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "audioconfiguration.hpp"
#include "livemutableprojectdefinition.hpp"
#include "libvideostitch-gui/videostitcher/projectdefinition.hpp"
#include "liveinputfactory.hpp"
#include "liveaudioprocessfactory.hpp"

#define VIDEO_WRITER_DEFAULT_FRAMERATE 30.0

/**
 * @brief A project definition extended for Vahana
 */

class LiveOutputFactory;
class LiveOutputList;

class LiveProjectDefinition : public ProjectDefinition {
  Q_OBJECT

 public:
  LiveProjectDefinition();

  virtual ~LiveProjectDefinition();

  virtual LiveMutableProjectDefinition* getDelegate() const override;

  virtual void createDelegate(const VideoStitch::Ptv::Value& value) override;

  virtual void destroyDelegate() override;

  LiveOutputList* getOutputConfigs() const;

  /**
   * @brief Gets the list of output names in the project.
   * @return List of names
   */
  QStringList getOutputNames() const;

  /**
   * @brief Sets the panorama width and height. Updates project data if needed (eg RTMP and HDD outputs).
   * @param width New panorama width.
   * @param height New panorama height.
   */
  void updateSize(int width, int height) override;

  /**
   * @brief Replace existing inputs and configure audio
   * @return false if no preset is found. true for success
   */
  bool updateInputs(LiveInputList inputs, const AudioConfiguration& audioConfiguration);

  /**
   * @brief Replace existing audio input
   */
  void updateAudioInput(const AudioConfiguration& audioConfiguration);

  /**
   * @brief Gets a liveoutput from its id
   * @param id A unique output id
   * @return A valid live output if id is present, otherwise, nullptr
   */
  LiveOutputFactory* getOutputById(const QString& id) const;

  /**
   * @brief Check if there is one or more active outputs
   * @return true if at least on output is active
   */
  bool areActiveOutputs() const;

  /**
   * @brief Change the value of the identifier in the output list
   * @param id An unique identifier
   */
  void updateOutputId(const QString& id);

  /**
   * @brief Check that the output can be added to the output list
   */
  bool canAddOutput(const LiveOutputFactory* output) const;

  /**
   * @brief Add the output to the output list (see also canAddOutput)
   * @return true if the operation succeed, otherwise, false
   */
  bool addOutput(LiveOutputFactory* output);

  /**
   * @brief Delete the output from the project
   * @param id A unique
   */
  void deleteOutput(const QString& id);

  /**
   * @brief Remove all the outputs
   */
  void clearOutputs();

  /**
   * @brief Build a LiveInput from the inputs parameters and return it. The caller has ownership.
   */
  LiveInputFactory* retrieveConfigurationInput() const;

  /**
   * @brief Build LiveInput objects from the video inputs parameters and return them.
   */
  LiveInputList retrieveVideoInputs() const;

  /**
   * @return the audio configuration
   */
  AudioConfiguration getAudioConfiguration() const;
  /**
   * @return the index of the input (in the list of inputs) which is used for the audio. -1 if audio is disabled or if
   * we had an audio-only input
   */
  int getInputIndexForAudio() const;

  /**
   * @brief Return the first divisors for the current panorama size
   * @return A list of factors
   */
  QVector<unsigned int> getDownsampledFactors() const;

  QString getOutputDisplayableString(unsigned int factor) const;

  /**
   * @brief Checks if the device is listed in the iput or outputs.
   * @param type The device type
   * @param deviceName The device name
   * @return True if the device appears in the inputs or the outputs
   */
  bool isDeviceInUse(const QString& type, const QString& name) const;

  bool updateAudioProcessor(LiveAudioProcessFactory* liveProcessor);

  bool removeAudioProcessor(const QString& name);

  bool setAudioProcessorConfiguration(LiveAudioProcessFactory* liveProcessor);

  QList<LiveAudioProcessFactory*> getAudioProcessors() const;

 private:
  std::vector<VideoStitch::Ptv::Value*> serializeAndInitializeInputs(LiveInputList list) const;

  QScopedPointer<LiveMutableProjectDefinition> delegate;
};
