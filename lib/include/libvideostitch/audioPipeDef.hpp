// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Audio input def parser

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "object.hpp"
#include "status.hpp"
#include "panoDef.hpp"
#include "ambDecoderDef.hpp"

namespace VideoStitch {

namespace Ptv {
class Value;
}
namespace Core {

static const std::string kDelayProcessorName{"delay"};
static const std::string kGainProcessorName{"gain"};
static const std::string kAmbRotateProcessorName{"ambRotator"};
static const std::string kOrah2bProcessorName{"orah2b"};

static const std::vector<std::string> kEnabledAudioProcessors{kDelayProcessorName, kGainProcessorName,
                                                              kAmbRotateProcessorName, kOrah2bProcessorName};

typedef struct InputParam {
  InputParam(const std::string& name, audioreaderid_t id, Audio::ChannelLayout l) : name(name), id(id), layout(l) {}
  std::string name;
  audioreaderid_t id;
  Audio::ChannelLayout layout;
} InputParam;

class VS_EXPORT AudioProcessorDef : public Ptv::Object {
 public:
  ~AudioProcessorDef();
  Status addDelay(const std::string& inputName, double delay);
  Status addGain(const std::string& inputName, double gaindB, bool reversePolarity, bool mute);
  static Potential<AudioProcessorDef> create(const Ptv::Value& value);
  static Potential<AudioProcessorDef> createDelayProcessor(const std::string& inputName, double delay);
  static Potential<AudioProcessorDef> createGainProcessor(const std::string& inputName, double gaindB,
                                                          bool reversePolarity, bool mute);
  AudioProcessorDef* clone() const;
  std::string& getName() const;
  Ptv::Value* getParameters() const;
  Ptv::Value* getParameters(const std::string& inputName) const;
  void setName(const std::string& name);
  void setParameters(Ptv::Value* params);
  Ptv::Value* serialize() const;

 private:
  AudioProcessorDef();
  class Pimpl;
  Pimpl* const pimpl;
};

class VS_EXPORT AudioSourceDefinition : public Ptv::Object {
 public:
  static Potential<AudioSourceDefinition> create(const Ptv::Value& value);
  static AudioSourceDefinition* create(audioreaderid_t readerId, size_t channelId);
  AudioSourceDefinition* clone() const;
  Ptv::Value* serialize() const;
  readerid_t getReaderId() const;
  void setReaderId(audioreaderid_t readerId);
  size_t getChannel() const;
  void setChannel(size_t channel);
  ~AudioSourceDefinition();

 private:
  AudioSourceDefinition();
  class Pimpl;
  std::unique_ptr<Pimpl> pimpl;
};

class VS_EXPORT AudioInputDefinition : public Ptv::Object {
 public:
  static Potential<AudioInputDefinition> create(const Ptv::Value& value);
  static AudioInputDefinition* create(const InputParam& param);
  static AudioInputDefinition* createDefault();
  AudioInputDefinition* clone() const;
  bool getIsMaster() const;
  std::string& getLayout() const;
  std::string& getName() const;
  AudioSourceDefinition* getSource(size_t i) const;
  size_t numSources() const;
  Ptv::Value* serialize() const;
  void setIsMaster(bool b);
  Status setLayout(const std::string& layout);
  void setName(const std::string& name);
  int getNumbChannels() const;
  ~AudioInputDefinition();

 private:
  AudioInputDefinition();
  class Pimpl;
  std::unique_ptr<Pimpl> pimpl;
};

class VS_EXPORT AudioMixDefinition : public Ptv::Object {
 public:
  Status addInput(const std::string& name);
  static Potential<AudioMixDefinition> create(const Ptv::Value& value);
  static AudioMixDefinition* create(const std::string& name, std::vector<std::string> inputs);
  AudioMixDefinition* clone() const;
  std::string& getInput(size_t i) const;
  std::vector<std::string> getInputs() const;
  std::string& getName() const;
  size_t numInputs() const;
  Ptv::Value* serialize() const;
  Status setName(const std::string& name);

  ~AudioMixDefinition();

 private:
  AudioMixDefinition();
  class Pimpl;
  std::unique_ptr<Pimpl> pimpl;
};

class VS_EXPORT AudioPipeDefinition : public Ptv::Object {
 public:
  Status addDelayProcessor(const std::string& inputName, double delay);
  Status addGainProcessor(const std::string& inputName, double gaindB, bool reversePolarity, bool mute);
  void addInput(AudioInputDefinition*);
  /**
   * Build from a Ptv::Value.
   * @param value Input value.
   * @return The parsed AudioPipeDefinition.
   */
  static AudioPipeDefinition* create(const Ptv::Value& value);
  static AudioPipeDefinition* create(const std::vector<InputParam>& inputParams);
  static AudioPipeDefinition* createAudioPipeFromPanoInputs(const PanoDefinition* pano);
  static AudioPipeDefinition* createDefault();
  AudioPipeDefinition* clone() const;
  Audio::AmbisonicDecoderDef* getAmbDecodingCoef() const;
  int getBlockSize() const;
  double getMaxDelayValue() const;
  PotentialValue<double> getDelay(const std::string& inputName);
  PotentialValue<double> getGain(const std::string& inputName);
  bool getHasVuMeter() const;
  PotentialValue<bool> getMute(const std::string& inputName);
  PotentialValue<bool> getReversePolarity(const std::string& inputName);
  std::string& getDebugFolder() const;
  const std::string& getSelectedAudio() const;
  PotentialValue<AudioInputDefinition*> getSelectedInput() const;
  AudioInputDefinition* getInput(size_t i) const;
  PotentialValue<AudioInputDefinition*> getInput(const std::string& inputName) const;
  std::vector<std::reference_wrapper<const std::string> > getInputNames() const;
  AudioMixDefinition* getMix(size_t i) const;
  AudioProcessorDef* getProcessor(size_t i) const;
  PotentialValue<AudioProcessorDef*> getProcessor(const std::string& procName) const;
  int getSamplingRate() const;
  Status isValidInput(const std::string& inputName) const;
  size_t numProcessors() const;
  audioreaderid_t numAudioInputs() const;
  size_t numAudioMixes() const;
  bool readersAreConsistent(PanoDefinition* pano) const;
  Status removeInput(audioreaderid_t i);
  Status removeProcessor(const std::string& procName);
  Status replaceInput(audioreaderid_t i, const std::string& name);
  Status setAmbDecodingCoef(Ptv::Value* ptv);
  void setDebugFolder(const std::string& s) const;
  void setBlockSize(const int bs);
  Status setDelay(const std::string& inputName, double delay);
  Status setGain(const std::string& inputName, double gain);
  void setHasVuMeter(bool has);
  Status setMute(const std::string& inputName, bool mute);
  Status setReversePolarity(const std::string& inputName, bool reversePolarity);
  void setSamplingRate(const int sr);
  Status setSelectedAudio(const std::string&);
  ~AudioPipeDefinition();

  Ptv::Value* serialize() const;

 private:
  PotentialValue<bool> getControlAsBool(const std::string& procName, const std::string& inputName,
                                        const std::string& controlName);
  PotentialValue<double> getControlAsDouble(const std::string& procName, const std::string& inputName,
                                            const std::string& controlName);
  Status setControlBool(const std::string& procName, const std::string& inputName, const std::string& controlName,
                        bool value);
  Status setControlDouble(const std::string& procName, const std::string& inputName, const std::string& controlName,
                          double value);
  AudioPipeDefinition();
  class Pimpl;
  std::unique_ptr<Pimpl> pimpl;
};

}  // namespace Core
}  // namespace VideoStitch
