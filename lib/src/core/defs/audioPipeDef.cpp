// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Audio input def parser

#include <sstream>

#include "libvideostitch/logging.hpp"
#include "audioPipeDefPimpl.hpp"

#include "parse/json.hpp"
#include "common/container.hpp"
#include "audio/sampleDelay.hpp"
#include "audio/audioPipeFactory.hpp"

#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Core {

namespace {

///
/// \brief parseAudioInputs
/// \param value
/// \param audioInputDefs
/// \return
///
Status parseAudioInputs(const Ptv::Value& value,
                        std::vector<std::unique_ptr<Core::AudioInputDefinition>>& audioInputDefs) {
  const Ptv::Value* var = value.has("audio_inputs");
  if (!Parse::checkVar("AudioPipeDefinition", "audioInputs", var, true)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "No audio input definition found"};
  }
  if (!Parse::checkType("inputs", *var, Ptv::Value::LIST)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "Wrong audio input definition type"};
  }
  const std::vector<Ptv::Value*>& inputs = var->asList();

  for (const Ptv::Value* inputPtv : inputs) {
    Potential<AudioInputDefinition> input = AudioInputDefinition::create(*inputPtv);
    FAIL_RETURN(input.status());
    audioInputDefs.emplace_back(input.release());
  }
  return Status::OK();
}

///
/// \brief parseAudioProcessors
/// \param value
/// \param audioProcDefs
/// \return
///
Status parseAudioProcessors(const Ptv::Value& value,
                            std::vector<std::unique_ptr<Core::AudioProcessorDef>>& audioProcDefs) {
  const Ptv::Value* var = value.has("audio_processors");
  if (!Parse::checkVar("AudioPipeDefinition", "audioProcessors", var, false)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "No audio processor definition found"};
  }
  if (!Parse::checkType("audioProcessors", *var, Ptv::Value::LIST)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "Wrong audio processor definition type"};
  }
  const std::vector<Ptv::Value*>& audioProcs = var->asList();
  for (const Ptv::Value* audioProc : audioProcs) {
    Potential<AudioProcessorDef> procDef = AudioProcessorDef::create(*audioProc);
    FAIL_RETURN(procDef.status());
    audioProcDefs.emplace_back(procDef.release());
  }
  return Status::OK();
}

///
/// \brief parseAudioMixes
/// \param value
/// \param audioMixDefinition
/// \return
///
Status parseAudioMixes(const Ptv::Value& value, std::vector<std::unique_ptr<Core::AudioMixDefinition>>& audioMixDefs) {
  const Ptv::Value* var = value.has("audio_mixes");
  if (!Parse::checkVar("AudioPipeDefinition", "audioMixes", var, false)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "No audio mix definition found"};
  }
  if (!Parse::checkType("audioMixes", *var, Ptv::Value::LIST)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "Wrong audio mix definition type"};
  }
  const std::vector<Ptv::Value*>& audioMixes = var->asList();
  for (const Ptv::Value* audioMix : audioMixes) {
    Potential<AudioMixDefinition> audioMixDef = AudioMixDefinition::create(*audioMix);
    FAIL_RETURN(audioMixDef.status());
    audioMixDefs.emplace_back(audioMixDef.release());
  }
  return Status::OK();
}

///
/// \brief parseAudioSources
/// \param value
/// \param audioSourcesDefs
/// \return
///
Status parseAudioSources(const Ptv::Value& value,
                         std::vector<std::unique_ptr<Core::AudioSourceDefinition>>& audioSources) {
  const Ptv::Value* var = value.has("sources");
  if (!Parse::checkVar("AudioInputDefinition", "sources", var, true)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "No audio source definition found"};
  }
  if (!Parse::checkType("sources", *var, Ptv::Value::LIST)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "Wrong audio source definition type"};
  }
  const std::vector<Ptv::Value*>& sources = var->asList();

  for (const Ptv::Value* sourcePtv : sources) {
    Potential<AudioSourceDefinition> source = AudioSourceDefinition::create(*sourcePtv);
    FAIL_RETURN(source.status());
    audioSources.emplace_back(source.release());
  }
  return Status::OK();
}

}  // end namespace

////////////////////////////////////////////////////////////////////////////////
/////////// AudioSource ////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
AudioSourceDefinition::Pimpl::Pimpl() : readerId(0), channelId(0) {}

AudioSourceDefinition::Pimpl::~Pimpl() {}

AudioSourceDefinition::AudioSourceDefinition() : pimpl(new Pimpl()) {}

AudioSourceDefinition::~AudioSourceDefinition() {}

Potential<AudioSourceDefinition> AudioSourceDefinition::create(const Ptv::Value& value) {
  // Make sure value is an object.
  if (!Parse::checkType("AudioSourceDefinition", value, Ptv::Value::OBJECT)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "Wrong type of audio source definition"};
  }
  std::unique_ptr<AudioSourceDefinition> res(new AudioSourceDefinition());
#define PROPAGATE_NOK(call)               \
  if (call != Parse::PopulateResult_Ok) { \
    return nullptr;                       \
  }
  PROPAGATE_NOK(Parse::populateInt("AudioSourceDefinition", value, "reader_id", res->pimpl->readerId, true));
#undef PROPAGATE_NOK
  if (Parse::populateInt("AudioSourceDefinition", value, "channel", res->pimpl->channelId, true) ==
      Parse::PopulateResult_DoesNotExist) {
    std::stringstream ss;
    ss << "No channel id found in this source" << res->pimpl->readerId;
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, ss.str()};
  }

  return res.release();
}

AudioSourceDefinition* AudioSourceDefinition::create(audioreaderid_t readerId, size_t channelId) {
  std::unique_ptr<AudioSourceDefinition> res(new AudioSourceDefinition());
  res->setReaderId(readerId);
  res->setChannel(channelId);
  return res.release();
}

AudioSourceDefinition* AudioSourceDefinition::clone() const {
  AudioSourceDefinition* result = new AudioSourceDefinition();
  result->setReaderId(getReaderId());
  result->setChannel(getChannel());
  return result;
}

Ptv::Value* AudioSourceDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("reader_id", new Parse::JsonValue((int)getReaderId()));
  res->push("channel", new Parse::JsonValue((int)getChannel()));
  return res;
}

readerid_t AudioSourceDefinition::getReaderId() const { return pimpl->readerId; }

void AudioSourceDefinition::setReaderId(audioreaderid_t readerId) { pimpl->readerId = readerId; }

size_t AudioSourceDefinition::getChannel() const { return pimpl->channelId; }

void AudioSourceDefinition::setChannel(size_t channel) { pimpl->channelId = channel; }

////////////////////////////////////////////////////////////////////////////////
/////////// AudioInputDefinition  //////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// \brief AudioInputDefinition::AudioInputDefinition
///
AudioInputDefinition::AudioInputDefinition() : pimpl(new Pimpl()) {}

Potential<AudioInputDefinition> AudioInputDefinition::create(const Ptv::Value& value) {
  // Make sure value is an object.
  if (!Parse::checkType("AudioInputDefinition", value, Ptv::Value::OBJECT)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "Wrong type of audio input definition"};
  }
  std::unique_ptr<AudioInputDefinition> res(new AudioInputDefinition());
#define PROPAGATE_NOK(call)               \
  if (call != Parse::PopulateResult_Ok) { \
    return nullptr;                       \
  }
  PROPAGATE_NOK(Parse::populateString("AudioInputDefinition", value, "name", res->pimpl->name, true));
#undef PROPAGATE_NOK
  Parse::populateBool("AudioInputDefinition", value, "master", res->pimpl->isMaster, false);
  std::string layout;
  Parse::populateString("AudioInputDefinition", value, "layout", layout, false);
  if (Parse::populateString("AudioInputDefinition", value, "layout", layout, false) !=
      Parse::PopulateResult_DoesNotExist) {
    FAIL_RETURN(res->setLayout(layout));
  }

  // Populate sources
  {
    std::vector<std::unique_ptr<Core::AudioSourceDefinition>> sources;
    FAIL_RETURN(parseAudioSources(value, sources));
    for (size_t i = 0; i < sources.size(); ++i) {
      res->pimpl->sources.emplace_back(std::move(sources[i]));
    }
  }
  return res.release();
}

AudioInputDefinition* AudioInputDefinition::create(const InputParam& param) {
  std::unique_ptr<AudioInputDefinition> res(new AudioInputDefinition());
  res->setName(param.name);
  res->setLayout(getStringFromChannelLayout(param.layout));
  for (size_t i = 0; i < static_cast<size_t>(getNbChannelsFromChannelLayout(param.layout)); i++) {
    res->pimpl->sources.emplace_back(AudioSourceDefinition::create(param.id, i));
  }
  return res.release();
}

AudioInputDefinition* AudioInputDefinition::createDefault() {
  std::unique_ptr<AudioInputDefinition> res(new AudioInputDefinition());
  // By default create a fake input which will be feeded by a default source
  // this default source will read the first channel of the first reader
  res->setName("defaultInput");
  res->pimpl->sources.push_back(std::unique_ptr<AudioSourceDefinition>(AudioSourceDefinition::create(0, 0)));
  res->pimpl->sources.push_back(std::unique_ptr<AudioSourceDefinition>(AudioSourceDefinition::create(0, 1)));
  return res.release();
}

AudioInputDefinition* AudioInputDefinition::clone() const {
  AudioInputDefinition* result = new AudioInputDefinition();
  result->setName(getName());
  result->setIsMaster(getIsMaster());
  if (!getLayout().empty()) {
    result->setLayout(getLayout());
  }
  for (size_t i = 0; i < numSources(); ++i) {
    result->pimpl->sources.emplace_back(getSource(i)->clone());
  }
  return result;
}

///
/// \brief AudioInputDefinition::~AudioInputDefinition
///
AudioInputDefinition::~AudioInputDefinition() {}

std::string& AudioInputDefinition::getName() const { return pimpl->name; }

void AudioInputDefinition::setName(const std::string& name) { pimpl->name = name; }

// TODO remove this function if unused. It's redundant with numSources()
int AudioInputDefinition::getNumbChannels() const {
  const std::string& layout = pimpl->layout;
  return getNbChannelsFromChannelLayout(getChannelLayoutFromString(layout.c_str()));
}

size_t AudioInputDefinition::numSources() const { return pimpl->sources.size(); }

bool AudioInputDefinition::getIsMaster() const { return pimpl->isMaster; }

std::string& AudioInputDefinition::getLayout() const { return pimpl->layout; }

AudioSourceDefinition* AudioInputDefinition::getSource(size_t i) const {
  assert(i < pimpl->sources.size());
  return pimpl->sources[i].get();
}

Ptv::Value* AudioInputDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("name", new Parse::JsonValue((std::string)getName()));
  if (getIsMaster()) {
    res->push("master", new Parse::JsonValue(getIsMaster()));
  }
  if (!getLayout().empty()) {
    res->push("layout", new Parse::JsonValue(getLayout()));
  }
  // Sources:
  Ptv::Value* jsonSources = new Parse::JsonValue((void*)NULL);
  jsonSources->asList();
  for (size_t i = 0; i < numSources(); ++i) {
    jsonSources->asList().push_back(getSource(i)->serialize());
  }
  res->push("sources", jsonSources);
  return res;
}

void AudioInputDefinition::setIsMaster(bool b) { pimpl->isMaster = b; }

Status AudioInputDefinition::setLayout(const std::string& layout) {
  if (getChannelLayoutFromString(layout.c_str()) == UNKNOWN) {
    std::stringstream ss;
    ss << "Wrong layout " << layout << "specified for the audio input " << getName();
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, ss.str()};
  }
  pimpl->layout = layout;
  return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////
/////////// AudioInputDefinition::Pimpl ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// \brief AudioInputDefinition::Pimpl::Pimpl
///
AudioInputDefinition::Pimpl::Pimpl() : name(""), isMaster(false), layout("") {}

///
/// \brief AudioInputDefinition::Pimpl::~Pimpl
///
AudioInputDefinition::Pimpl::~Pimpl() {}

////////////////////////////////////////////////////////////////////////////////
/////////// AudioPipeDefinition ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// \brief AudioPipeDefinition::AudioPipeDefinition
///
AudioPipeDefinition::AudioPipeDefinition() : pimpl(new Pimpl()) {}

///
/// \brief AudioPipeDefinition::~AudioPipeDefinition
///
AudioPipeDefinition::~AudioPipeDefinition() {}

///
/// \brief AudioPipeDefinition::addDelayProcessor
/// \param inputName
/// \param delay
/// \return
///
Status AudioPipeDefinition::addDelayProcessor(const std::string& inputName, double delay) {
  FAIL_RETURN(isValidInput(inputName));
  PotentialValue<AudioProcessorDef*> delayDef = getProcessor(kDelayProcessorName);
  if (delayDef.ok()) {
    return delayDef.value()->addDelay(inputName, delay);
  } else {
    Potential<AudioProcessorDef> delayProc = AudioProcessorDef::createDelayProcessor(inputName, delay);
    FAIL_RETURN(delayProc.status());
    pimpl->audioProcessors.emplace_back(delayProc.release());
    return Status::OK();
  }
}

///
/// \brief AudioPipeDefinition::addGainProcessor
/// \param inputName
/// \param gaindB
/// \param reversePolarity
/// \param mute
/// \return
///
Status AudioPipeDefinition::addGainProcessor(const std::string& inputName, double gaindB, bool reversePolarity,
                                             bool mute) {
  PotentialValue<AudioProcessorDef*> procDef = getProcessor(kGainProcessorName);
  if (procDef.ok()) {
    return procDef.value()->addGain(inputName, gaindB, reversePolarity, mute);
  } else {
    FAIL_RETURN(getInput(inputName).status());
    Potential<AudioProcessorDef> procDef =
        AudioProcessorDef::createGainProcessor(inputName, gaindB, reversePolarity, mute);
    FAIL_RETURN(procDef.status());
    pimpl->audioProcessors.emplace_back(procDef.release());
    return Status::OK();
  }
}

///
/// \brief AudioPipeDefinition::addInput
/// \param newInput
///
void AudioPipeDefinition::addInput(AudioInputDefinition* newInput) { pimpl->audioInputs.emplace_back(newInput); }

///
/// \brief AudioPipeDefinition::create
/// \param value
/// \return
///
AudioPipeDefinition* AudioPipeDefinition::create(const Ptv::Value& value) {
  // Make sure value is an object.
  if (!Parse::checkType("AudioPipeDefinition", value, Ptv::Value::OBJECT)) {
    return nullptr;
  }
  std::unique_ptr<AudioPipeDefinition> res(new AudioPipeDefinition());

#define PROPAGATE_DEFAULT(call, toFill, defaultVal) \
  if (call != Parse::PopulateResult_Ok) {           \
    toFill = defaultVal;                            \
  }
  PROPAGATE_DEFAULT(Parse::populateInt("AudioPipeDefinition", value, "sampling_rate", res->pimpl->samplingRate, false),
                    res->pimpl->samplingRate, 44100);
  PROPAGATE_DEFAULT(Parse::populateInt("AudioPipeDefinition", value, "block_size", res->pimpl->blockSize, false),
                    res->pimpl->blockSize, 512);
  PROPAGATE_DEFAULT(Parse::populateString("AudioPipeDefinition", value, "debug", res->pimpl->debugFolder, false),
                    res->pimpl->debugFolder, "");
  PROPAGATE_DEFAULT(
      Parse::populateString("AudioPipeDefinition", value, "audio_selected", res->pimpl->selectedAudio, false),
      res->pimpl->selectedAudio, "");
  PROPAGATE_DEFAULT(Parse::populateBool("AudioPipeDefinition", value, "vumeter", res->pimpl->hasVuMeter, false),
                    res->pimpl->hasVuMeter, true);
#undef PROPAGATE_DEFAULT

  // Populate inputs:
  {
    std::vector<std::unique_ptr<Core::AudioInputDefinition>> audioInputs;
    parseAudioInputs(value, audioInputs);
    for (size_t i = 0; i < audioInputs.size(); ++i) {
      res->pimpl->audioInputs.push_back(std::move(audioInputs[i]));
    }
  }

  // Populate audio processors:
  {
    std::vector<std::unique_ptr<Core::AudioProcessorDef>> audioProcessors;
    if (parseAudioProcessors(value, audioProcessors).ok()) {
      for (size_t i = 0; i < audioProcessors.size(); ++i) {
        res->pimpl->audioProcessors.push_back(std::move(audioProcessors[i]));
      }
    }
  }

  // Populate audio mixes:
  {
    std::vector<std::unique_ptr<Core::AudioMixDefinition>> audioMixes;
    if (parseAudioMixes(value, audioMixes).ok()) {
      for (size_t i = 0; i < audioMixes.size(); ++i) {
        res->pimpl->audioMixes.push_back(std::move(audioMixes[i]));
      }
    }
  }

  return res.release();
}

// Creates an audioPipe from the input defined in the pano definition
AudioPipeDefinition* AudioPipeDefinition::createAudioPipeFromPanoInputs(const PanoDefinition* pano) {
  std::vector<InputParam> inputParams;
  for (audioreaderid_t i = 0; i < pano->numAudioInputs(); ++i) {
    const InputDefinition& audioInput = pano->getAudioInput(i);
    if (!audioInput.getReaderConfig().asString().empty()) {
      // Case where the reader config is path to the file
      inputParams.push_back(
          InputParam(audioInput.getReaderConfig().asString(), pano->convertAudioInputIndexToInputIndex(i), STEREO));
    } else {
      Audio::ChannelLayout audioInputLayout = STEREO;  // Make a stereo input by default
      if (audioInput.getReaderConfig().has("audio_channels")) {
        audioInputLayout = getAChannelLayoutFromNbChannels(
            static_cast<int>(audioInput.getReaderConfig().has("audio_channels")->asInt()));
      } else if (audioInput.getReaderConfig().has("channel_layout")) {
        audioInputLayout =
            getChannelLayoutFromString(audioInput.getReaderConfig().has("channel_layout")->asString().c_str());
      }
      inputParams.push_back(
          InputParam("input" + std::to_string(i), pano->convertAudioInputIndexToInputIndex(i), audioInputLayout));
    }
  }
  return VideoStitch::Core::AudioPipeDefinition::create(inputParams);
}

///
/// \brief AudioPipeDefinition::createDefault
/// \return a default audio pipeline definition without any input and any processor
///
AudioPipeDefinition* AudioPipeDefinition::createDefault() {
  std::unique_ptr<AudioPipeDefinition> res(new AudioPipeDefinition());
  // Create a default audioPipeDefinition with:
  // - blockSize at the default internal block size
  // - sampling rate at the default sampling rate
  res->setBlockSize(Audio::getDefaultBlockSize());
  res->setSamplingRate(static_cast<int>(Audio::getDefaultSamplingRate()));
  return res.release();
}

///
/// \brief AudioPipeDefinition::create
/// \param inputNames with an input id
/// \return an audio pipeline definition
///
AudioPipeDefinition* AudioPipeDefinition::create(const std::vector<InputParam>& inputParams) {
  std::unique_ptr<AudioPipeDefinition> res(new AudioPipeDefinition());
  res->setBlockSize(Audio::getDefaultBlockSize());
  res->setSamplingRate(static_cast<int>(Audio::getDefaultSamplingRate()));

  // Populate inputs
  for (size_t i = 0; i < inputParams.size(); ++i) {
    res->pimpl->audioInputs.emplace_back(AudioInputDefinition::create(inputParams[i]));
    res->getInput(i)->getSource(0)->setReaderId(inputParams[i].id);
    if (i == 0) {
      // select the first input by default
      res->setSelectedAudio(res->getInput(0)->getName());
    }

    // Create a default mix for each input with the same input name
    std::vector<std::string> tmpInput{inputParams[i].name};
    res->pimpl->audioMixes.emplace_back(AudioMixDefinition::create(inputParams[i].name, tmpInput));
  }

  return res.release();
}

///
/// \brief AudioPipeDefinition::clone clones the audio pipeline definition
/// \return an audio pipeline defintion
///
AudioPipeDefinition* AudioPipeDefinition::clone() const {
  AudioPipeDefinition* result = new AudioPipeDefinition();

  result->setBlockSize(getBlockSize());
  result->setSamplingRate(getSamplingRate());
  result->setDebugFolder(getDebugFolder());
  result->setHasVuMeter(getHasVuMeter());

  for (audioreaderid_t i = 0; i < numAudioInputs(); ++i) {
    result->pimpl->audioInputs.emplace_back(getInput(i)->clone());
  }
  // set the audio selected after cloning the input if not the selected audio will be empty
  result->setSelectedAudio(getSelectedAudio());

  for (size_t i = 0; i < numProcessors(); ++i) {
    result->pimpl->audioProcessors.emplace_back(getProcessor(i)->clone());
  }

  for (size_t i = 0; i < numAudioMixes(); ++i) {
    result->pimpl->audioMixes.emplace_back(getMix(i)->clone());
  }
  return result;
}

///
/// \brief AudioPipeDefinition::getAmbDecodingCoef
/// \return Ambisonic decoding coefficients definition
///
AmbisonicDecoderDef* AudioPipeDefinition::getAmbDecodingCoef() const { return pimpl->ambDecCoef.get(); }

///
/// \brief AudioPipeDefinition::getBlockSize
/// \return the block size of the audio pipeline
///
int AudioPipeDefinition::getBlockSize() const { return pimpl->blockSize; }

///
/// \brief AudioPipeDefinition::getControlAsBool
/// \param processor name
/// \param input name
/// \param control name
/// \return
///
PotentialValue<bool> AudioPipeDefinition::getControlAsBool(const std::string& procName, const std::string& inputName,
                                                           const std::string& controlName) {
  PotentialValue<AudioProcessorDef*> procDef = getProcessor(procName);
  FAIL_RETURN(procDef.status());
  std::vector<Ptv::Value*> ptvDelays = procDef.value()->getParameters()->asList();
  for (auto controlParam : ptvDelays) {
    if (controlParam->has("input") && controlParam->has("input")->asString() == inputName &&
        controlParam->has(controlName)) {
      return controlParam->has(controlName)->asBool();
    }
  }
  return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                "No " + controlName + " found for input " + inputName);
}

///
/// \brief AudioPipeDefinition::getControlAsDouble
/// \param processor name
/// \param input name
/// \param control name
/// \return
///
PotentialValue<double> AudioPipeDefinition::getControlAsDouble(const std::string& procName,
                                                               const std::string& inputName,
                                                               const std::string& controlName) {
  PotentialValue<AudioProcessorDef*> procDef = getProcessor(procName);
  FAIL_RETURN(procDef.status());
  std::vector<Ptv::Value*> ptvDelays = procDef.value()->getParameters()->asList();
  for (auto controlParam : ptvDelays) {
    if (controlParam->has("input") && controlParam->has("input")->asString() == inputName &&
        controlParam->has(controlName)) {
      return controlParam->has(controlName)->asDouble();
    }
  }
  return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                "No " + controlName + " found for input " + inputName);
}

///
/// \brief AudioPipeDefinition::setControlBool
/// \param processor name
/// \param input name
/// \param control to set
/// \param value to set
/// \return
///
Status AudioPipeDefinition::setControlBool(const std::string& procName, const std::string& inputName,
                                           const std::string& controlName, bool value) {
  PotentialValue<AudioProcessorDef*> procDef = getProcessor(procName);
  FAIL_RETURN(procDef.status());
  std::vector<Ptv::Value*> params = procDef.value()->getParameters()->asList();
  for (auto controlParam : params) {
    if (controlParam->has("input") && controlParam->has("input")->asString() == inputName &&
        controlParam->has(controlName)) {
      controlParam->get(controlName)->asBool() = value;
      return Status::OK();
    }
  }
  return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                "No " + controlName + " found for input " + inputName);
}

///
/// \brief AudioPipeDefinition::setControlDouble
/// \param processor name
/// \param input name
/// \param control to set
/// \param value to set
/// \return
///
Status AudioPipeDefinition::setControlDouble(const std::string& procName, const std::string& inputName,
                                             const std::string& controlName, double value) {
  PotentialValue<AudioProcessorDef*> procDef = getProcessor(procName);
  FAIL_RETURN(procDef.status());
  std::vector<Ptv::Value*> params = procDef.value()->getParameters()->asList();
  for (auto controlParam : params) {
    if (controlParam->has("input") && controlParam->has("input")->asString() == inputName &&
        controlParam->has(controlName)) {
      controlParam->get(controlName)->asDouble() = value;
      return Status::OK();
    }
  }
  return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                "No " + controlName + " found for input " + inputName);
}

///
/// \brief AudioPipeDefinition::getDebugFolder
/// \return the debug folder where audio debug files are saved
///
std::string& AudioPipeDefinition::getDebugFolder() const { return pimpl->debugFolder; }

///
/// \brief AudioPipeDefinition::getMaxDelayValue
/// \return the maximum delay value that can be used with AudioPipeDefinition::setDelay
///
double AudioPipeDefinition::getMaxDelayValue() const { return SampleDelay::getMaxDelaySeconds(); }

///
/// \brief AudioPipeDefinition::getDelay
/// \param input name
/// \return the delay value corresponding to the input name
///
PotentialValue<double> AudioPipeDefinition::getDelay(const std::string& inputName) {
  return getControlAsDouble(kDelayProcessorName, inputName, "delay");
}

///
/// \brief AudioPipeDefinition::getGain
/// \param inputName
/// \return value in dB of the gain
///
PotentialValue<double> AudioPipeDefinition::getGain(const std::string& inputName) {
  return getControlAsDouble(kGainProcessorName, inputName, "gain");
}

bool AudioPipeDefinition::getHasVuMeter() const { return pimpl->hasVuMeter; }

///
/// \brief AudioPipeDefinition::getInput
/// \param index of the input
/// \return the i-th audio inputDefition of the audio pipeline
///
AudioInputDefinition* AudioPipeDefinition::getInput(size_t i) const {
  assert(i <= pimpl->audioInputs.size());
  return pimpl->audioInputs[i].get();
}

///
/// \brief AudioPipeDefinition::getInput
/// \param inputName
/// \return the audio input definition corresponding to the input name
///
PotentialValue<AudioInputDefinition*> AudioPipeDefinition::getInput(const std::string& inputName) const {
  for (audioreaderid_t i = 0; i < numAudioInputs(); i++) {
    if (pimpl->audioInputs[i]->getName() == inputName) return pimpl->audioInputs[i].get();
  }
  return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                "No audio input " + inputName + " found");
}

std::vector<std::reference_wrapper<const std::string>> AudioPipeDefinition::getInputNames() const {
  std::vector<std::reference_wrapper<const std::string>> res;
  for (audioreaderid_t i = 0; i < numAudioInputs(); i++) {
    res.push_back(pimpl->audioInputs[i]->getName());
  }
  return res;
}

AudioMixDefinition* AudioPipeDefinition::getMix(size_t i) const {
  assert(i <= pimpl->audioMixes.size());
  return pimpl->audioMixes[i].get();
}

///
/// \brief AudioPipeDefinition::getMute
/// \param input name
/// \return mute
///
PotentialValue<bool> AudioPipeDefinition::getMute(const std::string& inputName) {
  return getControlAsBool(kGainProcessorName, inputName, "mute");
}

///
/// \brief AudioPipeDefinition::getProcessor
/// \param i index of the processor
/// \return a pointer on the i-th audio processor definition
///
AudioProcessorDef* AudioPipeDefinition::getProcessor(size_t i) const {
  assert(i <= numProcessors());
  return pimpl->audioProcessors[i].get();
}

///
/// \brief AudioPipeDefinition::getProcessor
/// \param procName
/// \return the pointer on the audio processor definition corresponding to the processor requested
///
PotentialValue<AudioProcessorDef*> AudioPipeDefinition::getProcessor(const std::string& procName) const {
  for (size_t i = 0; i < numProcessors(); i++) {
    if (pimpl->audioProcessors[i]->getName() == procName) return pimpl->audioProcessors[i].get();
  }
  return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                "No audio processor " + procName + " found");
}

PotentialValue<bool> AudioPipeDefinition::getReversePolarity(const std::string& inputName) {
  return getControlAsBool(kGainProcessorName, inputName, "reverse_polarity");
}

///
/// \brief AudioPipeDefinition::getSamplingRate
/// \return
///
int AudioPipeDefinition::getSamplingRate() const { return pimpl->samplingRate; }

///
/// \brief AudioPipeDefinition::getSelectedAudio get selected audio
/// \param
/// \return mixName
///
const std::string& AudioPipeDefinition::getSelectedAudio() const { return pimpl->selectedAudio; }

///
/// \brief AudioPipeDefinition::getSelectedInput get selected audio input
/// \return selected input
///
PotentialValue<AudioInputDefinition*> AudioPipeDefinition::getSelectedInput() const {
  for (int i = 0; i < numAudioInputs(); ++i) {
    if (pimpl->selectedAudio == getInput(i)->getName()) {
      return getInput(i);
    }
  }
  return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                "No input selected found " + pimpl->selectedAudio);
}

///
/// \brief AudioPipeDefinition::isValidInput check if the inputName corresponds to a real input
/// \param inputName
/// \return
///
Status AudioPipeDefinition::isValidInput(const std::string& inputName) const {
  for (audioreaderid_t i = 0; i < numAudioInputs(); i++) {
    if (pimpl->audioInputs[i]->getName() == inputName) return Status::OK();
  }
  return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                "No audio input " + inputName + " found");
}

///
/// \brief AudioPipeDefinition::numAudioInputs
/// \return the number of inputs defined
///
audioreaderid_t AudioPipeDefinition::numAudioInputs() const { return (audioreaderid_t)pimpl->audioInputs.size(); }

///
/// \brief AudioPipeDefinition::numAudioMixes
/// \return the number of mixes defined
///
size_t AudioPipeDefinition::numAudioMixes() const { return pimpl->audioMixes.size(); }

///
/// \brief AudioPipeDefinition::numProcessors
/// \return the number of audio processors defined
///
size_t AudioPipeDefinition::numProcessors() const { return pimpl->audioProcessors.size(); }

#define READERS_ARE_NOT_CONSISTENT_MSG                                                                           \
  (Logger::verbose(kAudioPipeTag) << "Readers are not consistent between audio pipe (reader " << i               \
                                  << ") def and pano def (reader " << readerId << ") audio input channel index " \
                                  << channel << " doesn't match the number of audio channels of the pano def "   \
                                  << pano->getInput(readerId).getReaderConfig().has("audio_channels")->asInt()   \
                                  << std::endl)

///
/// \brief AudioPipeDefinition::readersAreConsistent check if readers of pano and audio pipe are consistent
/// \param pano panorama definition
/// \return
///
bool AudioPipeDefinition::readersAreConsistent(PanoDefinition* pano) const {
  std::vector<audioreaderid_t> panoAudioReaders;
  for (auto index = 0; index < pano->numInputs(); ++index) {
    if (pano->getInput(index).getIsAudioEnabled()) {
      panoAudioReaders.push_back(index);
    }
  }

  for (audioreaderid_t i = 0; i < numAudioInputs(); ++i) {
    const AudioInputDefinition* inputDef = getInput(i);
    for (size_t j = 0; j < inputDef->numSources(); ++j) {
      readerid_t readerId = inputDef->getSource(j)->getReaderId();
      if (readerId >= pano->numInputs() || readerId < 0) {
        return false;
      }
      if (std::find(panoAudioReaders.begin(), panoAudioReaders.end(), readerId) != panoAudioReaders.end()) {
        size_t channel = inputDef->getSource(j)->getChannel();
        // Check channel index is coherent for this reader config
        if (pano->getInput(readerId).getReaderConfig().has("audio_channels")) {
          if (int(channel) >= pano->getInput(readerId).getReaderConfig().has("audio_channels")->asInt()) {
            READERS_ARE_NOT_CONSISTENT_MSG;
            return false;
          }
        } else if (pano->getInput(readerId).getReaderConfig().has("channel_layout")) {
          // Case for audio procedurals
          int nbChannelsFromPano = getNbChannelsFromChannelLayout(getChannelLayoutFromString(
              pano->getInput(readerId).getReaderConfig().has("channel_layout")->asString().c_str()));
          if (int(channel) >= nbChannelsFromPano) {
            READERS_ARE_NOT_CONSISTENT_MSG;
            return false;
          }
        } else if (channel >= 2) {
          // Supports only mono or stereo for input files or streaming
          Logger::verbose(kAudioPipeTag) << "Audio input " << i << " : needs channel " << channel
                                         << " doesn't match the reader config "
                                         << pano->getInput(readerId).getReaderConfig().asString() << std::endl;
          return false;
        }
      } else {
        return false;
      }
    }
  }

  // Case for only one audio input, check number of channels
  if (panoAudioReaders.size() == 1 && numAudioInputs() == 1) {
    if (pano->getInput(panoAudioReaders[0]).getReaderConfig().has("audio_channels") &&
        pano->getInput(panoAudioReaders[0]).getReaderConfig().has("audio_channels")->asInt() !=
            int(getInput(0)->numSources())) {
      return false;
    }
  }
  return true;
}

///
/// \brief AudioPipeDefinition::removeInput removes the i-th input (index start at 0)
/// \param index of the input to remove
/// \return
///
Status AudioPipeDefinition::removeInput(audioreaderid_t i) {
  if (i >= numAudioInputs()) {
    return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                  "Cannot remove audio input " + std::to_string(i));
  } else {
    pimpl->audioInputs.erase(pimpl->audioInputs.begin() + i);
    return Status::OK();
  }
}

///
/// \brief AudioPipeDefinition::removeProcessor removes the processor procName
/// \param procName
/// \return
///
Status AudioPipeDefinition::removeProcessor(const std::string& procName) {
  for (size_t i = 0; i < numProcessors(); i++) {
    if (pimpl->audioProcessors[i]->getName() == procName) {
      pimpl->audioProcessors.erase(pimpl->audioProcessors.begin() + i);
      return Status::OK();
    }
  }
  return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
          "No audio processor " + procName + " found"};
}

///
/// \brief AudioPipeDefinition::replaceInput replace the i-th input by the new one
/// \param index of the input to be replaced
/// \param name of the new input
/// \return
///
Status AudioPipeDefinition::replaceInput(audioreaderid_t i, const std::string& name) {
  if (i >= numAudioInputs()) {
    return Status(Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
                  "Cannot remove audio input " + std::to_string(i));
  } else {
    pimpl->audioInputs[i]->setName(name);
    return Status::OK();
  }
}

///
/// \brief AudioPipeDefinition::setAmbDecodingCoef sets the ambisonic decoding coef
/// \param ptv value
/// \return
///
Status AudioPipeDefinition::setAmbDecodingCoef(Ptv::Value* ptv) {
  std::unique_ptr<AmbisonicDecoderDef> ambDecDef(new AmbisonicDecoderDef(*ptv));
  pimpl->ambDecCoef = std::move(ambDecDef);
  return Status::OK();
}

///
/// \brief AudioPipeDefinition::serialize
/// \return
///
Ptv::Value* AudioPipeDefinition::serialize() const {
  std::unique_ptr<Ptv::Value> res(Ptv::Value::emptyObject());
  res->push("sampling_rate", new Parse::JsonValue(getSamplingRate()));
  res->push("block_size", new Parse::JsonValue(getBlockSize()));
  res->push("audio_selected", new Parse::JsonValue(getSelectedAudio()));
  if (getHasVuMeter()) {
    // save only if you need a vumeter
    res->push("vumeter", new Parse::JsonValue(true));
  }
  if (!getDebugFolder().empty()) {
    res->push("debug", new Parse::JsonValue(getDebugFolder()));
  }

  // Inputs
  Ptv::Value* jsonInputs = new Parse::JsonValue((void*)nullptr);
  for (audioreaderid_t i = 0; i < numAudioInputs(); ++i) {
    jsonInputs->asList().push_back(getInput(i)->serialize());
  }
  res->push("audio_inputs", jsonInputs);

  // Processors
  if (numProcessors() > 0) {
    Ptv::Value* jsonProcessors = new Parse::JsonValue((void*)nullptr);
    for (size_t i = 0; i < numProcessors(); ++i) {
      jsonProcessors->asList().push_back(getProcessor(i)->serialize());
    }
    res->push("audio_processors", jsonProcessors);
  }

  // Mixes
  Ptv::Value* jsonMixes = new Parse::JsonValue((void*)nullptr);
  for (size_t i = 0; i < numAudioMixes(); ++i) {
    jsonMixes->asList().push_back(getMix(i)->serialize());
  }
  res->push("audio_mixes", jsonMixes);

  return res.release();
}

///
/// \brief AudioPipeDefinition::setBlockSize set the block size of the audio pipeline (by default 512 samples)
/// \param block size
///
void AudioPipeDefinition::setBlockSize(const int blockSize) { pimpl->blockSize = blockSize; }

///
/// \brief AudioPipeDefinition::setDebugFolder
///        set the path where the audio debug files should be saved
///        (by default empty string, means no debug files saved)
/// \param path
///
void AudioPipeDefinition::setDebugFolder(const std::string& s) const { pimpl->debugFolder = s; }

///
/// \brief AudioPipeDefinition::setDelay
///        Sets a delay for the input inputName
/// \param input name
/// \param delay value in s
/// \return Status
///
Status AudioPipeDefinition::setDelay(const std::string& inputName, double delay) {
  return setControlDouble(kDelayProcessorName, inputName, "delay", delay);
}

///
/// \brief AudioPipeDefinition::setGain
/// \param input name
/// \param gain value
/// \return
///
Status AudioPipeDefinition::setGain(const std::string& inputName, double gain) {
  return setControlDouble(kGainProcessorName, inputName, "gain", gain);
}

void AudioPipeDefinition::setHasVuMeter(bool has) { pimpl->hasVuMeter = has; }

///
/// \brief AudioPipeDefinition::setMute
/// \param input name
/// \param mute
/// \return
///
Status AudioPipeDefinition::setMute(const std::string& inputName, bool mute) {
  return setControlBool(kGainProcessorName, inputName, "mute", mute);
}

///
/// \brief AudioPipeDefinition::setReversePolarity
/// \param input name
/// \param reversePolarity
/// \return
///
Status AudioPipeDefinition::setReversePolarity(const std::string& inputName, bool reversePolarity) {
  return setControlBool(kGainProcessorName, inputName, "reverse_polarity", reversePolarity);
}

///
/// \brief AudioPipeDefinition::setSelectedAudio
/// \param inputName
/// \return
///
Status AudioPipeDefinition::setSelectedAudio(const std::string& inputName) {
  FAIL_RETURN(getInput(inputName).status());
  pimpl->selectedAudio = inputName;
  return Status::OK();
}

///
/// \brief AudioPipeDefinition::setSamplingRate
///        Set the sampling rate of the audio pipeline (by default 44100 Hz)
/// \param samplingRate in Hz
///
void AudioPipeDefinition::setSamplingRate(const int samplingRate) { pimpl->samplingRate = samplingRate; }

////////////////////////////////////////////////////////////////////////////////
/////////// AudioPipeDefinition::Pimpl /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// \brief AudioPipeDefinition::Pimpl::Pimpl
///
AudioPipeDefinition::Pimpl::Pimpl()
    : debugFolder(""), blockSize(0), samplingRate(0), hasVuMeter(false), audioInputs(0), audioProcessors(0) {}

///
/// \brief AudioPipeDefinition::Pimpl::~Pimpl
///
AudioPipeDefinition::Pimpl::~Pimpl() {}

////////////////////////////////////////////////////////////////////////////////
/////////// AudioProcessorDef //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// \brief AudioProcessorDef::AudioProcessorDef
///
AudioProcessorDef::AudioProcessorDef() : pimpl(new Pimpl()) {}

///
/// \brief AudioProcessorDef::~AudioProcessorDef
///
AudioProcessorDef::~AudioProcessorDef() { delete pimpl; }

Status AudioProcessorDef::addDelay(const std::string& inputName, double delay) {
  if (getParameters(inputName)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
            "Cannot add delay for " + inputName + " there's already one delay set for this input"};
  }
  Ptv::Value* params = getParameters();
  Ptv::Value* newDelayParam = Ptv::Value::emptyObject();
  newDelayParam->push("input", new Parse::JsonValue(inputName));
  newDelayParam->push("delay", new Parse::JsonValue(delay));
  params->asList().push_back(newDelayParam);
  return Status::OK();
}

Status AudioProcessorDef::addGain(const std::string& inputName, double gaindB, bool reversePolarity, bool mute) {
  if (getParameters(inputName)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
            "Cannot add gain for " + inputName + " there's already one gain set for this input"};
  }
  Ptv::Value* params = getParameters();
  Ptv::Value* newParam = Ptv::Value::emptyObject();
  newParam->push("input", new Parse::JsonValue(inputName));
  newParam->push("gain", new Parse::JsonValue(gaindB));
  newParam->push("reverse_polarity", new Parse::JsonValue(reversePolarity));
  newParam->push("mute", new Parse::JsonValue(mute));
  params->asList().push_back(newParam);
  return Status::OK();
}

Potential<AudioProcessorDef> AudioProcessorDef::create(const Ptv::Value& value) {
  std::unique_ptr<AudioProcessorDef> res(new AudioProcessorDef());
#define PROPAGATE_NOK(call)               \
  if (call != Parse::PopulateResult_Ok) { \
    return nullptr;                       \
  }
  PROPAGATE_NOK(Parse::populateString("AudioProcessorDefinition", value, "name", res->pimpl->name, true));
#undef PROPAGATE_NOK
  if (value.has("params")) {
    res->pimpl->parameters = std::unique_ptr<Ptv::Value>(value.has("params")->clone());
  } else {
    std::stringstream ss;
    ss << "No mandatory field params for audio processors. Will generate an audio pipe without processor "
       << res->pimpl->name;
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, ss.str()};
  }

  return res.release();
}

Potential<AudioProcessorDef> AudioProcessorDef::createDelayProcessor(const std::string& inputName, double delay) {
  std::unique_ptr<AudioProcessorDef> res(new AudioProcessorDef());
  res->setName(kDelayProcessorName);
  Ptv::Value* jsonParams = new Parse::JsonValue((void*)nullptr);
  Ptv::Value* jsonParam = Ptv::Value::emptyObject();
  jsonParam->push("input", new Parse::JsonValue(inputName));
  jsonParam->push("delay", new Parse::JsonValue(delay));
  jsonParams->asList().push_back(jsonParam);
  res->setParameters(jsonParams);
  return res.release();
}

Potential<AudioProcessorDef> AudioProcessorDef::createGainProcessor(const std::string& inputName, double gaindB,
                                                                    bool reversePolarity, bool mute) {
  std::unique_ptr<AudioProcessorDef> res(new AudioProcessorDef());
  res->setName(kGainProcessorName);
  Ptv::Value* jsonParams = new Parse::JsonValue((void*)nullptr);
  Ptv::Value* jsonParam = Ptv::Value::emptyObject();
  jsonParam->push("input", new Parse::JsonValue(inputName));
  jsonParam->push("gain", new Parse::JsonValue(gaindB));
  jsonParam->push("reverse_polarity", new Parse::JsonValue(reversePolarity));
  jsonParam->push("mute", new Parse::JsonValue(mute));
  jsonParams->asList().push_back(jsonParam);
  res->setParameters(jsonParams);
  return res.release();
}

AudioProcessorDef* AudioProcessorDef::clone() const {
  std::unique_ptr<AudioProcessorDef> res(new AudioProcessorDef());
  res->setName(getName());
  res->pimpl->parameters = std::unique_ptr<Ptv::Value>(getParameters()->clone());

  return res.release();
}

std::string& AudioProcessorDef::getName() const { return pimpl->name; }

Ptv::Value* AudioProcessorDef::getParameters() const { return pimpl->parameters.get(); }

Ptv::Value* AudioProcessorDef::getParameters(const std::string& inputName) const {
  for (Ptv::Value* param : pimpl->parameters->asList()) {
    if (param->has("input") && param->has("input")->asString() == inputName) {
      return param;
    }
  }
  return nullptr;
}

Ptv::Value* AudioProcessorDef::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("name", new Parse::JsonValue(getName()));

  Ptv::Value* jsonParams = new Parse::JsonValue((void*)nullptr);
  std::vector<Ptv::Value*> params = getParameters()->asList();
  for (Ptv::Value* param : params) {
    Ptv::Value* jsonParam = Ptv::Value::emptyObject();

    std::string inputName = param->has("input")->asString();
    jsonParam->push("input", new Parse::JsonValue(inputName));

    if (getName() == kDelayProcessorName) {
      double delay = param->has("delay")->asDouble();
      jsonParam->push("delay", new Parse::JsonValue(delay));
    }

    if (getName() == kGainProcessorName) {
      double gain = param->has("gain")->asDouble();
      jsonParam->push("gain", new Parse::JsonValue(gain));
      bool mute = param->has("mute")->asBool();
      jsonParam->push("mute", new Parse::JsonValue(mute));
      bool reversePolarity = param->has("reverse_polarity")->asBool();
      jsonParam->push("reverse_polarity", new Parse::JsonValue(reversePolarity));
    }

    if (getName() == kAmbRotateProcessorName) {
      int64_t order = param->has("order")->asInt();
      jsonParam->push("order", new Parse::JsonValue(order));
      // This is pretty ugly but unfortunately I don't know how to do better
      std::vector<Ptv::Value*>* tmpVector = new std::vector<Ptv::Value*>;
      std::vector<Ptv::Value*> offsets = param->has("offset")->asList();
      tmpVector->push_back(new Parse::JsonValue(offsets[0]->asDouble()));
      tmpVector->push_back(new Parse::JsonValue(offsets[1]->asDouble()));
      tmpVector->push_back(new Parse::JsonValue(offsets[2]->asDouble()));
      Ptv::Value* tmp = new Parse::JsonValue(tmpVector);
      jsonParam->push("offset", tmp);
    }

    jsonParams->asList().push_back(jsonParam);
  }

  res->push("params", jsonParams);
  // TODO add serialization for each time we add an audio processor

  return res;
}

void AudioProcessorDef::setName(const std::string& name) { pimpl->name = name; }

void AudioProcessorDef::setParameters(Ptv::Value* params) { pimpl->parameters = std::unique_ptr<Ptv::Value>(params); }

///
/// \brief AudioProcessorDef::Pimpl::Pimpl
///
AudioProcessorDef::Pimpl::Pimpl() : name("") {}

///
/// \brief AudioProcessorDef::Pimpl::~Pimpl
///
AudioProcessorDef::Pimpl::~Pimpl() {}

////////////////////////////////////////////////////////////////////////////////
/////////// AudioMixDefinition /////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// \brief AudioMixDefinition::AudioMixDefinition
///
AudioMixDefinition::AudioMixDefinition() : pimpl(new Pimpl()) {}

Status AudioMixDefinition::addInput(const std::string& name) {
  pimpl->inputs.push_back(name);
  return Status::OK();
}

Potential<AudioMixDefinition> AudioMixDefinition::create(const Ptv::Value& value) {
  // Make sure value is an object.
  if (!Parse::checkType("AudioMixDefinition", value, Ptv::Value::OBJECT)) {
    return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, "Wrong type of audio mix definition"};
  }
  std::unique_ptr<AudioMixDefinition> res(new AudioMixDefinition());
  if (Parse::populateString("AudioMixDefinition", value, "name", res->pimpl->name, true) != Parse::PopulateResult_Ok) {
    return nullptr;
  }
  // Populate inputs
  {
    const Ptv::Value* var = value.has("inputs");
    if (!Parse::checkType("inputs", *var, Ptv::Value::LIST)) {
      return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration,
              "Wrong audio mix definition type (invalid inputs)"};
    }
    const std::vector<Ptv::Value*>& inputs = var->asList();

    for (size_t i = 0; i < inputs.size(); ++i) {
      res->pimpl->inputs.emplace_back(inputs[i]->asString());
    }
  }
  return res.release();
}

AudioMixDefinition* AudioMixDefinition::create(const std::string& name, std::vector<std::string> inputs) {
  std::unique_ptr<AudioMixDefinition> res(new AudioMixDefinition());
  res->pimpl->name = name;
  for (auto input : inputs) {
    res->addInput(input);
  }
  return res.release();
}

AudioMixDefinition* AudioMixDefinition::clone() const {
  AudioMixDefinition* res = new AudioMixDefinition();
  res->setName(getName());
  res->pimpl->inputs = getInputs();
  return res;
}

std::string& AudioMixDefinition::getInput(size_t i) const {
  assert(i < pimpl->inputs.size());
  return pimpl->inputs[i];
}

std::vector<std::string> AudioMixDefinition::getInputs() const { return pimpl->inputs; }

std::string& AudioMixDefinition::getName() const { return pimpl->name; }

size_t AudioMixDefinition::numInputs() const { return pimpl->inputs.size(); }

Ptv::Value* AudioMixDefinition::serialize() const {
  Ptv::Value* res = Ptv::Value::emptyObject();
  res->push("name", new Parse::JsonValue(getName()));
  Ptv::Value* jsonParams = new Parse::JsonValue((void*)nullptr);
  std::vector<std::string> inputs = getInputs();
  for (auto input : inputs) {
    jsonParams->asList().push_back(new Parse::JsonValue(input));
  }
  res->push("inputs", jsonParams);
  return res;
}

Status AudioMixDefinition::setName(const std::string& name) {
  pimpl->name = name;
  return Status::OK();
}

///
/// \brief AudioMixDefinition::~AudioMixDefinition
///
AudioMixDefinition::~AudioMixDefinition() {}

///
/// \brief AudioProcessorDef::Pimpl::Pimpl
///
AudioMixDefinition::Pimpl::Pimpl() : name(""), inputs() {}

///
/// \brief AudioProcessorDef::Pimpl::~Pimpl
///
AudioMixDefinition::Pimpl::~Pimpl() {}

}  // namespace Core
}  // namespace VideoStitch
