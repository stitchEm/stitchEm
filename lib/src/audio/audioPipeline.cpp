// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioPipeline.hpp"

#include "envelopeDetector.hpp"
#include "gain.hpp"
#include "sampleDelay.hpp"
#include "resampler.hpp"

#include "orah/orahProcessor.hpp"
#include "common/angles.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/status.hpp"

#include <sstream>
#include <string>
#include <cmath>

namespace VideoStitch {
namespace Audio {

AudioPipeline::AudioPipeline(const Audio::BlockSize bs, const Audio::SamplingRate sr,
                             const Core::AudioPipeDefinition& audioPipeDef, const Core::PanoDefinition& p)
    : blockSize(bs),
      samplingRate(sr),
      audioPipeDef(audioPipeDef.clone()),
      pano(p.clone()),
      recorder(),
      selectedInput((audioPipeDef.numAudioInputs() > 0) ? audioPipeDef.getSelectedAudio() : std::string()),
      lastTimestamp(-1),
      lastTimestampMaster(-1),
      inputMaster(""),
      stabOrientation({0, 0, 0}),
      ambDecodingCoef((audioPipeDef.getAmbDecodingCoef() != nullptr)
                          ? audioPipeDef.getAmbDecodingCoef()->getCoefficients()
                          : ambCoefTable_t()) {}

AudioPipeline::~AudioPipeline() {
  for (auto& kv : recorder) {
    Logger::get(Logger::Verbose) << "delete wav writer " << kv.first << std::endl;
    kv.second->close();
    delete kv.second;
  }
  for (auto& kv : outputs) {
    Logger::get(Logger::Verbose) << "delete audio output resamplers" << kv.first << std::endl;
    delete kv.second;
  }
}

Status AudioPipeline::addProcessor(const Core::AudioProcessorDef& procDef) {
  Logger::info(kAudioPipeTag) << "Add audio processor " << procDef.getName() << std::endl;
  std::vector<Ptv::Value*> ptvParams = procDef.getParameters()->asList();
  bool added = false;
  for (Ptv::Value* param : ptvParams) {
    if (!param->has("input")) {
      Logger::get(Logger::Warning) << "No audio input specified cannot create audio processor " << procDef.getName()
                                   << std::endl;
      continue;
    }
    std::string inputName = param->has("input")->asString();

    if (procDef.getName() == Core::kDelayProcessorName) {
      const double delay = param->has("delay")->asDouble();
      if (0 <= delay && delay <= kMaxDelayTime) {
        PotentialValue<audioreaderid_t> inputId = getInputIdFromName(inputName);
        if (!inputId.ok()) {
          Logger::get(Logger::Warning) << "Cannot create audio delay processor." << std::endl;
          continue;
        } else {
          SampleDelay* del = new SampleDelay();
          del->setDelaySeconds(delay);
          inputPaths[inputName].push_back(std::unique_ptr<AudioObject>(del));
          added = true;
        }
      } else {
        Logger::get(Logger::Warning) << "audio delay value " << delay << " s out of bounds [0 s .. " << kMaxDelayTime
                                     << " s]" << std::endl;
        continue;
      }
    }

    else if (procDef.getName() == Core::kGainProcessorName && param->has("gain") && param->has("mute") &&
             param->has("reverse_polarity")) {
      double gaindB = param->has("gain")->asDouble();
      if (kGainMin <= gaindB && gaindB <= kGainMax) {
        Gain* gainObj = new Gain(gaindB, param->has("reverse_polarity")->asBool(), param->has("mute")->asBool());
        inputPaths[inputName].push_back(std::unique_ptr<AudioObject>(gainObj));
        added = true;
      } else {
        Logger::get(Logger::Warning) << "Wrong or missing parameters for processor " + procDef.getName() << std::endl;
      }
    }

    else if (procDef.getName() == Core::kAmbRotateProcessorName) {
      for (Ptv::Value* param : ptvParams) {
        if (!param->has("input")) {
          Logger::get(Logger::Warning) << "No audio input specified cannot create audio processor " << procDef.getName()
                                       << std::endl;
          continue;
        }
        std::string inputName = param->has("input")->asString();
        AmbisonicOrder o = AmbisonicOrder::FIRST_ORDER;
        if (param->has("order")) {
          o = getAmbisonicOrderFromInt(int(param->has("order")->asInt()));
        }
        std::unique_ptr<AmbRotator> ambObj(new AmbRotator(o));
        if (param->has("offset")) {
          std::vector<Ptv::Value*> offsets = param->has("offset")->asList();
          if (offsets.size() != 3) {
            return {Origin::AudioPipeline, ErrType::InvalidConfiguration, "Wrong offset rotation of the ambRotator"};
          }
          ambObj->setRotationOffset(offsets[0]->asDouble(), offsets[1]->asDouble(), offsets[2]->asDouble());
        } else {
          // For YAW, apply offset to compensate for the fact that the camera defaults
          // to looking "right" with respect to the Ethernet cable. We look forward by
          // default.
          ambObj->setRotationOffset(M_PI_2, 0., 0.);
        }
        inputPaths[inputName].push_back(std::unique_ptr<AudioObject>(ambObj.release()));
        added = true;
      }
    }

    else if (procDef.getName() == Core::kOrah2bProcessorName) {
      OrahProcessor* o2b = new OrahProcessor();
      inputPaths[inputName].push_back(std::unique_ptr<AudioObject>(o2b));
      added = true;
    } else {
      Logger::get(Logger::Warning) << "wrong parameters for the audio processor" + procDef.getName() << std::endl;
    }
  }

  if (added) {
    return Status::OK();
  } else {
    return {Origin::AudioPipeline, ErrType::InvalidConfiguration,
            "No processor " + procDef.getName() + " has been added"};
  }
}

bool AudioPipeline::addOutput(std::shared_ptr<Output::AudioWriter> o) {
  std::lock_guard<std::mutex> lk(paramsLock);
  Logger::info(kAudioPipeTag) << "Add output " << o->getName() << std::endl;
  auto outputRsp = new AudioOutputResampler(o, getSamplingRateFromInt(audioPipeDef->getSamplingRate()),
                                            audioPipeDef->getBlockSize(), ambDecodingCoef);
  outputs.emplace(std::make_pair(o->getName(), outputRsp));
  return true;
}

Status AudioPipeline::applyProcessorParam(const Core::AudioPipeDefinition& def) {
  std::lock_guard<std::mutex> lk(inputPathsLock);
  for (size_t i = 0; i < def.numProcessors(); ++i) {
    // Parse delay parameters
    std::string procName = def.getProcessor(i)->getName();
    std::vector<Ptv::Value*> ptvParams = def.getProcessor(i)->getParameters()->asList();
    for (Ptv::Value* param : ptvParams) {
      if (!param->has("input")) {
        Logger::get(Logger::Warning) << "Cannot apply parameters, no audio input specified for the processor"
                                     << procName << std::endl;
        continue;
      }
      std::string inputName = param->has("input")->asString();
      PotentialValue<AudioObject*> proc = getAudioProcessor(inputName, procName);
      FAIL_RETURN(proc.status());
      if (param->has("delay")) {
        double delay = param->has("delay")->asDouble();
        static_cast<SampleDelay*>(proc.value())->setDelaySeconds(delay);
      } else if (param->has("gain") && param->has("mute") && param->has("reverse_polarity")) {
        static_cast<Gain*>(proc.value())->setGainDB(param->has("gain")->asDouble());
        static_cast<Gain*>(proc.value())->setMute(param->has("mute")->asBool());
        static_cast<Gain*>(proc.value())->setReversePolarity(param->has("reverse_polarity")->asBool());
      } else if (procName == "ambRotator" && param->has("order") && param->has("offset")) {
        std::vector<Ptv::Value*> offsets = param->has("offset")->asList();
        if (offsets.size() != 3) {
          return {Origin::AudioPipeline, ErrType::InvalidConfiguration, "Wrong offset rotation of the ambRotator"};
        }
        static_cast<AmbRotator*>(proc.value())
            ->setRotationOffset(offsets[0]->asDouble(), offsets[1]->asDouble(), offsets[2]->asDouble());
      }
    }
  }
  return Status::OK();
}

void AudioPipeline::applyRotation(double yaw, double pitch, double roll) {
  Vector3<double> v(yaw, pitch, roll);
  stabOrientation = v;
  PotentialValue<AudioObject*> proc = getAudioProcessor("camera", "ambRotator");
  if (proc.ok()) {
    static_cast<AmbRotator*>(proc.value())->setRotation(yaw, pitch, roll);
  }
}

Status AudioPipeline::createAudioProcessors(const Core::AudioPipeDefinition& def) {
  std::lock_guard<std::mutex> lk(inputPathsLock);
  inputPaths.clear();
  if (audioPipeDef->getHasVuMeter()) {
    for (auto inputName : audioPipeDef->getInputNames()) {
      // In the case where the audio pipeline is created by the default constructor,
      // only one mono vumeter by audio input will be declared
      VuMeter* vm = new VuMeter(getIntFromSamplingRate(samplingRate));
      inputPaths[inputName].push_back(std::unique_ptr<AudioObject>(vm));
    }
  }

  for (size_t i = 0; i < def.numProcessors(); ++i) {
    // See if processor is in enabled processors list
    std::string procName = def.getProcessor(i)->getName();
    auto namePos = std::find(Core::kEnabledAudioProcessors.begin(), Core::kEnabledAudioProcessors.end(), procName);
    if (namePos != Core::kEnabledAudioProcessors.end()) {
      FAIL_RETURN(addProcessor(*def.getProcessor(i)));
    }
  }
  return Status::OK();
}

PotentialValue<AudioObject*> AudioPipeline::getAudioProcessor(const std::string& inputName,
                                                              const std::string& procName) const {
  if (inputPaths.find(inputName) != inputPaths.end()) {
    for (size_t i = 0; i < inputPaths.at(inputName).size(); ++i) {
      AudioObject* proc = inputPaths.at(inputName)[i].get();
      if (proc->getName() == procName) {
        return proc;
      }
    }
  }
  return Status(Origin::AudioPipeline, ErrType::InvalidConfiguration,
                "No proc " + procName + " for input " + inputName);
}

AudioBlock& AudioPipeline::getBlockFromInputName(const std::string& inputName) {
  if (inputs.find(inputName) != inputs.end()) {
    return inputs[inputName];
  }
  Logger::get(Logger::Verbose) << "No audio block found for \"" << inputName
                               << "\". Return the audio block from the first input by default." << std::endl;
  return inputs.begin()->second;
}

AudioBlock& AudioPipeline::getBlockFromMixName(const std::string& mixName) {
  if (mixes.find(mixName) != mixes.end()) {
    return mixes[mixName];
  }
  Logger::get(Logger::Verbose) << "No audio block found for mix \"" << mixName
                               << "\". Return the audio block from the first mix by default." << std::endl;
  return mixes.begin()->second;
}

BlockSize AudioPipeline::getBlockSize() const { return blockSize; }

std::string AudioPipeline::getInputNameFromId(audioreaderid_t i) const {
  if (i >= audioPipeDef->numAudioInputs()) {
    return std::string();
  }

  return audioPipeDef->getInput(i)->getName();
}

PotentialValue<audioreaderid_t> AudioPipeline::getInputIdFromName(const std::string& inputName) const {
  for (readerid_t i = 0; i < static_cast<audioreaderid_t>(audioPipeDef->numAudioInputs()); ++i) {
    if (audioPipeDef->getInput(i)->getName() == inputName) {
      return i;
    }
  }
  std::stringstream ss;
  ss << "No audio input " << inputName << " found ";
  return PotentialValue<readerid_t>(Status(Origin::AudioPipeline, ErrType::InvalidConfiguration, ss.str()));
}

bool hasReaderData(audioBlockGroupMap_t& audioPerGroup, readerid_t readerId) {
  for (auto& audioGr : audioPerGroup) {
    if (audioGr.second.find(readerId) != audioGr.second.end()) {
      return true;
    }
  }
  return false;
}

AudioBlock& getReaderData(audioBlockGroupMap_t& audioPerGroup, readerid_t readerId) {
  for (auto& audioGr : audioPerGroup) {
    if (audioGr.second.find(readerId) != audioGr.second.end()) {
      return audioGr.second.at(readerId);
    }
  }
  Logger::get(Logger::Warning) << "[audiopipeline] Getting data from " << readerId
                               << " reader not found. Return first audio data by default" << std::endl;
  return audioPerGroup.begin()->second.begin()->second;
}

bool AudioPipeline::hasVuMeter(const std::string& inputName) const {
  std::lock_guard<std::mutex> lk(inputPathsLock);
  return getAudioProcessor(inputName, "vumeter").ok();
}

PotentialValue<std::vector<double>> AudioPipeline::getPeakValues(const std::string& inputName) const {
  std::lock_guard<std::mutex> lk(inputPathsLock);
  PotentialValue<AudioObject*> vumeter = getAudioProcessor(inputName, "vumeter");
  if (!vumeter.ok()) {
    vumeter = getAudioProcessor(selectedInput, "vumeter");
  }
  FAIL_RETURN(vumeter.status());
  return static_cast<VuMeter*>(vumeter.value())->getPeakValues();
}

PotentialValue<std::vector<double>> AudioPipeline::getRMSValues(const std::string& inputName) const {
  std::lock_guard<std::mutex> lk(inputPathsLock);
  PotentialValue<AudioObject*> vumeter = getAudioProcessor(inputName, "vumeter");
  if (!vumeter.ok()) {
    vumeter = getAudioProcessor(selectedInput, "vumeter");
  }
  FAIL_RETURN(vumeter.status());
  return static_cast<VuMeter*>(vumeter.value())->getRmsValues();
}

Vector3<double> AudioPipeline::getRotation() const { return stabOrientation; }

SamplingRate AudioPipeline::getSamplingRate() const { return samplingRate; }

bool AudioPipeline::isInputEmpty(const std::string& inputName) const {
  if (inputs.find(inputName) != inputs.end()) {
    return inputs.at(inputName).empty();
  }
  return true;
}

bool AudioPipeline::isMixEmpty(const std::string& mixName) const {
  if (mixes.find(mixName) != mixes.end()) {
    return mixes.at(mixName).empty();
  }
  return true;
}

bool AudioPipeline::hasAudio() const { return (audioPipeDef->numAudioInputs() > 0); }

Status AudioPipeline::makeAudioInputs(audioBlockGroupMap_t& audioPerGroup) {
  for (audioreaderid_t inIdx = 0; inIdx < (audioreaderid_t)audioPipeDef->numAudioInputs(); ++inIdx) {
    Core::AudioInputDefinition* inputDef = audioPipeDef->getInput(inIdx);
    AudioBlock tmpBlk;
    ChannelLayout inL = getChannelLayoutFromString(inputDef->getLayout().c_str());
    tmpBlk.setChannelLayout((inL != UNKNOWN ? inL : getAChannelLayoutFromNbChannels(int(inputDef->numSources()))));
    if (getNbChannelsFromChannelLayout(tmpBlk.getLayout()) != int(inputDef->numSources())) {
      std::stringstream ss;
      ss << "Layout " << getStringFromChannelLayout(tmpBlk.getLayout()) << " does not match the number of sources "
         << inputDef->numSources();
      return {Origin::AudioPipelineConfiguration, ErrType::InvalidConfiguration, ss.str()};
    }
    size_t srcIdx = 0;
    for (auto& track : tmpBlk) {
      Core::AudioSourceDefinition* srcDef = inputDef->getSource(srcIdx++);
      if (!hasReaderData(audioPerGroup, srcDef->getReaderId())) {
        continue;
      }
      // The channel map is part of the track. As we only want to copy the data and not the channel map, we need to save
      // it before.
      ChannelMap curChan = track.channel();
      track = std::move(getReaderData(audioPerGroup,
                                      srcDef->getReaderId())[getChannelMapFromChannelIndex(int(srcDef->getChannel()))]);
      track.setChannel(curChan);
      tmpBlk.setTimestamp(getReaderData(audioPerGroup, srcDef->getReaderId()).getTimestamp());
    }

    if (!tmpBlk.empty()) {
      // Some inputs can be empty if there are not loaded at the same time
      inputs[inputDef->getName()] = std::move(tmpBlk);
    }
  }

  if (inputs.find(inputMaster) != inputs.end()) {
    if (inputs.at(inputMaster).getTimestamp() < lastTimestampMaster) {  // ->when the user seeks with the timeline
      Logger::get(Logger::Verbose) << "[audiopipeline] reset timeline" << std::endl;
      lastTimestamp = -1;
    }
    lastTimestampMaster = inputs.at(inputMaster).getTimestamp();
  }
  return Status::OK();
}

Status AudioPipeline::makeAudioMixes() {
  if (inputs.empty()) {
    return Status::OK();
  }
  for (size_t mixIdx = 0; mixIdx < audioPipeDef->numAudioMixes(); ++mixIdx) {
    Core::AudioMixDefinition* mixDef = audioPipeDef->getMix(mixIdx);
    AudioBlock empty;
    AudioBlock& tmpBlk = empty;
    if (!isInputEmpty(mixDef->getInput(0))) {
      tmpBlk = std::move(getBlockFromInputName(mixDef->getInput(0)));
    }
    for (size_t inIdx = 1; inIdx < mixDef->numInputs(); inIdx++) {
      std::string inputName = mixDef->getInput(inIdx);
      if (!isInputEmpty(inputName)) {
        tmpBlk += getBlockFromInputName(inputName);
      }
    }
    if (!tmpBlk.empty()) {
      mixes[mixDef->getName()] = std::move(tmpBlk);
    }
  }
  return Status::OK();
}

Status AudioPipeline::process(audioBlockGroupMap_t& samples) {
  std::lock_guard<std::mutex> lk(inputPathsLock);
  FAIL_RETURN(makeAudioInputs(samples));

  // Apply all input paths
  for (auto& inputPath : inputPaths) {
    if (inputs.find(inputPath.first) != inputs.end()) {
      recordDebugFile(inputPath.first + "input", inputs[inputPath.first]);
      for (auto& proc : inputPath.second) {
        proc->step(inputs.at(inputPath.first));  // if processor works in place
        recordDebugFile(inputPath.first + proc->getName(), inputs[inputPath.first]);
      }
    }
  }
  FAIL_RETURN(makeAudioMixes());

  pushAudio();

  inputs.clear();
  mixes.clear();

  return Status::OK();
}

void AudioPipeline::pushAudio() {
  std::lock_guard<std::mutex> lk(paramsLock);
  if (isMixEmpty(selectedInput)) {
    return;
  }

  AudioBlock& blkToSend = getBlockFromMixName(selectedInput);

  // Ensure monotonic timestamps
  if (blkToSend.getTimestamp() > lastTimestamp) {
    Logger::get(Logger::Verbose) << "[audiopipeline] push audio mix " << selectedInput << " send nb samples "
                                 << blkToSend.numSamples() << " at timestamp " << blkToSend.getTimestamp() << std::endl;
    for (auto& kv : outputs) {
      kv.second->pushAudio(blkToSend);
    }
    // Update last timestamp sent
    lastTimestamp = blkToSend.getTimestamp();
  }
}

void AudioPipeline::recordDebugFile(std::string name, AudioBlock& input) {
  if (debugFolder.empty()) {
    return;
  }
  if (!recorder[name]) {
    recorder[name] = new WavWriter(debugFolder + "/" + name + ".wav", input.getLayout(), getDefaultSamplingRate());
  }
  recorder[name]->step(input);
}

bool AudioPipeline::removeOutput(const std::string& id) {
  std::lock_guard<std::mutex> lk(paramsLock);
  auto output = outputs.find(id);
  if (output == outputs.end()) {
    return false;
  }
  auto cleanPtr = output->second;
  outputs.erase(id);
  delete cleanPtr;
  return true;
}

Status AudioPipeline::resetPano(const Core::PanoDefinition& newPano) {
  pano = std::unique_ptr<Core::PanoDefinition>(newPano.clone());
  return Status::OK();
}

void AudioPipeline::resetRotation() { stabOrientation = Vector3<double>(0, 0, 0); }

Status AudioPipeline::setDecodingCoefficients(const AmbisonicDecoderDef& def) {
  ambDecodingCoef = def.getCoefficients();
  return Status::OK();
}

void AudioPipeline::setDebugFolder(const std::string& s) { debugFolder = s; }

Status AudioPipeline::setDelay(double delay) {
  std::lock_guard<std::mutex> lk(inputPathsLock);
  Logger::get(Logger::Verbose) << "Set delay to " << delay << " s" << std::endl;
  PotentialValue<AudioObject*> proc = getAudioProcessor(selectedInput, "delay");
  FAIL_RETURN(proc.status());
  return static_cast<SampleDelay*>(proc.value())->setDelaySeconds(delay);
}

void AudioPipeline::setInput(const std::string& inputName) {
  std::lock_guard<std::mutex> lk(paramsLock);
  selectedInput = inputName;
}

}  // namespace Audio
}  // namespace VideoStitch
