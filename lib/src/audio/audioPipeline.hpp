// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "audio/resampler.hpp"

#include "libvideostitch/ambisonic.hpp"
#include "libvideostitch/audio.hpp"
#include "libvideostitch/audioBlock.hpp"
#include "libvideostitch/audioWav.hpp"
#include "libvideostitch/matrix.hpp"
#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/audioPipeDef.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/quaternion.hpp"
#include "libvideostitch/status.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <mutex>

namespace VideoStitch {
namespace Audio {

typedef std::map<readerid_t, AudioBlock> audioBlockReaderMap_t;
typedef std::map<groupid_t, audioBlockReaderMap_t> audioBlockGroupMap_t;

// originally a vector<audioBlockGroupMap_t>, but this doesn't compile in MSVC
// (https://stackoverflow.com/questions/58700780/vector-of-maps-of-non-copyable-objects-fails-to-compile-in-msvc)
// -> using std::map, where key is audio block index and can be disregarded
typedef std::map<int64_t, audioBlockGroupMap_t> audioBlocks_t;

static const std::string kAudioPipeTag{"audiopipeline"};

class AudioPipeline {
 public:
  AudioPipeline(const BlockSize blockSize, const SamplingRate samplingRate,
                const Core::AudioPipeDefinition& audioPipeDef, const Core::PanoDefinition& pano);
  ~AudioPipeline();

  bool addOutput(std::shared_ptr<Output::AudioWriter>);
  Status applyProcessorParam(const Core::AudioPipeDefinition& def);
  void applyRotation(double yaw, double pitch, double roll);
  PotentialValue<AudioObject*> getAudioProcessor(const std::string& inputName, const std::string& procName) const;
  bool hasVuMeter(const std::string& inputName) const;
  PotentialValue<std::vector<double>> getPeakValues(const std::string& inputName) const;
  PotentialValue<std::vector<double>> getRMSValues(const std::string& inputName) const;
  Vector3<double> getRotation() const;
  bool hasAudio() const;
  Status process(audioBlockGroupMap_t& samples);
  bool removeOutput(const std::string& id);
  Status resetPano(const Core::PanoDefinition& newPano);
  void resetRotation();
  Status setDecodingCoefficients(const AmbisonicDecoderDef& def);
  Status setDelay(double delay);
  void setInput(const std::string& inputName);

 private:
  friend class AudioPipeFactory;

  class AudioOutputResampler {
   public:
    AudioOutputResampler(std::shared_ptr<Output::AudioWriter> o, SamplingRate internalSamplingRate,
                         size_t internalBlockSize, const ambCoefTable_t& decodeCoef)
        : output(o),
          ambDec(o->getChannelLayout(), decodeCoef),
          ambEnc(AmbisonicOrder::FIRST_ORDER, AmbisonicNorm::SN3D) {
      rsp = AudioResampler::create(internalSamplingRate, SamplingDepth::DBL_P, o->getSamplingRate(),
                                   o->getSamplingDepth(), o->getChannelLayout(), internalBlockSize);
    }

    ~AudioOutputResampler() { delete rsp; }

    void pushAudio(AudioBlock& b) {
      Samples s;
      AudioBlock c;
      if (b.getLayout() == AMBISONICS_WXYZ && output->getChannelLayout() != AMBISONICS_WXYZ) {
        ambDec.step(c, b);
      }

      if (b.getLayout() != AMBISONICS_WXYZ && output->getChannelLayout() == AMBISONICS_WXYZ) {
        ambEnc.step(c, b);
      }

      if (c.empty()) {
        rsp->resample(b, s);
      } else {
        rsp->resample(c, s);
      }

      if (s.getNbOfSamples() > 0) {
        output->pushAudio(s);
      }
    }

   private:
    std::shared_ptr<Output::AudioWriter> output;
    AudioResampler* rsp;
    AmbDecoder ambDec;
    AmbEncoder ambEnc;
  };

  Status addProcessor(const Core::AudioProcessorDef& procDef);
  Status createAudioProcessors(const Core::AudioPipeDefinition& def);
  AudioBlock& getBlockFromInputName(const std::string& inputName);
  AudioBlock& getBlockFromMixName(const std::string& mixName);
  BlockSize getBlockSize() const;
  std::string getInputNameFromId(audioreaderid_t i) const;
  PotentialValue<audioreaderid_t> getInputIdFromName(const std::string& inputName) const;
  SamplingRate getSamplingRate() const;
  bool isInputEmpty(const std::string& inputName) const;
  bool isMixEmpty(const std::string& mixName) const;
  Status makeAudioInputs(audioBlockGroupMap_t& audioPerGroup);
  Status makeAudioMixes();
  void pushAudio();
  void recordDebugFile(std::string name, AudioBlock& input);
  void setDebugFolder(const std::string& s);

  typedef std::map<std::string, std::vector<std::unique_ptr<AudioObject>>> inputPath_t;
  const BlockSize blockSize;
  const SamplingRate samplingRate;
  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef;
  std::unique_ptr<Core::PanoDefinition> pano;

  std::map<std::string, WavWriter*> recorder;
  std::string debugFolder = "";

  std::map<std::string, AudioBlock> inputs;  /// audio data defined in the audio_inputs of the audio_pipe definition
  std::map<std::string, AudioBlock> mixes;   /// audio data defined in the audio_mixes of the audio_pipe definition
  std::string selectedInput;
  mutable std::mutex inputPathsLock;
  inputPath_t inputPaths;  // input paths are the audio processing chain

  mutable std::mutex paramsLock;
  std::map<std::string, AudioOutputResampler*> outputs;

  mtime_t lastTimestamp;
  mtime_t lastTimestampMaster;
  std::string inputMaster;

  Vector3<double> stabOrientation;  // yaw/pitch/roll
  ambCoefTable_t ambDecodingCoef;
};

}  // namespace Audio
}  // namespace VideoStitch
