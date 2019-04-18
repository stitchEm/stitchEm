// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioPipeFactory.hpp"

#include "parse/json.hpp"
#include "common/container.hpp"

#include "libvideostitch/audioPipeDef.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Audio {

Potential<AudioPipeline> AudioPipeFactory::create(const Core::AudioPipeDefinition& audioPipeDef,
                                                  const Core::PanoDefinition& pano) {
  BlockSize blockSize = getBlockSizeFromInt(audioPipeDef.getBlockSize());
  SamplingRate samplingRate = getSamplingRateFromInt(audioPipeDef.getSamplingRate());
  AudioPipeline* res = new AudioPipeline(blockSize, samplingRate, audioPipeDef, pano);
  res->setDebugFolder(audioPipeDef.getDebugFolder());
  res->createAudioProcessors(audioPipeDef);
  FAIL_RETURN(res->applyProcessorParam(audioPipeDef));

  for (audioreaderid_t inIdx = 0; inIdx < (audioreaderid_t)audioPipeDef.numAudioInputs(); ++inIdx) {
    Core::AudioInputDefinition* inputDef = audioPipeDef.getInput(inIdx);
    if (inputDef->getIsMaster()) {
      res->inputMaster = inputDef->getName();
    }

    if (!inputDef->getLayout().empty()) {
      ChannelLayout l = getChannelLayoutFromString(inputDef->getLayout().c_str());
      if (inputDef->numSources() != size_t(getNbChannelsFromChannelLayout(l))) {
        Logger::error(kAudioPipeTag) << "Layout specified (" << inputDef->getLayout()
                                     << ") not compatible with nb of sources " << inputDef->numSources() << std::endl;
        inputDef->setLayout(getStringFromChannelLayout(getAChannelLayoutFromNbChannels(int(inputDef->numSources()))));
        Logger::error(kAudioPipeTag) << "Use " << inputDef->getLayout() << " instead" << std::endl;
      }
    }
  }

  if (res->inputMaster.empty() && audioPipeDef.numAudioInputs() > 0) {
    res->inputMaster = audioPipeDef.getInput(0)->getName();
    audioPipeDef.getInput(0)->setIsMaster(true);
  }
  Logger::get(Logger::Verbose) << "[audiopipeline]: master is set to input \"" << res->inputMaster << "\"" << std::endl;
  return res;
}

}  // namespace Audio
}  // namespace VideoStitch
