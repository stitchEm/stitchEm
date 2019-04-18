// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audiogen.hpp"

#include "sigGen.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/ptv.hpp"

namespace VideoStitch {
namespace Audio {

Input::AudioReader* AudioGenFactory::create(readerid_t id, const Ptv::Value& config) {
  SamplingRate rate = getSamplingRateFromInt(static_cast<int>(getDefaultSamplingRate()));
  ChannelLayout layout = ChannelLayout::MONO;
  std::vector<double> freqs;

  if (config.has("sampling_rate")) rate = getSamplingRateFromInt((int)config.has("sampling_rate")->asInt());
  if (rate == SamplingRate::SR_NONE) {
    Logger::get(Logger::Error) << "Sampling Rate " << config.has("sampling_rate")->asInt() << " not managed"
                               << std::endl;
    return nullptr;
  }

  if (config.has("channel_layout")) {
    layout = getChannelLayoutFromString(config.has("channel_layout")->asString().c_str());
  } else if (config.has("audio_channels")) {
    layout = getAChannelLayoutFromNbChannels(config.has("audio_channels")->asInt());
  }

  if (layout == ChannelLayout::UNKNOWN) {
    Logger::get(Logger::Error) << "Channel layout " << config.has("channel_layout")->asString() << " not managed"
                               << std::endl;
    return nullptr;
  }

  if (config.has("freqs")) {
    freqs.clear();
    std::vector<Ptv::Value*> listVals = config.has("freqs")->asList();
    for (auto val : listVals) {
      if (val->isConvertibleTo(Ptv::Value::DOUBLE)) {
        freqs.push_back((double)val->asDouble());
      }
    }
  } else {
    // By default set a freq to a multiple of 220 Hz
    for (int c = 0; c < getNbChannelsFromChannelLayout(layout); c++) {
      freqs.push_back(220 * (c + 1));
    }
  }

  Logger::get(Logger::Info) << "rate " << getIntFromSamplingRate(rate) << " layout "
                            << getStringFromChannelLayout(layout) << " f " << freqs[0] << " Hz";
  return new SigGenSineInput(id, rate, layout, freqs);
}

}  // namespace Audio
}  // namespace VideoStitch
