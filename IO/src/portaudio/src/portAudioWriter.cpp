// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "portAudioWriter.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "portaudio.h"
#include <sstream>

namespace VideoStitch {
namespace Output {
static const std::string kPaWriterTag = "PortaudioWriter";

Potential<PortAudioWriter> PortAudioWriter::create(const Ptv::Value* config,
                                                   const Plugin::VSWriterPlugin::Config& runtime) {
  std::stringstream ss;
  PaDeviceIndex nbDevices;
  paDevice paDev;
  const PaDeviceInfo* device = nullptr;
  PaError paErr = Pa_Initialize();
  if (paErr != paNoError) {
    ss << "Creation of the portaudio writer failed. Could not initialize PortAudio. " << Pa_GetErrorText(paErr);
    return {Origin::Output, ErrType::InvalidConfiguration, ss.str()};
  }

  std::string outputName;
  if (Parse::populateString("PortAudioWriter", *config, "name", outputName, true) == Parse::PopulateResult_WrongType) {
    ss << "Wrong type for the output name";
    Pa_Terminate();
    return {Origin::Output, ErrType::InvalidConfiguration, ss.str()};
  }

  nbDevices = Pa_GetDeviceCount();

  PaDeviceIndex i;
  for (i = 0; i < nbDevices; i++) {
    device = Pa_GetDeviceInfo(i);
    std::string deviceName(Pa_GetDeviceInfo(i)->name);
    if (deviceName.find(outputName) != std::string::npos) {
      Logger::info(kPaWriterTag) << "Found device " << deviceName << std::endl;
      break;
    }
  }

  if (i >= nbDevices) {
    i = Pa_GetDefaultOutputDevice();
    device = Pa_GetDeviceInfo(i);
    Logger::info(kPaWriterTag) << "Specified device " << outputName << " not found. Using default device." << std::endl;
  }

  paDev.devInfo = device;
  paDev.stream = nullptr;
  paDev.params.channelCount = Audio::getNbChannelsFromChannelLayout(runtime.layout);
  paDev.params.device = i;
  paDev.params.sampleFormat = paFloat32;
  paDev.params.suggestedLatency = device->defaultHighOutputLatency;
  paDev.params.hostApiSpecificStreamInfo = nullptr;
  paDev.sampleRate = double(Audio::getIntFromSamplingRate(runtime.rate));
  paDev.offset = 0;
  std::unique_ptr<PortAudioWriter> paWriter(new PortAudioWriter(runtime, paDev));
  paErr = Pa_OpenStream(&paDev.stream, nullptr, /* no input */
                        &paDev.params, paDev.sampleRate, Audio::getDefaultBlockSize() * paDev.params.channelCount,
                        paClipOff, /* output will be in-range, so no need to clip */
                        paPlayCallback, paWriter.get());

  if (paErr != paNoError) {
    Pa_Terminate();
    ss << "Could not open output stream of the device " << paDev.devInfo->name;
    return {Origin::Output, ErrType::RuntimeError, ss.str()};
  }

  paErr = Pa_SetStreamFinishedCallback(paDev.stream, &paStreamFinished);

  if (paErr != paNoError) {
    Pa_Terminate();
    Pa_CloseStream(paDev.stream);
    ss << "Could not set stream finished callback for " << paDev.devInfo->name;
    return {Origin::Output, ErrType::RuntimeError, ss.str()};
  }

  const PaHostApiInfo* apiInf = Pa_GetHostApiInfo(paDev.devInfo->hostApi);
  Logger::info(kPaWriterTag) << "Audio device: " << paDev.devInfo->name << std::endl;
  Logger::info(kPaWriterTag) << "Sample rate:  " << paDev.sampleRate << std::endl;
  Logger::info(kPaWriterTag) << "Channels:     " << paDev.params.channelCount << std::endl;
  Logger::info(kPaWriterTag) << "Audio delay:  " << paDev.offset << std::endl;
  Logger::info(kPaWriterTag) << "Audio API:    " << apiInf->name << std::endl;

  const PaStreamInfo* streamInf = Pa_GetStreamInfo(paDev.stream);
  Logger::info(kPaWriterTag) << "stream info output latency " << streamInf->outputLatency << " s" << std::endl;
  Logger::info(kPaWriterTag) << "suggested output latency " << paDev.params.suggestedLatency << " s" << std::endl;

  paErr = Pa_StartStream(paDev.stream);
  if (paErr != paNoError) {
    Pa_CloseStream(paDev.stream);
    Pa_Terminate();
    ss << "Could not start output stream " << paDev.devInfo->name << " " << paErr;
    return {Origin::Output, ErrType::RuntimeError, ss.str()};
  }
  Logger::info(kPaWriterTag) << "Stream started on " << paDev.devInfo->name << std::endl;

  paWriter->dev = paDev;
  return paWriter.release();
}

bool PortAudioWriter::handles(const Ptv::Value* config) { return config->has("type")->asString() == "portaudio"; }

PortAudioWriter::~PortAudioWriter() {
  PaError paErr;
  stopping = true;

  paErr = Pa_StopStream(dev.stream);
  if (paErr != paNoError) {
    Logger::error(kPaWriterTag) << "Could not stop stream " << getName() << " " << paErr << std::endl;
  }
  paErr = Pa_CloseStream(dev.stream);
  if (paErr != paNoError) {
    Logger::error(kPaWriterTag) << "Could not close stream " << getName() << " " << paErr << std::endl;
  }
  Pa_Terminate();
}

void PortAudioWriter::pushAudio(Audio::Samples& audioSamples) {
  std::unique_lock<std::mutex> lock(m);
  size_t nbItems =
      audioSamples.getNbOfSamples() * Audio::getNbChannelsFromChannelLayout(audioSamples.getChannelLayout());
  audioData.push(reinterpret_cast<const float*>(audioSamples.getSamples()[0]), nbItems);
}

PortAudioWriter::PortAudioWriter(const Plugin::VSWriterPlugin::Config& runtime, const paDevice paDev)
    : Output(runtime.name), AudioWriter(runtime.rate, runtime.depth, runtime.layout), stopping(false), dev(paDev) {}

void PortAudioWriter::startStream() {
  std::unique_lock<std::mutex> lock(m);
  PaError paErr = Pa_StartStream(dev.stream);
  if (paErr != paNoError) {
    Logger::error(kPaWriterTag) << "Could not start output stream " << dev.devInfo->name << " " << paErr << std::endl;
  }
}

int PortAudioWriter::paPlayCallback(const void* /*inputBuffer*/, void* outputBuffer, unsigned long framesPerBuffer,
                                    const PaStreamCallbackTimeInfo* /*timeInfo*/, PaStreamCallbackFlags /*statusFlags*/,
                                    void* userData) {
  PortAudioWriter* that = static_cast<PortAudioWriter*>(userData);
  if (!that->stopping) {
    std::unique_lock<std::mutex> lock(that->m);
    float* out = static_cast<float*>(outputBuffer);
    if (that->audioData.size() >= framesPerBuffer * 2) {
      that->audioData.pop(out, framesPerBuffer * 2);
    } else if (!that->audioData.empty()) {
      that->audioData.pop(out, that->audioData.size());
    } else {
      Logger::verbose(kPaWriterTag) << " " << that->dev.devInfo->name << " Nothing to play" << std::endl;
    }
    return paContinue;
  } else {
    return paComplete;
  }
}

void PortAudioWriter::paStreamFinished(void* userData) {
  PortAudioWriter* that = static_cast<PortAudioWriter*>(userData);
  Logger::info(kPaWriterTag) << "Stream completed " << that->dev.devInfo->name << std::endl;
}

}  // namespace Output
}  // namespace VideoStitch
