// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "portAudioReader.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include <thread>
#include <chrono>
#include <iostream>

namespace VideoStitch {
namespace Input {

#define CHECK_AND_STORE_CONFIG_INT(str, param) Parse::populateInt("PortAudio reader", *config, str, param, false)
#define CHECK_AND_STORE_CONFIG_DOUBLE(str, param) Parse::populateDouble("PortAudio reader", *config, str, param, false)
#define CHECK_AND_STORE_CONFIG_STRING(str, param) Parse::populateString("PortAudio reader", *config, str, param, false)

const std::string kTag("PortAudio reader");

namespace {
enum StreamState { Stopped = 0, Active = 1 };
}

// ------------------------------ Lifecycle ----------------------------

bool PortAudioReader::handles(const Ptv::Value* config) {
  return config && config->has("type") &&
         ((config->has("type")->asString() == "portaudio") ||
          (config->has("type")->asString() == "openAL"));  // So we can open legacy projects
}

#define PORT_AUDIO_CLEANUP_AND_FAIL \
  {                                 \
    Pa_Terminate();                 \
    delete newDev;                  \
    return nullptr;                 \
  }

PortAudioReader* PortAudioReader::create(readerid_t id, const Ptv::Value* config) {
  auto* newDev = new paDevice;

  // Pa_Terminate() MUST be called once for each call to Pa_Initialize() or memory will leak.
  // If Pa_Initialize() does not return paNoError, Pa_Terminate() MUST NOT be called.
  // See portaudio.h
  PaError paErr = Pa_Initialize();
  if (paErr != paNoError) {
    Logger::error(kTag) << "Could not initialize PortAudio. " << Pa_GetErrorText(paErr) << std::endl;
    delete newDev;
    return nullptr;
  }

  // If there is a device specified in config, we will try to use that one.
  // If not, we will try to use the default system device.
  std::string cfgInput;
  newDev->params.device = paNoDevice;
  newDev->devInfo = Pa_GetDeviceInfo(0);
  if (CHECK_AND_STORE_CONFIG_STRING("name", cfgInput) == Parse::PopulateResult::OK) {
    if (cfgInput != "default") {
      int numDevices = Pa_GetDeviceCount();
      if (numDevices < 1) {
        Logger::error(kTag) << "Could not find any supported devices." << std::endl;
        // We won't even be able to find a default device, so return.
        PORT_AUDIO_CLEANUP_AND_FAIL
      }
      const PaDeviceInfo* info;
      for (int i = 0; i < numDevices; i++) {
        info = Pa_GetDeviceInfo(i);
        std::string devName(info->name);
        std::string apiName(Pa_GetHostApiInfo(info->hostApi) != nullptr ? Pa_GetHostApiInfo(info->hostApi)->name : "");
        if (cfgInput.find(devName) != std::string::npos && cfgInput.find(apiName) != std::string::npos) {
          newDev->params.device = i;
          newDev->devInfo = info;
          break;
        }
      }
      if (newDev->params.device == paNoDevice) {
        Logger::warning(kTag) << "Specified device \"" << cfgInput << "\" was not found." << std::endl;
      } else {
        if (CHECK_AND_STORE_CONFIG_DOUBLE("sampling_rate", newDev->sampleRate) != Parse::PopulateResult::OK &&
            CHECK_AND_STORE_CONFIG_DOUBLE("audio_sample_rate", newDev->sampleRate) != Parse::PopulateResult::OK) {
          Logger::warning(kTag) << "No sample rate specified in configuration. Using default ("
                                << newDev->devInfo->defaultSampleRate << ")." << std::endl;
          newDev->sampleRate = newDev->devInfo->defaultSampleRate;
        }
        if (CHECK_AND_STORE_CONFIG_INT("audio_channels", newDev->params.channelCount) != Parse::PopulateResult::OK) {
          Logger::warning(kTag) << "No channel count specified in configuration. Using 2 (stereo)." << std::endl;
          newDev->params.channelCount = 2;
        }
        if (newDev->params.channelCount < 1) {
          Logger::warning(kTag) << "Specified channel count is 0." << std::endl;
          newDev->params.device = paNoDevice;
        }
      }
    } else {
      newDev->params.device = Pa_GetDefaultInputDevice();
      if (newDev->params.device == paNoDevice) {
        Logger::get(Logger::Error) << "[PortAudio] Could not get default device" << std::endl;
        PORT_AUDIO_CLEANUP_AND_FAIL
      }
      newDev->devInfo = Pa_GetDeviceInfo(newDev->params.device);
      newDev->params.channelCount = 2;
      newDev->sampleRate = Audio::getDefaultSamplingRate();
    }
  }

  if (newDev->params.device == paNoDevice) {
    // If no device found, return a nullptr to raise an error in the plugin manager
    Logger::error(kTag) << "No audio input initialized." << std::endl;
    delete newDev;
    return nullptr;
  }

  newDev->params.sampleFormat = paFloat32;  // Internal working format
  newDev->params.suggestedLatency = newDev->devInfo->defaultHighInputLatency;
  newDev->params.hostApiSpecificStreamInfo = nullptr;

  // See if we have an offset delay saved in the preset
  int newOffset = 0;
  if (CHECK_AND_STORE_CONFIG_INT("audio_delay", newOffset) != Parse::PopulateResult::OK) {
    newDev->offset = 0;
  }
  if (newOffset < 0) {
    Logger::warning(kTag) << " Negative delay is not allowed. Setting audio_delay to 0." << std::endl;
    newDev->offset = 0;
  } else {
    newDev->offset = static_cast<mtime_t>(newOffset);
  }

  PortAudioReader* newReader = new PortAudioReader(id, newDev);

  assert(newDev->params.channelCount > 0);
  newReader->audioBufferSize = static_cast<size_t>(newDev->sampleRate * 20.0) *
                               newDev->params.channelCount;  // 20 second buffer (delay included)

  paErr = Pa_OpenStream(&newReader->dev->stream,
                        &newReader->dev->params,  // Input paramsters
                        nullptr,                  // Output parameters
                        newReader->dev->sampleRate,
                        0,  // Auto block size
                        paClipOff, paCaptureCallback, newReader);

  if (paErr != paNoError) {
    Logger::error(kTag) << " Could not open audio stream. " << Pa_GetErrorText(paErr) << std::endl;
    delete newReader;
    return nullptr;
  }

  const PaHostApiInfo* apiInf = Pa_GetHostApiInfo(newDev->devInfo->hostApi);
  Logger::info(kTag) << " Audio device: " << newDev->devInfo->name << std::endl;
  Logger::info(kTag) << " Sample rate:  " << newDev->sampleRate << std::endl;
  Logger::info(kTag) << " Channels:     " << newDev->params.channelCount << std::endl;
  Logger::info(kTag) << " Audio delay:  " << newDev->offset << std::endl;
  Logger::info(kTag) << " Audio API:    " << apiInf->name << std::endl;

  return newReader;
}

#undef PORT_AUDIO_CLEANUP_AND_FAIL

PortAudioReader::PortAudioReader(readerid_t id, paDevice* newDev)
    : Reader(id),
      AudioReader(Audio::getAChannelLayoutFromNbChannels(newDev->params.channelCount),
                  Audio::getSamplingRateFromInt((int)newDev->sampleRate), Audio::SamplingDepth::FLT),
      dev(newDev),
      audioDataTimestamp(0),
      audioDataTimestampLast(0),
      audioBufferSize(0),
      audioBufferOverflow(false) {
  size_t bufferDelay =
      static_cast<size_t>(((double)dev->offset / 1000000.0) * dev->sampleRate * dev->params.channelCount);
  audioData.assign(bufferDelay, 0.0);
}

PortAudioReader::~PortAudioReader() {
  stopping = true;
  // XXX: Pa_StopStream() is blocking and waits for the current output buffers to finish flushing.
  //      Pa_AbortStream() is non-blocking and can also be used, but it does not wait for
  //      current buffers to be flushed.
  if (dev->stream) {
    if (Pa_IsStreamActive(dev->stream) == StreamState::Active) {
      Pa_StopStream(dev->stream);
    }
    Pa_CloseStream(dev->stream);
  }
  Pa_Terminate();
  if (dev) {
    delete dev;
  }
}

void PortAudioReader::startStream() {
  if (Pa_IsStreamActive(dev->stream) == StreamState::Stopped) {
    /* Acquires the lock to have audioDataTimestamp set before the callback is processed */
    std::unique_lock<std::mutex> lock(m);
    PaError paErr = Pa_StartStream(dev->stream);
    if (paErr != paNoError) {
      Logger::debug(kTag) << "Could not start audio stream. " << Pa_GetErrorText(paErr) << std::endl;
      return;
    }
    int timeout = 100;
    while (Pa_IsStreamActive(dev->stream) != StreamState::Active) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      if (0 == timeout) {
        Logger::error(kTag) << "Timed out waiting for stream to start" << std::endl;
        assert(false && "PortAudio could not be started");
        return;
      }
      --timeout;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double streamTime = Pa_GetStreamTime(dev->stream);
    if (streamTime > 0) {
      audioDataTimestamp = static_cast<mtime_t>(streamTime * 1000000);
    } else {
      Logger::warning(kTag) << "Unable to get valid Audio timestamps : " << streamTime << std::endl;
      audioDataTimestamp = 0;
    }
    audioDataTimestampLast = audioDataTimestamp;
    Logger::info(kTag) << "Audio timestamps starting at " << audioDataTimestamp << std::endl;
  }
}

// -------------------------- Reading --------------------------

size_t PortAudioReader::available() {
  assert(dev->params.channelCount != 0 && "Audio device has no channels!?");
  startStream();
  return audioData.size() / dev->params.channelCount;
}

bool PortAudioReader::eos() { return stopping; }

ReadStatus PortAudioReader::readSamples(size_t nbSamples, Audio::Samples& samples) {
  uint8_t* raw[MAX_AUDIO_CHANNELS];

  // Start stream if it is not running
  startStream();

  std::unique_lock<std::mutex> lock(m);
  assert(dev->params.channelCount != 0);
  if (audioData.size() / dev->params.channelCount < nbSamples) {
    Logger::debug(kTag) << "Reader starved" << std::endl;
    return ReadStatus::fromCode<ReadStatusCode::TryAgain>();
  }
  if (stopping) {
    return ReadStatus::fromCode<ReadStatusCode::EndOfFile>();
  }

  raw[0] = new uint8_t[nbSamples * sizeof(float) * dev->params.channelCount];
  size_t read = audioData.pop((float*)raw[0], nbSamples * dev->params.channelCount);
  assert(read == nbSamples * dev->params.channelCount);

  samples =
      Audio::Samples(getSpec().sampleRate, getSpec().sampleDepth, getSpec().layout, audioDataTimestamp, raw, nbSamples);

  if (audioDataTimestampLast != audioDataTimestamp) {
    Logger::verbose(kTag) << "Audio timestamps updated from Stream : " << audioDataTimestampLast << " -> "
                          << audioDataTimestamp << " (" << (audioDataTimestamp - audioDataTimestampLast) << ")"
                          << std::endl;
  }

  /* Estimate timestamp for next samples */
  audioDataTimestamp += nbSamples * 1000000 / getIntFromSamplingRate(getSpec().sampleRate);
  audioDataTimestampLast = audioDataTimestamp;

  return ReadStatus::OK();
}

Status PortAudioReader::seekFrame(mtime_t) { return Status::OK(); }

/* NOTE: This function is called from a separate thread in libportaudio */
int PortAudioReader::paCaptureCallback(const void* inputBuffer, void* /* outputBuffer */, unsigned long framesPerBuffer,
                                       const PaStreamCallbackTimeInfo* /*timeInfo*/,
                                       PaStreamCallbackFlags /* statusFlags */, void* data) {
  PortAudioReader* that = static_cast<PortAudioReader*>(data);

  if (!that->stopping) {
    const float* in = static_cast<const float*>(inputBuffer);
    size_t size = framesPerBuffer * that->dev->params.channelCount;

    /* use Pa_GetStreamTime() instead of timeInfo->currentTime which is not available on all platform */
    PaTime streamTime = Pa_GetStreamTime(that->dev->stream);
    std::lock_guard<std::mutex> lock(that->m);

    if (that->audioData.size() + size > that->audioBufferSize) {
      /* avoid flooding with buffer overflow message */
      if (that->audioBufferOverflow == false) {
        Logger::verbose(kTag) << "Sample buffer overflow" << std::endl;
      }
      that->audioBufferOverflow = true;
      that->audioData.erase(size);
    } else {
      that->audioBufferOverflow = false;
    }

    if (streamTime > 0) {
      mtime_t newTime = static_cast<mtime_t>(streamTime * 1000000.0) + that->dev->offset;
      newTime -= static_cast<mtime_t>((that->audioData.size() + size) * 1000000.0 /
                                      (that->dev->sampleRate * that->dev->params.channelCount));
      if (newTime - that->audioDataTimestamp > mtime_t(0)) {
        /* update timestamp with real-time info instead of estimation while keeping them monotonic */
        that->audioDataTimestamp = newTime;
      }
    }

    that->audioData.push(in, size);
  }

  return 0;
}

}  // namespace Input
}  // namespace VideoStitch
