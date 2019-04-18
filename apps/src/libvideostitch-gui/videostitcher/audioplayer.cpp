// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audioplayer.hpp"
#include "libvideostitch/logging.hpp"

#include <QAudioOutput>
#include <QAudioDeviceInfo>
#include <iostream>

#define ERROR(tag) VideoStitch::Logger::error(tag)
#define WARNING(tag) VideoStitch::Logger::warning(tag)
#define INFO(tag) VideoStitch::Logger::info(tag)
#define VERBOSE(tag) VideoStitch::Logger::verbose(tag)
#define DEBUG(tag) VideoStitch::Logger::debug(tag)

namespace {
static const std::string tag = "AudioPlayer";
VideoStitch::Audio::SamplingDepth fmt2depth(QAudioFormat fmt) {
  // Manage only interleaved data
  switch (fmt.sampleType()) {
    case QAudioFormat::SampleType::SignedInt:
      if (fmt.sampleSize() == 16) {
        return VideoStitch::Audio::SamplingDepth::INT16;
      } else if (fmt.sampleSize() == 24) {
        return VideoStitch::Audio::SamplingDepth::INT24;
      } else if (fmt.sampleSize() == 32) {
        return VideoStitch::Audio::SamplingDepth::INT32;
      }
      break;
    case QAudioFormat::SampleType::UnSignedInt:
      if (fmt.sampleSize() == 8) {
        return VideoStitch::Audio::SamplingDepth::UINT8;
      }
      break;
    case QAudioFormat::SampleType::Float:
      if (fmt.sampleSize() == 32) {
        return VideoStitch::Audio::SamplingDepth::FLT;
      } else if (fmt.sampleSize() == 64) {
        return VideoStitch::Audio::SamplingDepth::DBL;
      }
      break;
    case QAudioFormat::SampleType::Unknown:
      return VideoStitch::Audio::SamplingDepth::SD_NONE;
  }
  return VideoStitch::Audio::SamplingDepth::SD_NONE;
}

VideoStitch::Audio::ChannelLayout fmt2layout(QAudioFormat fmt) {
  return VideoStitch::Audio::getAChannelLayoutFromNbChannels(fmt.channelCount());
}
}  // namespace

AudioPlayer::AudioPlayer(QObject* parent)
    : Output("playback"),
      QObject(parent),
      VideoStitch::Output::AudioWriter(VideoStitch::Audio::getSamplingRateFromInt(
                                           QAudioDeviceInfo::defaultOutputDevice().preferredFormat().sampleRate()),
                                       fmt2depth(QAudioDeviceInfo::defaultOutputDevice().preferredFormat()),
                                       fmt2layout(QAudioDeviceInfo::defaultOutputDevice().preferredFormat())),
      currentTimestamp(-1),
      fmt(QAudioDeviceInfo::defaultOutputDevice().preferredFormat()),
      audioOutput(nullptr),
      info(QAudioDeviceInfo::defaultOutputDevice()),
      volume(1.0) {
  audioOutput = new QAudioOutput(info, fmt);

  connect(audioOutput, &QAudioOutput::stateChanged, this, &AudioPlayer::handleStateChanged);
  dev = audioOutput->start();
  delay = (mtime_t)fmt.durationForBytes(audioOutput->bufferSize());

  worker = new std::thread(run, this);
}

AudioPlayer::~AudioPlayer() {
  // QAudioOutput (CoreAudioOutput) on Mac often crashes when deleted on a background thread
  audioOutput->deleteLater();

  {
    std::unique_lock<std::mutex> lk(mu);
    exit = true;
  }
  worker->join();
  delete worker;
}

void AudioPlayer::pushAudio(VideoStitch::Audio::Samples& audioSamples) {
  if (dev != nullptr && audioSamples.getNbOfSamples() > 0) {
    if (audioOutput->state() != QAudio::SuspendedState) {
      std::unique_lock<std::mutex> lk(mu);
      while (!audioQueue.empty() && audioQueue.front().getTimestamp() > audioSamples.getTimestamp()) {
        // must have seeked just before
        audioQueue.pop();
      }
      audioQueue.push(audioSamples.clone());
    }
  }
}

std::string AudioPlayer::getName() const { return "playback"; }
void AudioPlayer::render(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface>, mtime_t ts) { currentTimestamp = ts; }
void AudioPlayer::renderCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t ts) {
  currentTimestamp = ts;
}
void AudioPlayer::renderEquiangularCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface>, mtime_t ts) {
  currentTimestamp = ts;
}

// play the audio as fast as possible
void AudioPlayer::run(AudioPlayer* that) {
  for (;;) {
    VideoStitch::Audio::Samples as;
    {
      std::unique_lock<std::mutex> lock(that->mu);
      if (that->exit) {
        return;
      }
      if (that->audioQueue.empty() ||
          size_t(that->audioOutput->bytesFree() / that->fmt.channelCount() / (that->fmt.sampleSize() / 8)) <
              that->audioQueue.front().getNbOfSamples()) {
        continue;
      }
      if (that->audioQueue.front().getTimestamp() - that->delay > that->currentTimestamp) {
        continue;  // audio in advance
      }
      as = std::move(that->audioQueue.front());
      that->audioQueue.pop();
    }
    qint64 toWrite = that->fmt.bytesForFrames((qint32)as.getNbOfSamples());
    qint64 written = 0;
    while (toWrite > 0) {
      QByteArray b((const char*)as.getSamples()[0] + written, toWrite);
      qint64 actuallyWritten = that->dev->write(b);
      written += actuallyWritten;
      toWrite -= actuallyWritten;
      if (actuallyWritten == 0) {
        // cannot write in the queue
        break;
      }
    }
  }
}

void AudioPlayer::handleStateChanged(QAudio::State) {
  if (audioOutput->error() != QAudio::NoError) {
    logError(audioOutput->error());
  }
}

void AudioPlayer::logError(QAudio::Error err) {
  switch (err) {
    case QAudio::OpenError:
      ERROR(tag) << "An error opening the audio device, cannot play audio." << std::endl;
      break;
    case QAudio::IOError:
      ERROR(tag) << "An error occurred during read/write of audio device, cannot play audio." << std::endl;
      break;
    case QAudio::UnderrunError: {
      static auto underrunLogLevel = VideoStitch::Logger::Warning;
      VideoStitch::Logger::get(underrunLogLevel, tag)
          << "Audio data is not being fed to the audio device at a fast enough rate, cannot play audio." << std::endl;
      underrunLogLevel = VideoStitch::Logger::Verbose;
    } break;
    case QAudio::FatalError:
    default:
      ERROR(tag) << "A non-recoverable error has occurred, the audio device is not usable at this time." << std::endl;
      break;
  }
}

void AudioPlayer::onActivatePlayBack(bool b) {
  std::lock_guard<std::mutex> lk(mu);
  if (b) {
    audioOutput->resume();
  } else {
    audioOutput->suspend();
  }
}
