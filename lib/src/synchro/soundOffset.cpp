// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "soundOffset.hpp"

#include "util/registeredAlgo.hpp"
#include "util/fft.h"
#include "audio/filter.hpp"
#include "audio/converter.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>
#include <stdint.h>
#include <complex>
#include <climits>

namespace VideoStitch {
namespace Synchro {

static const uint32_t MAX_FFT_SIZE = 16777216;  // 2^23 samples and padding, or over 2 minutes at 48kHz

// Limit sample reading to what can be processed in FFT
// with a little error margin
static const mtime_t MAX_AUDIO_SYNC_SEQUENCE{
    FrameRate::MILLION * (mtime_t)((double)(MAX_FFT_SIZE / 2 / Audio::getDefaultSamplingRate()) * 0.8)};

enum SyncProgress {
  ReaderSetupProgress = 20,
  ReadingSamplesProgress = 50,
  FindingOffsetsProgress = 30,
};

namespace {

Util::RegisteredAlgo<SoundOffsetAlignAlgorithm> registeredSnd("sound_offset_align");

}

const char* SoundOffsetAlignAlgorithm::docString =
    "An algorithm that computes frame offsets using audio to align the inputs in time.\n"
    "Can be applied pre-calibration.\n"
    "The result is a { \"frames\": list of integer offsets (all >=0, in frames) }\n"
    "which can be used directly as a 'frame_offset' parameter for the 'inputs'.\n";

/* FFT cross correlation
 *
 * Uses Ooura SG_H FFT implementation
 * See fft.readme
 *
 */
Status SoundOffsetAlignAlgorithm::foffs_sgh(std::vector<Audio::AudioBlock>& in, int fs, audioSyncResult_t& res,
                                            ProgressReporter* progress) {
  /* Use the smallest audio buffer as max length */
  size_t len = in[0][Audio::MONO].size();
  for (size_t i = 1; i < in.size(); ++i) {
    if (in[i][Audio::MONO].size() < len) {
      len = in[i][Audio::MONO].size();
    }
  }

  /* Input to Ooura FFT must be 2^N, and twice as big as the number of samples */
  uint32_t size = 512;
  for (;;) {
    if ((uint32_t)(len * 2) < size) {
      break;
    } else if (size > MAX_FFT_SIZE) {
      /* TODO: Split into a few different blocks if there are too many samples? */
      // Will try to sync with MAX_FFT_SIZE...
      size = MAX_FFT_SIZE;
      len = MAX_FFT_SIZE / 2;
      break;
    } else {
      size *= 2;
    }
  }

  std::unique_ptr<std::complex<double>[]> cpxData(new std::complex<double>[size]);
  std::unique_ptr<std::complex<double>[]> cpxStore(new std::complex<double>[size]);

  double* const data = reinterpret_cast<double*>(&cpxData[0]);
  double* const store = reinterpret_cast<double*>(&cpxStore[0]);

  for (int i = 0; i < res.nSources; i++) {
    std::vector<double> timeRow;
    std::vector<double> corrRow;

    memcpy(data, in[i][Audio::MONO].data(), len * sizeof(double));
    memset(data + len, 0, (2 * size - len) * sizeof(double));

    rdft(size, 1, data);
    memcpy(store, data, (size_t)(size * 2 * sizeof(double)));

    for (int j = 0; j < res.nSources; j++) {
      /* Don't auto-correlate */
      if (j == i) {
        timeRow.push_back(0);
        corrRow.push_back(0);
        continue;
      }

      /* If we did A*B, B*A will just be the negative for
       * time, and the same for correlation score
       */
      if (j < i) {
        timeRow.push_back(-res.timeOffset[j].data()[i]);
        corrRow.push_back(res.corrVal[j].data()[i]);
        continue;
      }

      memcpy(data, in[j][Audio::MONO].data(), len * sizeof(double));
      memset(data + len, 0, (2 * size - len) * sizeof(double));

      rdft(size, 1, data);

      /* Don't use DC and Nyquist components
       * Ooura FFT stores the real Nyquist component in the DC imaginary slot.
       */
      cpxData[0] = 0;
      cpxStore[0] = 0;

      for (size_t k = 1; k < (size / 2); k++) {
        cpxData[k] = cpxStore[k] * conj(cpxData[k]);
      }
      rdft(size, -1, data);

      double max = 0;
      size_t idx = 0;
      for (size_t k = 0; k < size; k++) {
        double sample = data[k];
        if (sample > max) {
          max = sample;
          idx = k;
        }
      }

      corrRow.push_back(max / (size / 2));

      if (idx > (size / 2)) {
        timeRow.push_back(((double)idx - (double)size) * (1.0 / fs));
      } else {
        timeRow.push_back((double)idx * (1.0 / fs));
      }

      double prog =
          FindingOffsetsProgress * ((double)((i * res.nSources) + j + 1) / (double)(res.nSources * res.nSources));
      if (progress && progress->notify("Finding offsets", ReaderSetupProgress + ReadingSamplesProgress + prog)) {
        return {Origin::SynchronizationAlgorithm, ErrType::OperationAbortedByUser, "Sound offset algorithm cancelled"};
      }
    }

    res.timeOffset.push_back(timeRow);
    res.corrVal.push_back(corrRow);
  }

  return Status::OK();
}

//#define AUDIO_SYNC_SHOW_OUTPUT

static void calcFrameOffsets(const audioSyncResult_t& res, std::vector<int>& frames, const double frameLen,
                             const std::vector<mtime_t>& initialOffset) {
#if defined(AUDIO_SYNC_SHOW_OUTPUT)
  printf("\n");

  for (int i = 0; i < res.nSources; i++) {
    printf("File %2d: ", i);
    for (int j = 0; j < res.nSources; j++) {
      printf("%10.3f ", res.timeOffset[i].data()[j]);
    }
    printf("\n         ");
    for (int j = 0; j < res.nSources; j++) {
      printf("%10.3f ", res.corrVal[i].data()[j]);
    }
    printf("\n\n");
  }
#endif

  // Spanning tree for best relative time offsets
  //
  // From http://www.geeksforgeeks.org/greedy-algorithms-set-5-prims-minimum-spanning-tree-mst-2
  std::vector<int> parent(res.nSources, 0);
  {
    std::vector<double> key(res.nSources, 0.0);
    std::vector<bool> set(res.nSources, false);

    parent.data()[0] = -1;

    for (int count = 0; count < res.nSources - 1; count++) {
      double max = -1;
      int maxIndex = 0;

      for (int i = 0; i < res.nSources; i++) {
        if ((set[i] == false) && (key.data()[i] > max)) {
          max = key.data()[i];
          maxIndex = i;
        }
      }

      set[maxIndex] = true;

      for (int i = 0; i < res.nSources; i++) {
        if ((res.corrVal[maxIndex].data()[i] != 0) && (set[i] == false) &&
            (res.corrVal[maxIndex].data()[i] > key.data()[i])) {
          parent.data()[i] = maxIndex;
          key.data()[i] = res.corrVal[maxIndex].data()[i];
        }
      }
    }

#if defined(AUDIO_SYNC_SHOW_OUTPUT)
    printf("\n");
    for (int i = 1; i < res.nSources; i++) {
      printf("%d --> %d : Score %f\n", i, parent.data()[i], key.data()[i]);
    }
    printf("\n");
#endif
  }

  // Get absolute time offsets
  std::vector<double> time(res.nSources, 0.0);
  {
    std::vector<bool> set(res.nSources, false);

    set[0] = true;

    double minTime = HUGE_VAL;
    for (int i = 0; i < res.nSources; i++) {
      for (int j = 0; j < res.nSources; j++) {
        if (set[j] == false && set[parent.data()[j]] == true) {
          set[j] = true;
          time.data()[j] = time.data()[parent.data()[j]] + res.timeOffset[parent.data()[j]].data()[j];
          if (minTime > time.data()[j]) {
            minTime = time.data()[j];
          }
          i = 0;
          break;
        }
      }
    }

    double maxTime = 0;
    if (minTime < 0) {
      for (int i = 0; i < res.nSources; i++) {
        time.data()[i] += (-1.0 * minTime);
        if (time.data()[i] > maxTime) {
          maxTime = time.data()[i];
        }
      }
    }

    for (int i = 0; i < res.nSources; i++) {
      time.data()[i] = maxTime - time.data()[i];
    }
  }

  // Add existing offset if there was one
  double maxOffset = 0;
  std::vector<double> additionalOffsets;
  for (int i = 0; i < res.nSources; i++) {
    if (maxOffset < ((double)initialOffset.data()[i] / 1000000.0)) {
      maxOffset = (double)initialOffset.data()[i] / 1000000.0;
    }
  }
  for (int i = 0; i < res.nSources; i++) {
    additionalOffsets.push_back(maxOffset - ((double)initialOffset.data()[i] / 1000000.0));
  }
  std::vector<double> outTimes;
  maxOffset = 0;
  for (int i = 0; i < res.nSources; i++) {
    if (maxOffset < (additionalOffsets.data()[i] - time.data()[i])) {
      maxOffset = additionalOffsets.data()[i] - time.data()[i];
    }
    outTimes.push_back(additionalOffsets.data()[i] - time.data()[i]);
  }
  for (int i = 0; i < res.nSources; i++) {
    outTimes.data()[i] = maxOffset - outTimes.data()[i];
  }

  // Convert to frames and place in output
  for (int i = 0; i < res.nSources; i++) {
#if defined(AUDIO_SYNC_SHOW_OUTPUT)
    std::cout << "Source " << i << " Offset: " << (int)rint(outTimes.data()[i] / frameLen) << std::endl;
#endif
    frames.push_back((int)rint(outTimes.data()[i] / frameLen));
  }
}

SoundOffsetAlignAlgorithm::SoundOffsetAlignAlgorithm(const Ptv::Value* config) : firstFrame(0), lastFrame(1) {
  if (config != NULL) {
    const Ptv::Value* value = config->has("first_frame");
    if (value && (value->getType() == Ptv::Value::INT)) {
      firstFrame = value->asInt();
    }

    value = config->has("last_frame");
    if (value && (value->getType() == Ptv::Value::INT)) {
      lastFrame = value->asInt();
    }
  }
}

Potential<Ptv::Value> SoundOffsetAlignAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                                       Util::OpaquePtr**) const {
  FAIL_RETURN(GPU::useDefaultBackendDevice());

  std::vector<int> offsetsFrames;

  FAIL_RETURN(doAlign(*pano, offsetsFrames, progress));

  if (offsetsFrames.size() != (size_t)pano->numInputs()) {
    return Status{Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Input size mismatch"};
  }

  for (readerid_t i = 0; i < (readerid_t)offsetsFrames.size(); ++i) {
    pano->getInput(i).setFrameOffset(offsetsFrames[i]);
  }

  return Potential<Ptv::Value>(Status::OK());
}

Status SoundOffsetAlignAlgorithm::setupReaders(const Core::PanoDefinition& pano, ProgressReporter* progress,
                                               FrameRate& frameRate, Audio::SamplingRate& sampleRate,
                                               std::vector<std::unique_ptr<Input::Reader>>& readers) const {
  Input::DefaultReaderFactory readerFactory((int)firstFrame, (int)lastFrame);

  for (readerid_t i = 0; i < pano.numInputs(); ++i) {
    if (progress &&
        progress->notify("Preparing to read audio tracks", (double)i / pano.numInputs() * ReaderSetupProgress)) {
      return {Origin::SynchronizationAlgorithm, ErrType::OperationAbortedByUser, "Sound offset algorithm cancelled"};
    }

    const Core::InputDefinition* im = &pano.getInput(i);
    Potential<Input::Reader> potReader = readerFactory.create(i, *im);
    Input::AudioReader* audioReader;
    Input::VideoReader* videoReader;
    if (potReader.ok()) {
      Input::Reader* r = potReader.object();
      audioReader = r->getAudioReader();
      if (!audioReader) {
        return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration,
                "Sound alignment requires that all inputs have an audio track. Unable to read audio for input " +
                    std::to_string(i) + "."};
      }
      videoReader = r->getVideoReader();
      if (!videoReader) {
        return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration,
                "Sound alignment requires that all inputs have a video track. Unable to read video for input " +
                    std::to_string(i) + "."};
      }
      readers.emplace_back(std::unique_ptr<Input::Reader>(potReader.release()));
    } else {
      continue;
    }

    if (i == 0) {
      frameRate = videoReader->getSpec().frameRate;
      assert(frameRate.den != 0);
      assert(frameRate.num != 0);
      sampleRate = audioReader->getSpec().sampleRate;
    } else {
      /* Videos must have the same sample rate and frame rate */
      if (audioReader->getSpec().sampleRate != sampleRate) {
        return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration,
                "Sound alignment requires that all audio inputs have the same sample rate"};
      }
      if (videoReader->getSpec().frameRate != frameRate) {
        return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration,
                "Sound alignment requires that all video inputs have the same frame rate"};
      }
    }
  }
  return Status::OK();
}

Status SoundOffsetAlignAlgorithm::readSamples(const std::vector<std::unique_ptr<Input::Reader>>& readers,
                                              const FrameRate& frameRate, ProgressReporter* progress,
                                              std::vector<mtime_t>& initialPos,
                                              std::vector<Audio::AudioBlock>& samplesToRead) const {
  unsigned inputID = 0;

  frameid_t numVideoFramesToRead = (frameid_t)lastFrame - (frameid_t)firstFrame;
  mtime_t sequenceLengthToRead = std::min(frameRate.frameToTimestamp(numVideoFramesToRead), MAX_AUDIO_SYNC_SEQUENCE);

  for (auto& reader : readers) {
    Input::AudioReader* audioReader = reader->getAudioReader();
    if (!audioReader) {
      return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Encountered invalid audio reader"};
    }

    mtime_t maxPos = 0;
    Audio::AudioTrack snd(Audio::SPEAKER_FRONT_LEFT);
    Audio::AudioBlock tmp(Audio::MONO);
    Audio::Samples currentSamples;
    size_t nbSamples = 0;

    while (audioReader->readSamples(1024, currentSamples).ok()) {
      if (currentSamples.getNbOfSamples() == 0) {
        break;
      }
      if (nbSamples == 0) {
        maxPos = currentSamples.getTimestamp() + sequenceLengthToRead;
        initialPos.push_back(currentSamples.getTimestamp());
      } else {
        // reading might take considerable time per input
        // update the progress bar (and enably cancelling) continuously while reading samples
        const auto progressStepPerInput = static_cast<double>(ReadingSamplesProgress) / readers.size();
        const auto sampleLength = maxPos - initialPos[inputID];
        if (sampleLength > 0) {
          const auto currentSampleProgress =
              static_cast<double>(currentSamples.getTimestamp() - initialPos[inputID]) / sampleLength;
          if (progress && progress->notify("Reading audio samples from input " + std::to_string(inputID + 1),
                                           ReaderSetupProgress + inputID * progressStepPerInput +
                                               progressStepPerInput * currentSampleProgress)) {
            return {Origin::SynchronizationAlgorithm, ErrType::OperationAbortedByUser,
                    "Sound offset algorithm cancelled"};
          }
        }
      }

      /* Check user bounds */
      if (currentSamples.getTimestamp() >= maxPos) {
        break;
      }

      nbSamples += currentSamples.getNbOfSamples();
      Audio::convertSamplesToMonoDouble(currentSamples, snd,
                                        Audio::getNbChannelsFromChannelLayout(audioReader->getSpec().layout),
                                        audioReader->getSpec().sampleDepth);
    }

    if (nbSamples < 256) {
      std::stringstream msg;
      msg << "Sound alignment requires at least 256 samples of audio data. Only received " << nbSamples
          << " samples in input " << inputID;
      return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration, msg.str()};
    }
    tmp[Audio::MONO] = std::move(snd);
    samplesToRead.push_back(std::move(tmp));
    inputID++;
  }
  return Status::OK();
}

Status SoundOffsetAlignAlgorithm::doAlign(const Core::PanoDefinition& pano, std::vector<int>& frames,
                                          ProgressReporter* progress) const {
  if (pano.numInputs() < 2) {
    return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration,
            "Sound alignment requires at least 2 inputs"};
  }

  FrameRate frameRate;
  Audio::SamplingRate sampleRate{Audio::SamplingRate::SR_NONE};

  std::vector<std::unique_ptr<Input::Reader>> readers;
  FAIL_RETURN(setupReaders(pano, progress, frameRate, sampleRate, readers));

  if (readers.empty()) {
    return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration,
            "Unable to read inputs for sound alignment"};
  }

  std::vector<Audio::AudioBlock> in;
  std::vector<mtime_t> initialPos;
  FAIL_RETURN(readSamples(readers, frameRate, progress, initialPos, in));

  int fs = Audio::getIntFromSamplingRate(sampleRate);

  /* Filter input */
  Audio::IIR filt(fs);
  filt.setFilterTFGQ(Audio::FilterType::HIGH_PASS, 200.0, 0, 1);
  for (auto&& i : in) {
    filt.step(i);
  }

  filt.setFilterTFGQ(Audio::FilterType::HIGH_PASS, 100.0, 0, 1);
  for (auto&& i : in) {
    filt.step(i);
  }

  /* Cross-correlate and find max value */
  audioSyncResult_t res;
  res.nSources = (int)pano.numInputs();
  FAIL_RETURN(foffs_sgh(in, fs, res, progress));

  /* Calculate frame offsets */
  double fLen = (double)frameRate.den / (double)frameRate.num;
  calcFrameOffsets(res, frames, fLen, initialPos);

  return Status::OK();
}

}  // namespace Synchro
}  // namespace VideoStitch
