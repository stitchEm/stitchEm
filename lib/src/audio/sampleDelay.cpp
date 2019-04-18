// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sampleDelay.hpp"

#include <cmath>
#include <cassert>

namespace VideoStitch {
namespace Audio {

// --- Constructor ------------------------------

SampleDelay::SampleDelay() : AudioObject("delay", AudioFunction::PROCESSOR), curDelayTime_(0) {}

// --- Helpers ----------------------------------

namespace {
inline size_t convertSecondsToSamples(double seconds) {
  return static_cast<size_t>(seconds * getDefaultSamplingRate());
}
inline double convertSamplesToSeconds(size_t samples) {
  return static_cast<double>(samples) / getDefaultSamplingRate();
}
}  // namespace

// --- Set / get --------------------------------

Status SampleDelay::setDelaySeconds(double delayInSeconds) {
  std::lock_guard<std::mutex> lock(delayMutex_);
  curDelayTime_ = convertSecondsToSamples(delayInSeconds);
  return Status::OK();
}

double SampleDelay::getDelaySeconds() {
  std::lock_guard<std::mutex> lock(delayMutex_);
  return convertSamplesToSeconds(curDelayTime_);
}

void SampleDelay::setDelaySamples(size_t delayInSamples) {
  std::lock_guard<std::mutex> lock(delayMutex_);
  curDelayTime_ = delayInSamples;
}

size_t SampleDelay::getDelaySamples() {
  std::lock_guard<std::mutex> lock(delayMutex_);
  return curDelayTime_;
}

// --- Processing -------------------------------

namespace {
void applyWindow(const std::vector<audioSample_t>& inFadeOut, const std::vector<audioSample_t>& inFadeIn,
                 std::vector<audioSample_t>* winOut) {
  size_t numPoints = inFadeOut.size() * 2;
  std::vector<audioSample_t> win;
  for (size_t i = 0; i < numPoints; i++) {
    // Generate Hann window (https://www.youtube.com/watch?v=oHg5SJYRHA0)
    audioSample_t point = 0.5 - (0.5 * cos((2.0 * M_PI * (double)i) / ((double)numPoints - 1)));
    win.push_back(point);
  }
  size_t offset = (numPoints / 2) - 1;
  for (size_t i = 0; i < inFadeIn.size(); i++) {
    audioSample_t sample = (inFadeIn.data()[i] * win.data()[i]) + (inFadeOut.data()[i] * win.data()[i + offset]);
    winOut->push_back(sample);
  }
}

size_t getDiff(size_t delayTime, size_t oldDelayTime, size_t inputSize) {
  size_t diff = std::abs((int)delayTime - (int)oldDelayTime);

  // Below we do  xxEnd = CURRENT_DELAY_BUFFER.end() - inputSize - oldDelayTime + diff;
  // and          xxEnd = CURRENT_DELAY_BUFFER.end() - inputSize - delayTime + diff;
  // so the following checks are necessary to not go beyond the end of CURRENT_DELAY_BUFFER.
  // Also, the window can't be bigger that the block size.

  if (diff > (int)inputSize - oldDelayTime) {
    diff = inputSize - oldDelayTime;
  }
  if (diff > (int)inputSize - delayTime) {
    diff = inputSize - delayTime;
  }
  if (diff > inputSize) {
    diff = inputSize;
  }

  return diff;
}
}  // namespace

/// \fn void SampleDelay::step(AudioBlock& in, AudioBlock& out)
/// \brief A sample delay audio processing block
/// \param in The audio input
/// \param out Delayed output
///
/// If the delay value changes, we apply Hann windowing to avoid clicks
///
///   input        :                 +=======+
///   delay buffer : +-- // ---------+=======+
///   old delay    :                   *******     oldDelayTime
///   old output   :          +=======+
///   old window   :          |--|                 inFadeOut
///   new delay    :                      ****     delayTime
///   new output   :             +=======+
///   new window   :             |--|              inFadeIn
///   output       :             +<>+====+         winOut + samples from delay buffer
///   difference   :       -->|  |<--              abs(delayTime - oldDelayTime)
///
///
/// If the delay doesn't change, we simply copy
///
///   input        :              +=======+
///   delay buffer : +-- // ------+=======+
///   output       :         +=======+
///   delay time   :                  *****

#define CURRENT_DELAY_BUFFER delayBuffers_[track.channel()]

void SampleDelay::step(AudioBlock& out, const AudioBlock& in) {
  for (auto& track : in) {
    const size_t inputSize = track.size();

    // Insert new samples into delay buffer
    for (auto& s : track) {
      CURRENT_DELAY_BUFFER.push_back(s);
    }
    // TODO DEBUG this with lucas
    //    CURRENT_DELAY_BUFFER.insert(CURRENT_DELAY_BUFFER.begin(), track.begin(), track.end());

    size_t delayTime;
    //    size_t oldDelayTime;
    {
      std::lock_guard<std::mutex> lock(delayMutex_);
      delayTime = curDelayTime_;
    }
    // TO DEBUG WITH LUCAS
    //    {
    //      std::lock_guard<std::mutex> lock(delayMutex_);
    //      // The maximum delay we can apply right now
    //      size_t possibleDelay = CURRENT_DELAY_BUFFER.size() - inputSize;
    //      // The delay we will apply : the minimum of requested and possible
    //      delayTime = (possibleDelay < curDelayTime_) ? possibleDelay : curDelayTime_;
    //      // This loop we'll check the previous delay time against the current
    //      // delay time to see if we need windowing. We'll save the delay time
    //      // we're using this loop to check the next time around.
    //      oldDelayTime = oldDelayTime_;
    //      oldDelayTime_ = delayTime;
    //    }

    // If the delay has changed, apply windowing
    //    if (delayTime != oldDelayTime) {
    //      std::cout << "windowing when lucas will come back" << std::endl;

    //      size_t diff = getDiff(delayTime, oldDelayTime, inputSize);

    //      // inFadeOut = from : delayBuf.end() - in.size() - oldDelay
    //      //             to   : delayBuf.end() - in.size() - oldDelay + delayDiff
    //      std::vector<audioSample_t> inFadeOut;
    //      auto fadeOutStart = CURRENT_DELAY_BUFFER.end() - inputSize - oldDelayTime;
    //      auto fadeOutEnd   = CURRENT_DELAY_BUFFER.end() - inputSize - oldDelayTime + diff;
    //      inFadeOut.insert(inFadeOut.begin(), fadeOutStart, fadeOutEnd);

    //      // inFadeIn = from : delayBuf.end() - in.size() - delay
    //      //            to   : delayBuf.end() - in.size() - delay + delayDiff
    //      std::vector<audioSample_t> inFadeIn;
    //      auto fadeInStart = CURRENT_DELAY_BUFFER.end() - inputSize - delayTime;
    //      auto fadeInEnd   = CURRENT_DELAY_BUFFER.end() - inputSize - delayTime + diff;
    //      inFadeIn.insert(inFadeIn.begin(), fadeInStart, fadeInEnd);

    //      std::vector<audioSample_t> winOut;
    //      applyWindow(inFadeOut, inFadeIn, &winOut);

    //      out[track.channel()].assign(winOut.begin(), winOut.end());

    //      // fill = from : delayBuf.end() - in.size() - delay + delayDiff
    //      //        to   : delayBuf.end() - delay
    //      if (winOut.size() < inputSize) {
    //        auto fillStart = CURRENT_DELAY_BUFFER.end() - inputSize - delayTime + diff;
    //        auto fillEnd   = CURRENT_DELAY_BUFFER.end() - delayTime;
    //        copy(fillStart, fillEnd, back_inserter(out[track.channel()]));
    //      }
    //    } else {
    //      auto copyStart = CURRENT_DELAY_BUFFER.end() - delayTime - inputSize;
    //      auto copyEnd   = CURRENT_DELAY_BUFFER.end() - delayTime;
    //      out[track.channel()].assign(copyStart, copyEnd);
    //    }

    if ((inputSize + delayTime) > CURRENT_DELAY_BUFFER.size()) {
      // Not enough samples to fill a buffer : zero padding
      out[track.channel()].assign(inputSize, 0);
    } else {
      auto copyStart = CURRENT_DELAY_BUFFER.end() - delayTime - inputSize;
      auto copyEnd = CURRENT_DELAY_BUFFER.end() - delayTime;
      out[track.channel()].assign(copyStart, copyEnd);
    }

    // Remove oldest samples if we're up to kMaxTime
    size_t maxSamples = static_cast<size_t>(getDefaultSamplingRate() * kMaxDelayTime);
    if (CURRENT_DELAY_BUFFER.size() > maxSamples) {
      size_t numSamplesToErase = CURRENT_DELAY_BUFFER.size() - maxSamples;
      CURRENT_DELAY_BUFFER.erase(CURRENT_DELAY_BUFFER.begin(), CURRENT_DELAY_BUFFER.begin() + numSamplesToErase);
    }
  }
}

#undef CURRENT_DELAY_BUFFER

void SampleDelay::step(AudioBlock& buf) { step(buf, buf); }

}  // namespace Audio
}  // namespace VideoStitch
