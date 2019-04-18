// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "orahAudioSync.hpp"

#include "util/fft.h"
#include "libvideostitch/logging.hpp"

#include <iostream>
#include <cassert>
#include <cmath>
#include <complex>
#include <algorithm>
#include <thread>
#include <cstdlib>

// The number of blanks injected is hard coded in Orah firmware.
#ifndef ORAH_SYNC_NUM_BLANKS
#error "[OrahAudioSync] ORAH_SYNC_NUM_BLANKS must be defined"
#endif
static_assert(ORAH_SYNC_NUM_BLANKS > 0, "[OrahAudioSync] ORAH_SYNC_NUM_BLANKS must be > 0");

namespace VideoStitch {
namespace Audio {
namespace Orah {

#define BLOCK_SIZE_DURATION (mtime_t(double(blockSize_) / double(sampleRate_) * 1000000))
#define _LOG Logger::get(Logger::Info)

// Constants
//
// Find offset
static const float kSyncTimeout{6};  // XXX TBD (Note: has effect on test)
static const float kBlankSearchTimeout{5};
static const int kMinZerosInBlank{60};
static const int kNumBlanks{ORAH_SYNC_NUM_BLANKS};
// Click remover
static const size_t kClickDelay{110};      // 55 samples, 2 channels
static const size_t kClickEraseSize{100};  // 50 samples, 2 channels
static const int kTail{10};                // 5 samples, 2 channels
static_assert(kClickDelay > kClickEraseSize + kTail / 2,
              "[OrahAudioSync] Click delay too small, or tail or erase size too big");
// Fractional delay
static const int kFadeSize{50};  // Number of blocks for longer crossfades
static const int kMaxDelay{2};   // XXX TBD (Seconds per channel)
// Helper
static const int kOther[2]{1, 0};
#define late_ kOther[early_]
static const std::string o2bSyncTag = "OrahAudioSync";

OrahAudioSync::OrahAudioSync(const BlockSize blockSize, groupid_t gr)
    : AudioPreProcessor(getOrahAudioSyncName(), gr),
      sampleRate_{-1},
      blockSize_{(size_t)getIntFromBlockSize(blockSize)},
      syncState_{SyncState::UNSYNCHRONIZED},
      prevState_{SyncState::UNSYNCHRONIZED},
      zeroCount_({{0, 0}})  // This looks dumb, but is needed for MSVC2013.
      ,
      blankCount_({{0, 0}})  // [https://stackoverflow.com/questions/19877757]
      ,
      early_{-1},
      syncTimeoutCounter_{0},
      offsetCounter_{0},
      channelOffset_({{{{0}}, {{0}}}})  // This also looks dumb. See above.
      ,
      offset_{0},
      disableClickRemoval_{false},
      fadeDelay_{0},
      prevDelay_{0},
      fade_{0},
      lastTimestamps_({{0, 0}}) {
  size_t wbSize = (blockSize_ * 2) + kClickDelay;
  // Seed work buffer with something other than zero so we
  // don't inadvertently detect a blank where none exists.
  std::vector<orahSample_t> v(wbSize, 1);
  workBuffer_.push_back(v);
  workBuffer_.push_back(v);
}

OrahAudioSync::~OrahAudioSync() {}

float OrahAudioSync::getProcessingDelay() {
  std::lock_guard<std::mutex> lk(delayMtx_);
  return (kClickDelay / 2) + offset_;
}

/* === Buffer Assignment ===

  Input is two stereo streams, passed as a two-element vector of
  Samples, with each block of Samples 2 * blockSize_ samples
  long. These streams are interleaved stereo.
  +----------------------       --+
  |             <in[0]>           |
  +---+---+---+---+---+--  ...  --+
  | L | R | L | R | L | R         |
  +---+---+---+---+---+--       --+

  The exact same formatting, as well as the following descriptions,
  hold true for stream 2 (in[1]).

  These are copied to their respective places in the working buffers.
  +---------------------------------------+
  |                Stream 1               |
  +---------------------------------------+
  |             <workBuffer_[0]>          |
  +--------------+------------------------+
  | <clickDelay> |         <in[0]>        |
  +--------------+------------------------+

  This buffer is passed to the search function, which looks for blanks.
  If one is found at position pos, we will set samples from
  [pos - ClickEraseSize + tail/2] to [pos + tail/2] to zero, so we
  make sure we always have enough samples on either side of pos
  (hence the padding from clickDelay and tail).
  +--------------------------------------------------------+
  |                        Stream 1                        |
  +-----------------------+--------------------------------+
  |      <clickDelay>     |             <in[0]>            |
  +----------------+------+-------------------------+------+
  |                |      (Blank search here)       | tail |
  +----------------+--------------------------------+------+
                   |
                   | <-- Start zero search here

  The last clickDelaySize samples from both streams in the workBuffer_
  are copied to the beginning of each stream to serve as the clickDelay
  for the next input.

*/

void OrahAudioSync::process(const std::vector<Samples>& in, std::vector<Samples>& out) {
  // <in> is 2 audio streams (one from each camera board) with
  // 2 * blockSize_ samples in each stream (interleaved stereo).
  assert(in.size() == 2);

  // Set sample rate first time, and assert if it changes later on.
  // Also make sure the two blocks of samples have the same sampling
  // rate. This is just a sanity check since the rate is only used to
  // calculate timeouts and has no real effect on the processing.
  // We also check that we're getting 16-bit interleaved audio. This
  // does have an impact since our work buffer is of type int16_t
  // (orahSample_t) and the samples must be interleaved to correctly
  // locate the blanks.
  assert(in[0].getSamplingRate() == in[1].getSamplingRate());
  if (sampleRate_ == -1) {
    sampleRate_ = getIntFromSamplingRate(in[0].getSamplingRate());
  } else {
    assert(sampleRate_ == getIntFromSamplingRate(in[0].getSamplingRate()));
  }
  assert(in[0].getSamplingDepth() == SamplingDepth::INT16);
  assert(in[1].getSamplingDepth() == SamplingDepth::INT16);

#define BLOCK_SIZE_DURATION_AND_TOLERANCE (BLOCK_SIZE_DURATION + 5000)
  if (in[0].getTimestamp() - lastTimestamps_[0] > BLOCK_SIZE_DURATION_AND_TOLERANCE) {
    Logger::warning(o2bSyncTag) << "Reader 0: " << (in[0].getTimestamp() - lastTimestamps_[0]) / 1000
                                << " ms are missing" << std::endl;
  }
  if (in[1].getTimestamp() - lastTimestamps_[1] > BLOCK_SIZE_DURATION_AND_TOLERANCE) {
    Logger::warning(o2bSyncTag) << "Reader 1: " << (in[1].getTimestamp() - lastTimestamps_[1]) / 1000
                                << " ms are missing" << std::endl;
  }
  lastTimestamps_[0] = in[0].getTimestamp();
  lastTimestamps_[1] = in[1].getTimestamp();
  // Get pointers to samples from each stream, and copy them into the
  // working buffer after the click delay samples. Also place incoming
  // audio into the delay buffer.
  orahSample_t* s[2];
  for (int i = 0; i < 2; i++) {
    assert(in[i].getNbOfSamples() == blockSize_);
    s[i] = ((orahSample_t**)in[i].getSamples().data())[0];
    std::copy(s[i], s[i] + blockSize_ * 2, workBuffer_[i].begin() + kClickDelay);
  }

  // Continuously run samples through the offset detector.
  findOffset();

  std::lock_guard<std::mutex> lk(delayMtx_);

  // Add samples to delay buffer
  for (int i = 0; i < 2; i++) {
    delayBuffer_[i].insert(delayBuffer_[i].end(), workBuffer_[i].end() - (2 * blockSize_), workBuffer_[i].end());
  }

  // Apply any needed delay and assign output
  fracDelay(out);

  // Save last kClickDelay samples at the beginning of the working buffer
  // for next time through (see diagram above). Erase front of delay buffer
  // if it has accumulated enough audio.
  for (int i = 0; i < 2; i++) {
    assert(kClickDelay <= workBuffer_[i].size());
    std::copy(workBuffer_[i].end() - kClickDelay, workBuffer_[i].end(), workBuffer_[i].begin());
    if ((int)delayBuffer_[i].size() > kMaxDelay * sampleRate_ * 2) {
      delayBuffer_[i].erase(delayBuffer_[i].begin(), delayBuffer_[i].begin() + blockSize_ * 2);
    }
  }
}

void OrahAudioSync::findOffset() {
  // Pointer to search start position (see description above).
  const orahSample_t* s[2];
  s[0] = &workBuffer_[0][kClickDelay - kTail];
  s[1] = &workBuffer_[1][kClickDelay - kTail];

  for (size_t sample = 0; sample < blockSize_ * 2; sample++) {
    switch (syncState_) {
      case SyncState::UNSYNCHRONIZED:
        syncTimeoutCounter_++;
        if (syncTimeoutCounter_ > sampleRate_ * kSyncTimeout * 2) {
          std::stringstream mesg;
          mesg << "[OrahAudioSync] UNSYNCHRONIZED: "
               << "Timeout looking for start of sync" << std::endl;
          _LOG << mesg.str();
          syncTimeoutCounter_ = 0;
          std::unique_lock<std::mutex> blk(blockMtx_);
          std::thread(xcorrSync, this).detach();
          cv_.wait(blk);
          syncState_ = SyncState::SYNCHRONIZED;
        }
        // no break

      case SyncState::SYNCHRONIZED:  // and UNSYNCHRONIZED...
        // Look for at least kMinZerosInBlank consecutive zero-values
        // samples (i.e. a blank).
        for (int channel = 0; channel < 2; channel++) {
          if (0 == s[channel][sample]) {
            zeroCount_[channel]++;
          } else {
            if (zeroCount_[channel] > kMinZerosInBlank) {
              std::stringstream mesg;
              mesg << "[OrahAudioSync] (UN)SYNCHRONIZED:  Found blank " << blankCount_[channel] << " in channel "
                   << channel << " at sample " << sample << std::endl;
              _LOG << mesg.str();
              removeClick((int)sample, channel);
              blankCount_[channel] = 1;  // We found a blank in this channel
              early_ = channel;
              prevState_ = syncState_;
              syncState_ = SyncState::SYNCHRONIZING;
            }
            zeroCount_[channel] = 0;
          }
        }
        break;

      case SyncState::SYNCHRONIZING:
        offsetCounter_++;
        for (int channel = 0; channel < 2; channel++) {
          if (blankCount_[channel] < kNumBlanks) {
            if (0 == s[channel][sample]) {
              zeroCount_[channel]++;
            } else {
              if (zeroCount_[channel] > kMinZerosInBlank) {
                std::stringstream mesg;
                mesg << "[OrahAudioSync] SYNCHRONIZING: "
                     << "Found blank " << blankCount_[channel] << " in channel " << channel << " at sample " << sample
                     << std::endl;
                _LOG << mesg.str();
                channelOffset_[channel][blankCount_[channel]] = offsetCounter_;
                removeClick((int)sample, channel);
                blankCount_[channel]++;
              }
              zeroCount_[channel] = 0;
            }
          }
        }
        if (blankCount_[0] >= kNumBlanks && blankCount_[1] >= kNumBlanks) {
          std::stringstream mesg;
          mesg << "[OrahAudioSync] SYNCHRONIZING: Done" << std::endl;
          _LOG << mesg.str();
          // We've found all the blanks
          calcOffset();
          blankCount_.fill(0);
          zeroCount_.fill(0);
          channelOffset_[0].fill(0);
          channelOffset_[1].fill(0);
          offsetCounter_ = 0;
          syncState_ = SyncState::SYNCHRONIZED;
        } else if (offsetCounter_ == sampleRate_ * kBlankSearchTimeout) {
          std::stringstream mesg;
          mesg << "[OrahAudioSync] SYNCHRONIZING: "
               << "Timeout searching for matching blanks" << std::endl;
          _LOG << mesg.str();
          blankCount_.fill(0);
          zeroCount_.fill(0);
          channelOffset_[0].fill(0);
          channelOffset_[1].fill(0);
          offsetCounter_ = 0;
          syncState_ = prevState_;
        }
        break;

    }  // switch (syncState_)
  }    // for (sample < blockSize_)
}

namespace {
// FFT size must be 2^N for Ooura FFT. Find the nearest
// power of two for the amount of audio we have in the
// delay buffer to use for FFT. We need to then double the
// size of our buffer since we're using circular cross-
// correlation.
// See [ https://www.youtube.com/watch?v=dQw4w9WgXcQ ]
int getFftSize(int bufSize) {
  assert(bufSize > 0x400);  // Set some minimum size
  int fftSize = 0x400;
  if (bufSize > 0x80000) {
    fftSize = 0x80000;  // Set a maximum limit (~0.5 million samples)
  } else {
    while (fftSize < bufSize) {
      fftSize <<= 1;
    }
    if (fftSize - bufSize < bufSize - (fftSize >> 1)) {
      fftSize <<= 1;
    }
  }
  return fftSize;
}
}  // namespace

void OrahAudioSync::xcorrSync(OrahAudioSync* that) {
  std::unique_lock<std::mutex> blk(that->blockMtx_);
  that->cv_.notify_all();

  std::stringstream mesg;
  mesg << "[OrahAudioSync] xcorrSync(): Starting cross-correlation sync" << std::endl;
  _LOG << mesg.str();

  int fftSize;
  std::vector<std::vector<double>> fftBuf;

  {
    std::lock_guard<std::mutex> lk(that->delayMtx_);  // For delayBuffer_
    int bufSize = (int)that->delayBuffer_[0].size() / 2;
    fftSize = getFftSize(bufSize);

    // Allocate the FFT buffer. Ooura rdft() requires a buffer
    // of N_FFT doubles. We have four channels (two stereo) to
    // process.
    std::vector<double> v(fftSize, 0);  // Initialize FFT buffers to 0
    for (int i = 0; i < 4; i++) {
      fftBuf.push_back(v);
    }

    // Copy audio from delay buffers to fftBuffers. We also need to
    // deinterleave the samples and convert them to doubles before FFT.
    int numSamples = std::min(fftSize / 2, bufSize);
    for (int i = 0, j = 0; i < numSamples; i++) {
      fftBuf[0][i] = (double)that->delayBuffer_[0][j];
      fftBuf[2][i] = (double)that->delayBuffer_[1][j++];
      fftBuf[1][i] = (double)that->delayBuffer_[0][j];
      fftBuf[3][i] = (double)that->delayBuffer_[1][j++];
    }
  }

  // Perform FFT, and alias output to a complex pointer for easier
  // manipulation in following step.
  std::complex<double>* cpxFftData[4];
  for (int i = 0; i < 4; i++) {
    rdft(fftSize, 1, fftBuf[i].data());
    cpxFftData[i] = reinterpret_cast<std::complex<double>*>(fftBuf[i].data());
    // Set DC and Nyquist to 0 (see fftsg_h.c).
    cpxFftData[i][0] = {0, 0};
  }

  // Cross correlate all pairs in different streams (i.e. [0 2], [0 3],
  // [1 2], and [1 3]).
  int pairs[4][2] = {{0, 2}, {0, 3}, {1, 2}, {1, 3}};
  std::vector<std::vector<double>> xcorrResults;
#define CPX_VAL(_x) cpxFftData[pairs[pair][_x]][i]  // Access helper
  for (int pair = 0; pair < 4; pair++) {
    std::vector<double> slot;
    for (int i = 0; i < fftSize / 2; i++) {
      std::complex<double> val = CPX_VAL(0) * std::conj(CPX_VAL(1));
      slot.push_back(real(val));
      slot.push_back(imag(val));
    }
    xcorrResults.push_back(slot);
  }

  // Move results back to time domain. We don't care about the absolute level
  // of the cross-correlation samples, so we don't need to scale the IFFT
  // output as we would if this were audio. We also find the maximum value in
  // in the output buffer (the strongest correlation) and add this to our
  // results.
  int result = 0;
  for (int i = 0; i < 4; i++) {
    int resultIdx;
    rdft(fftSize, -1, xcorrResults[i].data());
    resultIdx =
        (int)std::distance(xcorrResults[i].begin(), std::max_element(xcorrResults[i].begin(), xcorrResults[i].end()));
    if (resultIdx > (fftSize / 2)) {
      result += resultIdx - fftSize;
    } else {
      result += resultIdx;
    }
  }

  // The offset is the average of the four cross-correlations.
  std::lock_guard<std::mutex> lk(that->delayMtx_);  // For offset_
  that->offset_ = (float)result / 4.0f;
  if (that->offset_ < 0) {
    that->early_ = 0;
    that->offset_ = -that->offset_;
  } else {
    that->early_ = 1;
  }

  mesg << "[OrahAudioSync] xcorrSync(): Found " << that->offset_ << " (early: " << that->early_ << ")" << std::endl;
  _LOG << mesg.str();
}

// Interpolate from sample before blank to a sample after blank with
// a half cosine curve.
void OrahAudioSync::removeClick(const int pos, const int channel) {
  if (disableClickRemoval_) {
    return;
  }

  assert(pos >= 0);
  size_t offset = pos + kClickDelay - kTail / 2 - kClickEraseSize;
  std::vector<orahSample_t> intCurve(kClickDelay);
  // We need to treat the L and R channels separately. Since they
  // are interleaved, do two passes, stepping by 2 through the
  // samples.
  for (int n = 0; n < 2; n++) {
    float startVal = (float)workBuffer_[channel][offset - 2 + n];
    float endVal = (float)workBuffer_[channel][offset + kClickDelay + n];
    float range = std::abs(startVal - endVal) / 2;
    float dc = range + std::min(startVal, endVal);
    // See if we're going upwards (i.e. [pi .. 2*pi)).
    if (startVal < endVal) {
      range = -range;
    }
    for (int i = 0; i < int(kClickDelay); i += 2) {
      float sample = range * std::cos(float(M_PI) * float(i) / float(kClickDelay - 1)) + dc;
      // Add a bit of randomness to avoid pure tones
      if (i % 10 == 0) {
        sample += 200.0f * (0.5f - float(std::rand()) / float(RAND_MAX));
      }
      intCurve[i + n] = orahSample_t(sample);
    }
  }
  std::copy(intCurve.begin(), intCurve.end(), workBuffer_[channel].begin() + offset);
}

void OrahAudioSync::calcOffset() {
  float meanOffset = 0;
  for (int i = 0; i < kNumBlanks; i++) {
    std::stringstream mesg;
    mesg << "[OrahAudioSync] calcOffset(): "
         << "Late: " << channelOffset_[late_][i] << "    Early: " << channelOffset_[early_][i]
         << "    Diff: " << (channelOffset_[late_][i] - channelOffset_[early_][i]) << std::endl;
    _LOG << mesg.str();
    meanOffset += (float)(channelOffset_[late_][i] - channelOffset_[early_][i]);
  }
  meanOffset /= kNumBlanks;
  std::lock_guard<std::mutex> lk(delayMtx_);  // For offset_
  offset_ = std::round(meanOffset) / 2.0f;
  std::stringstream mesg;
  mesg << "[OrahAudioSync] calcOffset(): Found " << offset_ << " (early: " << early_ << ")" << std::endl;
  _LOG << mesg.str();
}

void OrahAudioSync::fracDelay(std::vector<Samples>& out) {
  // Make sure we have something to do
  if (early_ == -1 || (offset_ == 0 && fade_ == 0)) {
    // Just copy to output
    for (int i = 0; i < 2; i++) {
      std::copy(workBuffer_[i].begin(), workBuffer_[i].begin() + (blockSize_ * 2),
                ((orahSample_t**)out[i].getSamples().data())[0]);
    }
    return;
  }
  assert(early_ == 0 || early_ == 1);

  // Copy "late" audio directly to output since its not delayed
  std::copy(workBuffer_[late_].begin(), workBuffer_[late_].begin() + (blockSize_ * 2),
            ((orahSample_t**)out[late_].getSamples().data())[0]);

  // When we calculate the fractional delay we need two additional samples
  // at the end of the delayBuffer. So we limit the delay to the size of delayBuffer - 2.
  // See VSA-6861 for more information
  float delay = std::min(std::ceil(offset_ * 2.0f), float(delayBuffer_[early_].size() - 2));
  float frac = offset_ - std::floor(offset_);  // Fractional part

  // If delay changes, cross fade to avoid clicks
  if (delay != prevDelay_) {
    if (size_t(std::ceil(std::abs(delay - prevDelay_))) > blockSize_ * 2) {
      fade_ = kFadeSize;
    } else {
      fade_ = int(kFadeSize / 4);
    }
    fadeDelay_ = prevDelay_;
    prevDelay_ = delay;
  }

  // Disable cross-fading if we're not using click removal
  if (disableClickRemoval_) {
    fade_ = 0;
  }

  std::vector<orahSample_t> o;
  orahSample_t* sPtr;
  // We don't need an intermediate buffer if we're not cross fading
  if (fade_ == 0) {
    sPtr = ((orahSample_t**)out[early_].getSamples().data())[0];
  } else {
    o.resize(blockSize_ * 2u);
    sPtr = o.data();
  }

  if (frac == 0) {
    // Simple delay
    auto dBegin = size_t(delay) + (blockSize_ * 2u) + kClickDelay;
    auto dEnd = size_t(delay) + kClickDelay;
    dBegin = std::min(delayBuffer_[early_].size(), dBegin);
    dEnd = std::min(delayBuffer_[early_].size(), dEnd);
    std::copy(delayBuffer_[early_].end() - dBegin, delayBuffer_[early_].end() - dEnd, sPtr);
  } else {
    // Fractional delay
    int dBegin = int(std::floor(delay)) + int(blockSize_ * 2u);
    dBegin = int(delayBuffer_[early_].size()) - dBegin - int(kClickDelay);
    int dEnd = dBegin + int(blockSize_ * 2);
    dEnd = std::min(int(delayBuffer_[early_].size()), dEnd);
    dBegin = std::max(dBegin, 0);
    for (int n = dBegin; n < dEnd; n++) {
      // out = S[n] + ((S[n+1] - S[n]) * frac)
      // Since we have interleaved samples, S[n+1] is delayBuffer_[][n+2].
      *sPtr++ = delayBuffer_[early_][n] +
                orahSample_t(std::round(float(delayBuffer_[early_][n + 2] - delayBuffer_[early_][n]) * frac));
    }
  }

  if (fade_ > 0) {
    float alpha = float(fade_) / float(kFadeSize);
    int dBegin = int(fadeDelay_) + int(blockSize_ * 2u) + int(kClickDelay);
    dBegin = int(delayBuffer_[early_].size()) - dBegin;
    dBegin = std::max(dBegin, 0);
    orahSample_t* oPtr = ((orahSample_t**)out[early_].getSamples().data())[0];
    for (int i = 0; i < int(blockSize_ * 2); i++) {
      *oPtr++ = orahSample_t((float(o[i]) * (1.0f - alpha)) + (alpha * delayBuffer_[early_][i + dBegin]));
    }
    fade_--;
  }
}

}  // namespace Orah
}  // namespace Audio
}  // namespace VideoStitch
