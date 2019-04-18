// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/audio.hpp"
#include "libvideostitch/circularBuffer.hpp"
#include "audio/audioPreProcessor.hpp"

#include <cstdint>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <array>

// The number of blanks injected is hard coded in Orah firmware.
#define ORAH_SYNC_NUM_BLANKS 5

namespace VideoStitch {
namespace Audio {
namespace Orah {

typedef int16_t orahSample_t;

static inline std::string getOrahAudioSyncName() { return "OrahAudioSync"; }

class OrahAudioSync : public AudioPreProcessor {
 public:
  explicit OrahAudioSync(const BlockSize blockSize, groupid_t gr = 0);
  ~OrahAudioSync();

  /** @fn void process()
   *
   * @param in  [in]  The two audio streams to be aligned.
   * @param out [out] Output from the alignment function.
   *
   * In must be a vector of two `Audio::Samples`. `out` must
   * be allocated to the same size as `in`.
   */
  void process(const std::vector<Samples>& in, std::vector<Samples>& out);

  /** @fn void process() in place method
   *
   * @param inOut  [inOut]  The two audio streams to be aligned.
   *
   * In must be a vector of two `Audio::Samples`.
   */
  void process(std::vector<Samples>& inOut) { process(inOut, inOut); }

  /** @fn float getOffset()
   *
   * @return The detected offset, in samples.
   */
  float getOffset() {
    std::lock_guard<std::mutex> dlk(delayMtx_);
    return offset_;
  }

  /** @fn float getOffsetBlocking()
   *
   * @return The detected offset, in samples.
   *
   * Will block while xcorrSync() is running. This is mainly
   * intended to be used by the tests to allow the xcorr to
   * finish before checking that we got the expected result,
   * as the build type can greatly affect the time required
   * by the FFT / IFFT.
   */
  float getOffsetBlocking() {
    std::lock_guard<std::mutex> blk(blockMtx_);
    std::lock_guard<std::mutex> dlk(delayMtx_);
    return offset_;
  }

  /** @fn void getProcessingDelay()
   *
   * @return The induced latency in samples, per channel.
   */
  float getProcessingDelay();

  /** @fn void diableClickSuppresion()
   *
   * @param disable [in] Disable click remover
   *                       true: disabled
   *                      false: enabled
   *
   * Mainly for testing purposes. Also causes fading between
   * delay value changes to be turned off.
   */
  void diableClickSuppresion(const bool disable) { disableClickRemoval_ = disable; }

 private:
  enum class SyncState { UNSYNCHRONIZED, SYNCHRONIZED, SYNCHRONIZING };

  std::vector<std::vector<orahSample_t>> workBuffer_;
  int sampleRate_;

  // Find offset (blank search)
  size_t blockSize_;
  SyncState syncState_;
  SyncState prevState_;
  std::array<int, 2> zeroCount_;
  std::array<int, 2> blankCount_;
  int early_;
  size_t syncTimeoutCounter_;
  size_t offsetCounter_;
  std::array<std::array<size_t, ORAH_SYNC_NUM_BLANKS>, 2> channelOffset_;
  float offset_;
  void findOffset();

  // Cross-correlation
  static void xcorrSync(OrahAudioSync* that);

  // Click remover
  bool disableClickRemoval_;
  void removeClick(const int pos, const int channel);

  // Fractional delay
  std::mutex blockMtx_;
  std::condition_variable cv_;
  std::mutex delayMtx_;
  std::deque<orahSample_t> delayBuffer_[2];
  float fadeDelay_;
  float prevDelay_;
  int fade_;
  void fracDelay(std::vector<Samples>& out);

  // Helpers
  void calcOffset();

  // Timestamps
  std::array<mtime_t, 2> lastTimestamps_;
};

}  // namespace Orah
}  // namespace Audio
}  // namespace VideoStitch
