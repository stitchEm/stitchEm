// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "bufferedReader.hpp"
#include "audio/asrc.hpp"
#include "audio/audioPipeline.hpp"
#include "audio/audioPreProcessor.hpp"
#include "gpu/hostBuffer.hpp"
#include "input/inputFrame.hpp"

#include "libvideostitch/controller.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/panoDef.hpp"

#include <list>
#include <mutex>

namespace VideoStitch {
class ThreadPool;

namespace Core {

/**
 * @brief kAudioPreRoll: maximum audio pre-roll set to 250 ms
 */
static const mtime_t kAudioPreRoll = 250000;

typedef std::vector<std::unique_ptr<Audio::AudioAsyncReader>> audioReaderVector_t;
typedef std::map<groupid_t, audioReaderVector_t> audioReaderGroupMap_t;

/**
 * @brief ReaderController
 */

class ReaderController {
 public:
  /**
   * Factory function to create a Controller with the @pano PanoDefinition and the @readerFactory input ReaderFactory.
   * @param pano The panorama to stitch.
   * @param readerFactory The factory for reader. We take ownership.
   */
  static Potential<ReaderController> create(const PanoDefinition& pano, const AudioPipeDefinition& audioPipeDef,
                                            Input::ReaderFactory* readerFactory, unsigned preloadCacheSize = 2);

  ~ReaderController();

  const PanoDefinition& getPano() const { return *pano; }

  void resetPano(const PanoDefinition& newPano);
  void resetAudioPipe(const AudioPipeDefinition& newAudioPipeDef);

  int getCurrentFrame() const;

  std::tuple<Input::ReadStatus, Input::ReadStatus, Input::ReadStatus> load(
      mtime_t&, std::map<readerid_t, Input::PotentialFrame>& frames,
      std::list<Audio::audioBlockGroupMap_t>& audioBlocks, Input::MetadataChunk& imu_metadata);
  mtime_t reload(std::map<readerid_t, Input::PotentialFrame>& frames);
  void releaseBuffer(std::map<readerid_t, Input::PotentialFrame>& frames);

  /**
   * Reader accessor. This is internal. In particular, the current thread posess a stitcher instance for this to work,
   * and inputs must be locked.
   * @returns The i-th reader.
   * @note Inputs must be locked by the calling thread.
   */
  // TODO only make bufferedReader available
  std::vector<Input::VideoReader*> getReaders() const {
    std::vector<Input::VideoReader*> delegates;
    for (const auto& bufReader : videoReaders) {
      delegates.push_back(bufReader->getDelegate().get());
    }
    return delegates;
  }

  /**
   * Get a video reader
   * @param id id in PanoDefinition
   * @returns the i'th input video reader
   */
  std::shared_ptr<Input::VideoReader> getReader(readerid_t i) const {
    for (auto& reader : videoReaders) {
      if (reader->getDelegate()->id == i) {
        return reader->getDelegate();
      }
    }
    assert(false);
    return videoReaders[0]->getDelegate();  // silence warnings
  }

  /**
   * Get a video reader spec
   * @param id id in PanoDefinition
   * @returns the i'th input video reader
   */
  const Input::VideoReader::Spec& getReaderSpec(readerid_t i) const {
    for (const auto& reader : videoReaders) {
      if (reader->getDelegate()->id == i) {
        return reader->getSpec();
      }
    }
    assert(false);
    return videoReaders[0]->getSpec();  // silence warnings
  }

  /**
   * Audio accessors.
   */
  bool hasAudio() const;
  Audio::SamplingRate getAudioSamplingRate() const;
  Audio::SamplingDepth getAudioSamplingDepth() const;

  /**
   * Returns the initial time offset. This is the base time for counting output frames.
   */
  frameid_t getInitialFrameOffset() const { return initialFrameOffset; }

  Status seekFrame(frameid_t date);

  frameid_t getFirstReadableFrame() const;
  frameid_t getLastReadableFrame() const;
  frameid_t getLastStitchableFrame() const;
  std::vector<frameid_t> getLastFrames() const;

  FrameRate getFrameRate() const;
  mtime_t getLatency() const;
  Status addSink(const Ptv::Value* config);
  void removeSink();

  Status setupReaders();

  void cleanReaders();

  /**
   * @fn apply the audio preprocessor
   * @param inOutMap Map of audio samples corresponding to the group id gr.
   * @param gr Group id of the group to process.
   */
  void applyAudioPreProc(std::map<readerid_t, Audio::Samples>& inOutMap, groupid_t gr);

  /**
   * @fn setup an audio preprocessor
   * @param name Name of the audio preprocessor to setup.
   * @param gr Group id on which the audio preprocessor needs to be applied.
   */
  Status setupAudioPreProc(const std::string& name, groupid_t gr);

 private:
  Input::ReadStatus loadVideo(mtime_t& date, std::map<readerid_t, Input::PotentialFrame>& frames);
  Input::ReadStatus loadAudio(std::list<Audio::audioBlockGroupMap_t>& audioIn, groupid_t gr);
  Input::ReadStatus loadMetadata(Input::MetadataChunk& Measure);

  /**
   * Helper function to know if at least one audio group needs to be synchronized with the video
   * @return true if one of the audio group needs to be sync
   */
  bool needAudioVideoResync();

  /**
   * Create a controller. Call init() after creation.
   * @param pano The panorama to stitch.
   * @param readers Vector of the intput readers.
   * @param audioReaders audio-only input reader
   * @param frameOffset Initial frame offset.
   * @param frameRate project frame rate
   */
  ReaderController(const PanoDefinition&, const AudioPipeDefinition&,
                   std::vector<std::unique_ptr<BufferedReader>> videoReaders, audioReaderVector_t audioReaders,
                   std::vector<std::shared_ptr<Input::MetadataReader>> metadataReaders,
                   std::vector<std::shared_ptr<Input::SinkReader>> sinkReaders, frameid_t frameOffset, const FrameRate);

  mtime_t getCommonReaderDate(std::vector<mtime_t> dates);

  PanoDefinition* pano;
  AudioPipeDefinition* audioPipeDef;

  const frameid_t initialFrameOffset;

  std::vector<std::unique_ptr<BufferedReader>> videoReaders;

  // audio reader memory is manually managed to
  // avoid double deletion of audio-video readers
  audioReaderGroupMap_t audioAsyncReaders;
  std::vector<std::shared_ptr<Input::MetadataReader>> metadataReaders;
  std::vector<std::shared_ptr<Input::SinkReader>> sinkReaders;

  std::map<groupid_t, bool> audioAndVideoInSameGroup;

  FrameRate frameRate;
  mtime_t videoFrameLen;

  // Loading the frames for the stitchers
  std::map<groupid_t, bool> audioVideoResync;
  std::map<groupid_t, mtime_t> audioTimestampOffsets;
  std::mutex inputMutex;

  std::atomic<mtime_t> videoTimeStamp;
  std::map<groupid_t, mtime_t> audioTimestampsPerGroup;
  std::map<groupid_t, std::unique_ptr<Audio::AudioPreProcessor>> audioPreProcs;
};
}  // namespace Core
}  // namespace VideoStitch
