// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "readerController.hpp"

#include "common/container.hpp"
#include "common/thread.hpp"
#include "audio/asrc.hpp"
#include "audio/orah/orahAudioSync.hpp"

#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/logging.hpp"

#include <future>
#include <vector>
#include <unordered_set>
#include <cstdlib>

static std::string CTRLtag("Reader Controller");

namespace VideoStitch {
namespace Core {

ReaderController::ReaderController(const PanoDefinition& pano, const AudioPipeDefinition& audioPipe,
                                   std::vector<std::unique_ptr<BufferedReader>> video, audioReaderVector_t audio,
                                   std::vector<std::shared_ptr<Input::MetadataReader>> meta,
                                   std::vector<std::shared_ptr<Input::SinkReader>> sink, frameid_t frameOffset,
                                   const FrameRate frameRate)
    : pano(pano.clone()),
      audioPipeDef(audioPipe.clone()),
      initialFrameOffset(frameOffset),
      videoReaders(std::move(video)),
      metadataReaders(meta),
      sinkReaders(sink),
      frameRate(frameRate),
      videoTimeStamp(0) {
  // Set up clock groups for audio.
  videoFrameLen = (frameRate.den * 1000000) / frameRate.num;

  for (auto& audioReader : audio) {
    groupid_t audioGrId = getPano().getInput(audioReader->getId()).getGroup();
    audioTimestampOffsets[audioGrId] = 0;
    audioAsyncReaders[audioGrId].push_back(std::move(audioReader));
    audioTimestampsPerGroup[audioGrId] = -1;
    bool inSameGroup = false;
    for (auto& videoReader : videoReaders) {
      groupid_t videoGrId = getPano().getInput(videoReader->getDelegate()->id).getGroup();
      if (audioGrId == videoGrId) {
        inSameGroup = true;
      }
    }
    audioVideoResync[audioGrId] = !inSameGroup;  // ask for resync if audio and video are not in same group
    audioAndVideoInSameGroup[audioGrId] = inSameGroup;
  }
}

PotentialValue<FrameRate> findCommonFrameRate(const std::vector<std::unique_ptr<BufferedReader>>& readers) {
  struct FrameRateHasher {
    std::size_t operator()(const FrameRate& val) const { return 1024 * val.num / val.den; }
  };

  std::unordered_set<FrameRate, FrameRateHasher> frameRateSet, proceduralFrameRateSet;
  for (auto& reader : readers) {
    if (reader->getSpec().frameRateIsProcedural) {
      // keep track of procedural framerates
      if (reader->getSpec().frameRate.den) {
        proceduralFrameRateSet.insert(reader->getSpec().frameRate);
      }
    } else {
      // keep track of valid video framerates, discarding the ones from procedural readers
      if (reader->getSpec().frameRate.num > 0 && reader->getSpec().frameRate.den) {
        frameRateSet.insert(reader->getSpec().frameRate);
      }
    }
  }

  // check that we have at most one valid video framerate
  if (frameRateSet.size() > 1) {
    for (auto frameRate : frameRateSet) {
      Logger::error(CTRLtag) << "Found frame rate " << frameRate.num << '/' << frameRate.den << std::endl;
    }
    return Status{Origin::Input, ErrType::InvalidConfiguration,
                  "Not all inputs have the same frame rate. This generally indicates an incorrect input."};
  } else {
    // set the framerate, if we got one
    if (frameRateSet.size() == 1) {
      return *frameRateSet.begin();
    }
    // else if we got a single procedural framerate, use it
    else if (proceduralFrameRateSet.size() == 1) {
      return *proceduralFrameRateSet.begin();
    }
  }

  // No frame rate defined by readers, go with default
  FrameRate defaultFrameRate(VIDEO_WRITER_DEFAULT_FRAMERATE_NUM, VIDEO_WRITER_DEFAULT_FRAMERATE_DEN);
  return defaultFrameRate;
}

void destroyMetadataReader(Input::MetadataReader*& metadataReader) {
  if (metadataReader == nullptr) {
    return;
  }
  if (!metadataReader->getVideoReader()) {
    if (metadataReader->getAudioReader()) {
      Logger::error(CTRLtag)
          << "destroyMetadataReader: cannot have audio + metadata without having video (not supported)" << std::endl;
      assert(false);
      return;
    }

    // all video readers will be destroyed separately (including video-metadata readers)
    // only destroy metadata-only readers here
    delete metadataReader;
    metadataReader = nullptr;
  }
}

Potential<ReaderController> ReaderController::create(const PanoDefinition& pano, const AudioPipeDefinition& audioPipe,
                                                     Input::ReaderFactory* readerFactory, unsigned preloadCacheSize) {
  std::unique_ptr<Input::ReaderFactory> readerFactoryDeleter(readerFactory);

  std::vector<std::unique_ptr<BufferedReader>> videoReaders;
  std::vector<std::unique_ptr<Audio::AudioAsyncReader>> audioAsyncReaders;
  std::vector<std::shared_ptr<Input::MetadataReader>> metadataReaders;
  std::vector<std::shared_ptr<Input::SinkReader>> sinkReaders;

  // Instantiate all readers through the factory
  for (readerid_t imId = 0; imId < pano.numInputs(); ++imId) {
    std::shared_ptr<Input::Reader> reader;
    {
      auto potentialReader = readerFactory->create(imId, pano.getInput(imId));
      if (!potentialReader.ok()) {
        Logger::error(CTRLtag) << "cannot create reader for input " << imId << std::endl;
        return potentialReader.status();
      }
      reader = std::shared_ptr<Input::Reader>(potentialReader.release());
    }

    // Promote the reader to video
    std::shared_ptr<Input::VideoReader> videoReader = Input::getVideoSharedReader(reader);
    if (videoReader) {
      auto bufferedReader = BufferedReader::create(videoReader, preloadCacheSize);
      if (!bufferedReader.ok()) {
        return bufferedReader.status();
      }
      videoReaders.emplace_back(bufferedReader.release());
    }

    // Promote the reader to audio
    std::shared_ptr<Input::AudioReader> audioReader = Input::getAudioSharedReader(reader);
    if (audioReader) {
      auto audioAsyncReader =
          Audio::AudioAsyncReader::create(audioReader, Audio::getBlockSizeFromInt(audioPipe.getBlockSize()),
                                          Audio::getSamplingRateFromInt(audioPipe.getSamplingRate()));
      audioAsyncReaders.emplace_back(audioAsyncReader);
    }

    std::shared_ptr<Input::MetadataReader> currentMetadataReader = Input::getMetadataSharedReader(reader);
    if (currentMetadataReader) {
      metadataReaders.push_back(currentMetadataReader);
    }
    std::shared_ptr<Input::SinkReader> currentSinkReader = Input::getSinkSharedReader(reader);
    if (currentSinkReader) {
      sinkReaders.push_back(currentSinkReader);
    }
  }

  auto potFrameRate = findCommonFrameRate(videoReaders);
  FAIL_RETURN(potFrameRate.status());
  FrameRate frameRate = potFrameRate.value();

  return new ReaderController(pano, audioPipe, std::move(videoReaders), std::move(audioAsyncReaders),
                              std::move(metadataReaders), std::move(sinkReaders), readerFactory->getFirstFrame(),
                              frameRate);
}

ReaderController::~ReaderController() {
  delete pano;
  delete audioPipeDef;
}

// ------------------------ Controller implementation ------------------

std::tuple<Input::ReadStatus, Input::ReadStatus, Input::ReadStatus> ReaderController::load(
    mtime_t& date, std::map<readerid_t, Input::PotentialFrame>& frames,
    std::vector<Audio::audioBlockGroupMap_t>& audioBlocks, Input::MetadataChunk& metadata) {
  // protect from concurrent seeks
  std::lock_guard<std::mutex> lock(inputMutex);

  Input::ReadStatus audioRet = Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>();
  Input::ReadStatus videoRet = loadVideo(date, frames);
  if (!videoRet.ok()) {
    // we lost the video clock, request synchronization only for audio and video not in the same group
    for (auto& kv : audioVideoResync) {
      if (!audioAndVideoInSameGroup.at(kv.first)) {
        Logger::verbose(CTRLtag) << "lost video clock resynch audio group " << kv.first << std::endl;
        kv.second = true;
      }
    }
  }

  // we can load audio independently from video, but only when resync is not needed
  if (videoRet.getCode() != Input::ReadStatusCode::EndOfFile || !needAudioVideoResync()) {
    for (auto& kv : audioAsyncReaders) {
      groupid_t grId = kv.first;
      audioRet = loadAudio(audioBlocks, grId);
      if (!audioRet.ok() && !audioAndVideoInSameGroup.at(grId) &&
          audioRet.getCode() != Input::ReadStatusCode::TryAgain) {
        // we lost the audio clock, request synchronization only for audio and video not in the same group
        // No need to resynch if the group couldn't feed more data
        Logger::warning(CTRLtag) << "lost audio clock resynch audio group " << grId << std::endl;
        audioVideoResync.at(grId) = true;
      }
    }
  }

  if (!audioBlocks.empty()) {
    audioRet = Input::ReadStatus::OK();
  }

  auto metaRet = loadMetadata(metadata);

  return std::make_tuple(videoRet, audioRet, metaRet);
}

mtime_t ReaderController::getLatency() const {
  mtime_t maxLatency = 0;
  for (auto& reader : videoReaders) {
    int latency = (int)reader->getDelegate()->getLatency();
    if (latency > maxLatency) maxLatency = latency;
  }
  return maxLatency;
}

Status ReaderController::addSink(const Ptv::Value* config) {
  Status sinkStatus = Status();
  mtime_t lastVideoTimeStamp = videoTimeStamp;
  for (auto& reader : sinkReaders) {
    mtime_t audioTimeStamp = lastVideoTimeStamp;
    if (reader->getAudioReader()) {
      audioTimeStamp += audioTimestampOffsets.at(getPano().getInput(reader->id).getGroup());
    }

    Status tmpStatus = reader->addSink(
        config, lastVideoTimeStamp + getFrameRate().frameToTimestamp(getPano().getInput(reader->id).getFrameOffset()),
        audioTimeStamp);
    if (!tmpStatus.ok()) {
      sinkStatus = tmpStatus;
    }
  }
  return sinkStatus;
}

void ReaderController::removeSink() {
  for (auto& reader : sinkReaders) {
    reader->removeSink();
  }
}

Input::ReadStatus ReaderController::loadVideo(mtime_t& date, std::map<readerid_t, Input::PotentialFrame>& frames) {
  std::map<readerid_t, InputFrame> videoIn;
  std::map<readerid_t, Input::PotentialFrame> videoFrames;

  // gather timings from the readers' groups
  // readers belonging to a same group have a synchronized clock
  std::map<int, std::vector<std::pair<BufferedReader*, mtime_t>>> timingsPerGroup;

  Input::ReadStatus readerStatus;
  for (auto& reader : videoReaders) {
    videoIn[reader->getDelegate()->id] = reader->load();

    readerStatus = (readerStatus.ok() ? videoIn[reader->getDelegate()->id].readerStatus : readerStatus);

    if (!reader->getSpec().frameRateIsProcedural && getPano().getInput(reader->getDelegate()->id).getGroup() != -1 &&
        readerStatus.ok()) {
      std::vector<std::pair<BufferedReader*, mtime_t>>& groupTimes =
          timingsPerGroup[getPano().getInput(reader->getDelegate()->id).getGroup()];
      groupTimes.push_back(std::make_pair(
          reader.get(),
          videoIn[reader->getDelegate()->id].date -
              getFrameRate().frameToTimestamp(getPano().getInput(reader->getDelegate()->id).getFrameOffset())));
    }
  }

  if (!readerStatus.ok()) {
    for (auto& r : videoReaders) {
      r->releaseBuffer(videoIn[r->getDelegate()->id].buffer);
    }
    return readerStatus;
  }

  // only need synchronization when the inputs are not images
  if (frameRate.num > 0) {
    // drop frames if a reader is late compared to the others inside its group
    struct CompareSecond {
      bool operator()(const std::pair<BufferedReader*, mtime_t>& left,
                      const std::pair<BufferedReader*, mtime_t>& right) const {
        return left.second < right.second;
      }
    };
    for (auto groupTimings : timingsPerGroup) {
      mtime_t latest =
          std::max_element(groupTimings.second.begin(), groupTimings.second.end(), CompareSecond())->second;
      for (auto timing : groupTimings.second) {
        BufferedReader* reader = timing.first;
        mtime_t readerLatestFrame = timing.second;
        mtime_t readerFrame = timing.second;
        if (videoIn[reader->getDelegate()->id].date >= 0) {
          int dropped = 0;
          // Drop frames on that stream until we are within the half frame period.
          // To prevent looping forever, if this would require forwarding more than 3 seconds, we give up and let the
          // next loadVideo continue the sync
          while (readerLatestFrame <= latest - (500000 * frameRate.den) / frameRate.num &&
                 dropped < 3 * frameRate.num / frameRate.den) {
            dropped++;
            reader->releaseBuffer(videoIn[reader->getDelegate()->id].buffer);
            videoIn[reader->getDelegate()->id] = reader->load();
            readerStatus = videoIn[reader->getDelegate()->id].readerStatus;
            if (!readerStatus.ok()) {
              for (auto& r : videoReaders) {
                r->releaseBuffer(videoIn[r->getDelegate()->id].buffer);
              }
              return readerStatus;
            }
            readerLatestFrame =
                videoIn[reader->getDelegate()->id].date -
                getFrameRate().frameToTimestamp(getPano().getInput(reader->getDelegate()->id).getFrameOffset());
          }
          if (dropped != 0) {
            if (videoTimeStamp == 0) {
              /* reset latency as we are dropping frames decoded before synchronization */
              reader->getDelegate()->setLatency(0);
            }
            Logger::info(CTRLtag) << "fast forwarding input " << reader->getDelegate()->id << " from "
                                  << readerFrame / 1000 << " to " << readerLatestFrame / 1000 << " (dropped " << dropped
                                  << ")" << std::endl;
          }
        }
      }
    }
  }

  // Video timestamps management
  std::vector<mtime_t> dates;
  for (auto read : videoIn) {
    dates.push_back(read.second.date);
    videoFrames.insert(std::map<readerid_t, Input::PotentialFrame>::value_type(
        read.first, {read.second.readerStatus, read.second.buffer}));
  }
  mtime_t lastVideoTimeStamp = videoTimeStamp;
  videoTimeStamp = date = getCommonReaderDate(dates);
  if (date - lastVideoTimeStamp < 0) {
    // output has been seeked backward, need to reset audioTimeStamp
    for (auto& audioTimeStamp : audioTimestampsPerGroup) {
      audioTimeStamp.second = -1;
    }
  }

  assert(videoTimeStamp >= 0);
  Logger::verbose(CTRLtag) << "read a frame at timestamp " << videoTimeStamp << std::endl;

  frames = videoFrames;
  return Input::ReadStatus::OK();
}

/// \fn Status ReaderController::loadAudio(std::vector<std::map<readerid_t, Audio::AudioBlock>>& audioBlocks)
/// \param audioBlocks First dimension is the block, second index the input
///                    (e.g. audioIn[1][0] => second block of the input 0)
/// \return Code::Ok on success, else an error code
Input::ReadStatus ReaderController::loadAudio(std::vector<Audio::audioBlockGroupMap_t>& audioBlocks, groupid_t gr) {
  std::vector<std::map<readerid_t, Audio::Samples>> audioIn;

  size_t nbSamples = audioPipeDef->getBlockSize();

  // Read audio by blocks of size `nbSamples` until we're about to pass the next video frame in the timeline.
  mtime_t videoEnd = videoTimeStamp;

  // Add a pre-roll to allow for audio processing latency
  videoEnd += kAudioPreRoll;

  bool eos = false;
  while ((audioTimestampsPerGroup.at(gr) < videoEnd) && !eos) {
    std::map<readerid_t, Audio::Samples> samplesByReader;
    mtime_t max_ts = std::numeric_limits<mtime_t>::lowest();

    bool enoughSamplesAvailable = true;
    // Make sure we will be able to read from all audioReaders
    for (auto& audioReader : audioAsyncReaders.at(gr)) {
      if (audioReader->available() < nbSamples) {
        eos = audioReader->eos();
        enoughSamplesAvailable = false;
      }
    }
    if (!enoughSamplesAvailable) {
      break;
    }

    for (auto& audioReader : audioAsyncReaders.at(gr)) {  // get a block from each input
      Audio::Samples smpls;
      readerid_t rId = audioReader->getId();

      if (audioReader->readSamples(nbSamples, smpls).getCode() == Input::ReadStatusCode::EndOfFile) {
        eos = true;
      }

      // Make the timestamp be relative to the controller's origin by taking into account the synchronization offset
      const double frameTime = (1000000.0 * double(getFrameRate().den)) / double(getFrameRate().num);
      const double frameOffset = double(getPano().getInput(audioReader->getId()).getFrameOffset());
      const mtime_t newOffset = mtime_t(round(frameOffset * frameTime));
      smpls.setTimestamp(smpls.getTimestamp() - newOffset);
      // get the latest block in order to align them afterward
      if (smpls.getTimestamp() > max_ts) {
        max_ts = smpls.getTimestamp();
      }

      samplesByReader[rId] = std::move(smpls);
    }

    if (audioPreProcs.find(gr) != audioPreProcs.end()) {
      applyAudioPreProc(samplesByReader, gr);
    }

    // Align the blocks by dropping some samples from "early" audio
    if (!eos) {
      for (size_t j = 0; j < audioAsyncReaders.at(gr).size(); ++j) {
        readerid_t id = audioAsyncReaders.at(gr)[j]->getId();
        if (samplesByReader[id].getTimestamp() < max_ts) {
          // Drop first samples, fetch new ones
          size_t offsetInSamples =
              (max_ts - samplesByReader[id].getTimestamp()) * audioPipeDef->getSamplingRate() / 1000000;
          while (offsetInSamples > 0) {
            size_t toDrop = std::min(offsetInSamples, nbSamples);
            Audio::Samples smpls;
            if (toDrop > audioAsyncReaders.at(gr)[j]->available()) {
              Logger::warning(CTRLtag) << "not enough audio to align outputs" << std::endl;
              audioIn.clear();
              return Input::ReadStatus::fromCode<Input::ReadStatusCode::TryAgain>();
            }
            audioAsyncReaders.at(gr)[j]->readSamples(toDrop, smpls);
            eos |= smpls.getNbOfSamples() != audioAsyncReaders.at(gr)[j]->rescale(toDrop);
            samplesByReader[id].drop(audioAsyncReaders.at(gr)[j]->rescale(toDrop));
            samplesByReader[id].append(smpls);
            offsetInSamples -= toDrop;
          }
        }
      }
    }

    // If the clocks of the panorama and the ambisonics are not running at the same reference,
    // we must assume something about their synchronization. Here the assumption is that the
    // first sample retrieved is at the same date as the first video frame retrieved. For real-time
    // setup this ensure close-to-synchronization, and it can be enhanced by setting a delay.
    if (audioVideoResync.at(gr)) {
      audioVideoResync.at(gr) = false;
      if (audioAndVideoInSameGroup.at(gr)) {
        // If audio and video in same group no need to add an offset
        audioTimestampOffsets.at(gr) = 0;
      } else {
        audioTimestampOffsets.at(gr) = videoTimeStamp - samplesByReader.begin()->second.getTimestamp();
      }
    }

    Logger::debug(CTRLtag) << "read group " << gr << " " << nbSamples << " samples at timestamp "
                           << samplesByReader.begin()->second.getTimestamp() << " -> "
                           << samplesByReader.begin()->second.getTimestamp() + audioTimestampOffsets.at(gr)
                           << std::endl;

    audioTimestampsPerGroup.at(gr) = max_ts + audioTimestampOffsets.at(gr);

    // Add samples to output
    audioIn.push_back(std::move(samplesByReader));
  }

  // If one of our audio inputs returns EOF, we don't return any audio
  if (eos) {
    audioIn.clear();
    Logger::debug(CTRLtag) << "load audio return EndOfFile" << std::endl;
    return Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>();
  }

  // EAGAIN if the non blocking IOs are not ready
  if (audioIn.size() == 0) {
    Logger::debug(CTRLtag) << "load audio return TryAgain" << std::endl;
    return Input::ReadStatus::fromCode<Input::ReadStatusCode::TryAgain>();
  }

  Audio::AudioBlock blk;
  // First dimension is the block, second dimension the input (e.g. audioIn[1][0] => second block of the first input)
  for (size_t i = 0; i < audioIn.size(); ++i) {  // for each block
    Audio::audioBlockReaderMap_t audioBlockPerReader;
    for (auto& readerData : audioIn[i]) {  // for each reader
      readerData.second.setTimestamp(readerData.second.getTimestamp() + audioTimestampOffsets.at(gr));
      // find the audio reader corresponding to this block
      for (size_t j = 0; j < audioAsyncReaders.at(gr).size(); ++j) {
        if (audioAsyncReaders.at(gr)[j]->getDelegate()->id == readerData.first) {
          audioAsyncReaders.at(gr)[j]->resample(readerData.second, blk);
          break;
        }
      }
      audioBlockPerReader[readerData.first] = std::move(blk);
    }

    if (i < audioBlocks.size()) {
      audioBlocks.at(i)[gr] = std::move(audioBlockPerReader);
    } else {
      Audio::audioBlockGroupMap_t audioBlockPerGroup;
      audioBlockPerGroup[gr] = std::move(audioBlockPerReader);
      audioBlocks.push_back(std::move(audioBlockPerGroup));
    }
  }

  Logger::verbose(CTRLtag) << "read group " << gr << " " << audioIn.size() << " audio blocks starting at timestamp "
                           << audioIn[0].begin()->second.getTimestamp() << std::endl;

  return Input::ReadStatus::OK();
}

Input::ReadStatus readMetadata(std::shared_ptr<Input::MetadataReader>& metadataReader, Input::MetadataChunk& data) {
  using StatusCode = Input::MetadataReader::MetadataReadStatus::StatusCode;

  auto metaStatusToReadStatus = [](Input::MetadataReader::MetadataReadStatus status) -> Input::ReadStatus {
    switch (status.getCode()) {
      case StatusCode::Ok:
      case StatusCode::MoreDataAvailable:
        return Input::ReadStatus::OK();
      case StatusCode::EndOfFile:
        return Input::ReadStatus::fromCode<Input::ReadStatusCode::EndOfFile>();
      case StatusCode::ErrorWithStatus:
        return Input::ReadStatus::fromError(status.getStatus());
    }
    assert(false);
    return Status{Origin::Input, ErrType::ImplementationError, "Could not load metadata, unknown error code"};
  };

  // IMU
  Input::ReadStatus status = metadataReader->readIMUSamples(data.imu);

  // Exposure
  {
    auto exposureStatus = Input::MetadataReader::MetadataReadStatus::fromCode<StatusCode::MoreDataAvailable>();
    while (exposureStatus.getCode() == StatusCode::MoreDataAvailable) {
      std::map<videoreaderid_t, Metadata::Exposure> exposureSample;
      exposureStatus = metadataReader->readExposure(exposureSample);
      if (exposureStatus.getCode() == StatusCode::Ok || exposureStatus.getCode() == StatusCode::MoreDataAvailable) {
        data.exposure.push_back(exposureSample);
      }
      status = (status.ok() ? metaStatusToReadStatus(exposureStatus) : status);
    }
  }

  // Camera response curves
  {
    auto tcStatus = Input::MetadataReader::MetadataReadStatus::fromCode<StatusCode::MoreDataAvailable>();
    while (tcStatus.getCode() == StatusCode::MoreDataAvailable) {
      std::map<videoreaderid_t, Metadata::ToneCurve> tcSample;
      tcStatus = metadataReader->readToneCurve(tcSample);
      if (tcStatus.getCode() == StatusCode::Ok || tcStatus.getCode() == StatusCode::MoreDataAvailable) {
        data.toneCurve.push_back(tcSample);
      }
      status = (status.ok() ? metaStatusToReadStatus(tcStatus) : status);
    }
  }

  return status;
}

bool ReaderController::needAudioVideoResync() {
  for (auto kv : audioVideoResync) {
    if (kv.second) {
      return true;
    }
  }
  return false;
}

Input::ReadStatus ReaderController::loadMetadata(Input::MetadataChunk& data) {
  Input::ReadStatus status;

  if (metadataReaders.empty()) {
    return Input::ReadStatus::fromCode<Input::ReadStatus::StatusCode::EndOfFile>();
  }

  // drop all old data from the caller
  data.clear();

  // one MetaDataChunk object shared by all inputs to limit copying/moving of data
  // the different metadata readers are expected to append only
  for (std::shared_ptr<Input::MetadataReader>& reader : metadataReaders) {
    auto metaStatus = readMetadata(reader, data);
    status = status.ok() ? metaStatus : status;
  }

  return status;
}

void ReaderController::applyAudioPreProc(std::map<readerid_t, Audio::Samples>& inOutMap, groupid_t gr) {
  std::vector<Audio::Samples> in;

  for (auto& kv : inOutMap) {
    in.push_back(std::move(kv.second));
  }
  audioPreProcs[gr]->process(in);
  int i = 0;
  for (auto& kv : inOutMap) {
    inOutMap[kv.first] = std::move(in[i]);
    i++;
  }
}

mtime_t ReaderController::reload(std::map<readerid_t, Input::PotentialFrame>& frames) {
  std::map<readerid_t, InputFrame> videoIn;

  {
    // protect from concurrent seeks
    std::lock_guard<std::mutex> lock(inputMutex);
    for (auto& reader : videoReaders) {
      videoIn[reader->getDelegate()->id] = reader->reload();
    }
  }

  std::map<readerid_t, Input::PotentialFrame> videoFrames;
  for (auto read : videoIn) {
    videoFrames.insert(std::map<readerid_t, Input::PotentialFrame>::value_type(
        read.first, {read.second.readerStatus, read.second.buffer}));
  }
  frames = videoFrames;
  return videoTimeStamp;
}

mtime_t ReaderController::getCommonReaderDate(std::vector<mtime_t> dates) {
  // select the correct video clock, taking into account:
  // - the state of each reader
  // - the initial time offset of the project
  // - the initial offset of each reader
  std::vector<mtime_t> realDates(dates.size());
  videoreaderid_t source = 0;
  for (auto& videoReader : videoReaders) {
    realDates[source] =
        dates[source] - mtime_t(round((getPano().getInput(videoReader->getDelegate()->id).getFrameOffset())) *
                                1000000.0 * double(getFrameRate().den) / double(getFrameRate().num));
    source++;
  }

  // if the reader is in a group with audio, select its clock!
  for (auto& videoReader : videoReaders) {
    groupid_t videoGrId = getPano().getInput(videoReader->getDelegate()->id).getGroup();
    if (videoReader->getSpec().frameRateIsProcedural) {
      continue;
    }
    if (videoGrId != -1) {
      if (audioAsyncReaders.find(videoGrId) != audioAsyncReaders.end()) {
        videoreaderid_t source = getPano().convertInputIndexToVideoInputIndex(videoReader->getDelegate()->id);
        assert(source >= 0 && (size_t)source < realDates.size());
        return realDates[source];
      }
    }
  }

  // no audio -> return the first real (non-procedural) date
  for (auto& videoReader : videoReaders) {
    if (videoReader->getSpec().frameRateIsProcedural) {
      continue;
    }
    videoreaderid_t source = getPano().convertInputIndexToVideoInputIndex(videoReader->getDelegate()->id);
    assert(source >= 0 && (size_t)source < realDates.size());
    return realDates[source];
  }

  // else randomly select one...
  assert(realDates.size());
  return realDates[0];
}

void ReaderController::releaseBuffer(std::map<readerid_t, Input::PotentialFrame>& frames) {
  // attribute a frame to each reader
  assert(videoReaders.size() == frames.size());
  for (auto& reader : videoReaders) {
    reader->releaseBuffer(frames.find(reader->getDelegate()->id)->second.frame);
  }
}

frameid_t ReaderController::getCurrentFrame() const { return getFrameRate().timestampToFrame(videoTimeStamp); }

void ReaderController::resetPano(const PanoDefinition& newPano) {
  PanoDefinition* myOldPano = pano;
  pano = newPano.clone();
  delete myOldPano;
}

void ReaderController::resetAudioPipe(const AudioPipeDefinition& newAudioPipeDef) {
  AudioPipeDefinition* oldAudioPipeDef = audioPipeDef;
  audioPipeDef = newAudioPipeDef.clone();
  delete oldAudioPipeDef;
}

Status ReaderController::seekFrame(frameid_t frame) {
  // protect from concurrent reads
  std::lock_guard<std::mutex> lock(inputMutex);

  if (frame < initialFrameOffset) {
    // Can only seek to a known time.
    std::stringstream msg;
    msg << "Trying to seek to frame " << frame << " which is before the beginning of the stream at "
        << initialFrameOffset;
    return {Origin::Input, ErrType::InvalidConfiguration, msg.str()};
  }

  if (frame == getCurrentFrame() + 1) {
    return Status::OK();
  }

  std::vector<std::future<Status>> futures;

  for (auto& reader : videoReaders) {
    auto fut = std::async(std::launch::async, [&] {
      return reader->seekFrame(frame + getPano().getInput(reader->getDelegate()->id).getFrameOffset());
    });
    futures.push_back(std::move(fut));
  }
  for (auto& grReader : audioAsyncReaders) {
    for (auto& reader : grReader.second) {
      auto fut = std::async(std::launch::async, [&] {
        return reader->getDelegate()->seekFrame(frame + getPano().getInput(reader->getDelegate()->id).getFrameOffset());
      });
      futures.push_back(std::move(fut));
    }
  }

  // Wait for all seek operations to return and record possible failure status
  Status seekStatus;
  for (std::future<Status>& fut : futures) {
    const Status s = fut.get();
    if (!s.ok()) {
      seekStatus = s;
    }
  }
  return seekStatus;
}

int ReaderController::getFirstReadableFrame() const {
  // No need to lock because we are only accessing const members in readers.
  int firstFrame = 0;
  for (auto& reader : videoReaders) {
    if ((int)reader->getFirstFrame() > firstFrame) {
      firstFrame = reader->getFirstFrame();
    }
  }
  return firstFrame;
}

frameid_t ReaderController::getLastReadableFrame() const {
  // No need to lock because we are only accessing const members in readers.
  int lastFrame = NO_LAST_FRAME;
  for (auto& reader : videoReaders) {
    if ((int)reader->getLastFrame() < lastFrame) {
      lastFrame = reader->getLastFrame();
    }
  }
  return lastFrame;
}

frameid_t ReaderController::getLastStitchableFrame() const {
  // No need to lock because we are only accessing const members in readers.
  frameid_t lastFrame = NO_LAST_FRAME;
  for (auto& reader : videoReaders) {
    frameid_t realLastFrame = reader->getLastFrame() - getPano().getInput(reader->getDelegate()->id).getFrameOffset();
    if (realLastFrame < lastFrame) {
      lastFrame = realLastFrame;
    }
  }
  return lastFrame;
}

std::vector<frameid_t> ReaderController::getLastFrames() const {
  // No need to lock because we are only accessing const members in readers.
  std::vector<frameid_t> lastFrames;
  for (const auto& reader : videoReaders) {
    lastFrames.push_back(reader->getLastFrame());
  }
  return lastFrames;
}

FrameRate ReaderController::getFrameRate() const { return frameRate; }

Status ReaderController::setupReaders() {
  FAIL_RETURN(GPU::useDefaultBackendDevice());

  for (size_t i = 0; i < videoReaders.size(); ++i) {
    const Status status = videoReaders[i]->perThreadInit();
    if (!status.ok()) {
      for (int j = (int)i - 1; j >= 0; --j) {
        videoReaders[j]->perThreadCleanup();
      }
      return status;
    }
  }
  return Status::OK();
}

void ReaderController::cleanReaders() {
  for (auto& reader : videoReaders) {
    GPU::useDefaultBackendDevice();
    reader->perThreadCleanup();
  }
}

Status ReaderController::setupAudioPreProc(const std::string& name, groupid_t gr) {
  Logger::info(CTRLtag) << "Setup the audio preprocessor " << name << std::endl;
  if (name == Audio::Orah::getOrahAudioSyncName()) {
    audioPreProcs[gr] = std::unique_ptr<Audio::AudioPreProcessor>(
        new Audio::Orah::OrahAudioSync(Audio::getBlockSizeFromInt(audioPipeDef->getBlockSize()), gr));
    return Status::OK();
  }
  return {Origin::AudioPreProcessor, ErrType::SetupFailure, "Cannot setup audio preprocessor: " + name};
}

}  // namespace Core
}  // namespace VideoStitch