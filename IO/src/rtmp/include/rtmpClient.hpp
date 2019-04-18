// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "rtmpEnums.hpp"
#include "rtmpStructures.hpp"

#include "audioDecoder.hpp"
#include "videoDecoder.hpp"

#include "libvideostitch/imuData.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/orah/exposureData.hpp"
#include "libvideostitch/utils/semaphore.hpp"
#ifdef USE_AVFORMAT
#include "avMuxer.hpp"
#endif

#include "librtmpIncludes.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace VideoStitch {
namespace Input {

class RTMPClient : public VideoReader, public AudioReader, public MetadataReader, public SinkReader {
 public:
  virtual ~RTMPClient();

  static RTMPClient* create(readerid_t id, const Ptv::Value* config, const int64_t width, const int64_t height);

  static bool handles(const Ptv::Value* config);

  ReadStatus readFrame(mtime_t& date, unsigned char* video) override;
  ReadStatus readSamples(size_t, Audio::Samples&) override;
  size_t available() override;
  bool eos() override;

  Status readIMUSamples(std::vector<VideoStitch::IMU::Measure>& imuData) override;
  MetadataReadStatus readExposure(std::map<videoreaderid_t, Metadata::Exposure>& exposure) override;
  MetadataReadStatus readWhiteBalance(std::map<videoreaderid_t, Metadata::WhiteBalance>& whiteBalance) override;
  MetadataReadStatus readToneCurve(std::map<videoreaderid_t, Metadata::ToneCurve>& toneCurve) override;

  virtual Status seekFrame(frameid_t) override { return VideoStitch::Status::OK(); }

  virtual Status seekFrame(mtime_t) override { return VideoStitch::Status::OK(); }

  Status addSink(const Ptv::Value* config, const mtime_t videoTimeStamp, const mtime_t audioTimeStamp) override;
  void removeSink() override;

 private:
  RTMPClient(readerid_t id, const Ptv::Value* config, const std::string& displayName, int64_t width, int64_t height,
             VideoStitch::PixelFormat fmt, FrameRate framerate, VideoDecoder::Type decoderType,
             Audio::ChannelLayout chanLayout, Audio::SamplingRate srate, Audio::SamplingDepth sdepth,
             FrameRate frameRateIMU);

  void readPacket(const RTMPPacket& packet);
  void readVideoPacket(const RTMPPacket& packet);
  void readInfoPacket(const RTMPPacket& packet);
  void readAudioPacket(const RTMPPacket& packet);

  void initEncoderData();

  void metaDataParse(AMFObject* amfObj);

  void metaDataParseOnText(AMFObject* amfObj, const uint32_t timestamp);

  videoreaderid_t getOrahInputID() const;

  /** @brief Retrieve the IMU data from an RTMP AMF object
   *
   *  @param amfObj: rtmp AMF object
   *  @param imuData: struct which contains the fields of IMU data
   *                  note: the timestamp field is not retrived from the AMF object,
   *                        it is retrived from the timestamp of the RTMP packet
   *  @return code: True if all the fields have been properly retrieved. False otherwise.
   */
  bool metaDataParseIMU(AMFObject* amfObj, VideoStitch::IMU::Measure& imuData);

  void readLoop();
  void decodeLoop();
  void audioDecodeLoop();

  void flushAudio();
  void flushVideo();
  void decodeFrame();

  bool connect();

  std::unique_ptr<AudioDecoder> audioDecoder;
  VideoDecoder* videoDecoder;

  std::atomic<mtime_t> inputVideoTimestamp; /* in ms */

  FrameRate fps;
  VideoDecoder::Type decoderType;
  uint64_t sampleRate;  // Hz
  uint8_t audioChannels;

  //-----------------------------------------------
  // stream startup stuff
  std::string URL;
  const Ptv::Value* config;

  // change connection status and notify listeners
  void setConnectionStatus(IO::RTMPConnectionStatus rtmpConnectionStatus);
  IO::RTMPConnectionStatus rtmpConnectionStatus;

  std::atomic<bool> stopping;

  std::unique_ptr<RTMP, std::function<void(RTMP*)>> rtmp;

  std::mutex frameMu;
  bool stoppingFrames = false;

  Semaphore videoSem;
  std::condition_variable frameCV;

  typedef std::pair<mtime_t, VideoPtr> Frame;
  std::queue<Frame> frames;

  // audio rtmp pkt buffer
  std::mutex audioPktQueueMutex;
  std::condition_variable audioPktQueueCond;
  std::queue<VideoStitch::IO::DataPacket> audioPktQueue;
  bool stoppingAudioQueue = false;

  // decoded audio buffer
  AudioStream audioStream;

  // decoded imu data queue
  std::mutex imuQueueMutex;
  std::queue<VideoStitch::IMU::Measure> imuQueue;

  // decoded exposure data queue
  std::mutex exposureQueueMutex;
  std::queue<std::map<videoreaderid_t, VideoStitch::Metadata::Exposure>> exposureQueue;

  // decoded tone curve data queue
  std::mutex toneCurveQueueMutex;
  std::queue<std::map<videoreaderid_t, VideoStitch::Metadata::ToneCurve>> toneCurveQueue;

  // These threads are launched during construction.
  // Please keep their declaration at the end to make sure that the rest of the object is initialized when they are
  // launched.
  std::thread* readThread;
  std::thread* decodeThread;
  std::thread* audioDecodeThread;

  Span<unsigned char> videoHeader;
#ifdef USE_AVFORMAT
  VideoStitch::Output::AvMuxer avSink;
#endif
};

}  // namespace Input
}  // namespace VideoStitch
