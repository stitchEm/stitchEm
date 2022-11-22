// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "filemuxer.hpp"
#include "livemuxer.hpp"
#include "avMuxer.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/parse.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

static std::string AVtag("libavoutput");
static std::string AVparsetag("LibavOutputWriter");

namespace VideoStitch {
namespace Output {

// allocate and initialize a AVPacket
// has a custom destructor, so you should not use 'new AVPacket' anymore (otherwise it will leak)
std::shared_ptr<struct AVPacket> newPacket() {
  auto deleter = [](struct AVPacket* pkt) {
    av_packet_unref(pkt);
    delete pkt;
  };
  auto pkt = std::shared_ptr<struct AVPacket>(new AVPacket, deleter);
  av_init_packet(pkt.get());
  pkt->data = nullptr;  // packet data will be allocated by the encoder
  pkt->size = 0;
  pkt->pts = 0;
  pkt->dts = 0;
  return pkt;
}

AvMuxer_pimpl* AvMuxer_pimpl::create(const Ptv::Value& config, AVCodecContext* videoCodecCtx,
                                     AVCodecContext* audioCodecCtx, FrameRate framerate, int currentImplNumber) {
  if (!videoCodecCtx) {
    Logger::warning(AVtag) << "AvMuxer : cannot create implementation without context" << std::endl;
    return nullptr;
  }
  std::vector<AVEncoder> codecs;
  codecs.push_back(AVEncoder(0, nullptr, videoCodecCtx));
  codecs.push_back(AVEncoder(1, nullptr, audioCodecCtx));
  AvMuxer_pimpl* pimpl =
      create_priv(config, codecs, videoCodecCtx->width, videoCodecCtx->height, framerate, -1, currentImplNumber);
  if (pimpl) pimpl->start();
  return pimpl;
}

// parse the configuration, perform sanity checks
AvMuxer_pimpl* AvMuxer_pimpl::create(const Ptv::Value& config, std::vector<AVEncoder>& codecs, unsigned width,
                                     unsigned height, FrameRate framerate, mtime_t firstPTS, int currentImplNumber) {
  AvMuxer_pimpl* pimpl = create_priv(config, codecs, width, height, framerate, firstPTS, currentImplNumber);

  if (pimpl) {
    pimpl->start();
  }
  return pimpl;
}

// parse the configuration, perform sanity checks
AvMuxer_pimpl* AvMuxer_pimpl::create_priv(const Ptv::Value& config, std::vector<AVEncoder>& codecs, unsigned width,
                                          unsigned height, FrameRate framerate, mtime_t firstPTS,
                                          int currentImplNumber) {
  // MAXIMUM MUXED FILE SIZE
  int64_t maxMuxedSize = 0;
  if (Parse::populateInt(AVparsetag, config, "max_video_file_chunk", maxMuxedSize, false) == Parse::PopulateResult_Ok) {
    Logger::verbose(AVtag) << "Maximum muxed chunk size: " << maxMuxedSize << std::endl;
    // take some margin to leave some room to write the video file trailer
    if (maxMuxedSize >= 100) {
      maxMuxedSize = (maxMuxedSize / 100) * 95;
    }
    if (maxMuxedSize < 200000000) {
      Logger::warning(AVtag) << "The maximum muxed chunk size " << maxMuxedSize << " is too small" << std::endl;
      Logger::warning(AVtag) << "Generated chunk size might exceed the limit because of flushing mechanism"
                             << std::endl;
    }
  }

  // MOOV ATOM SIZE CONTROL
  int moovAtomSize = 0;
  int maxFrameId = 0;
  // if the internal esimated moov atom size is lower,
  // we will use the min_moov_size as a miminum for the moov_size reservation
  int64_t minMoovSize = 0;
  if (Parse::populateInt(AVparsetag, config, "min_moov_size", minMoovSize, false) == Parse::PopulateResult_Ok) {
    Logger::verbose(AVtag) << "Minimum moov atom reserved size : " << minMoovSize << std::endl;
    if (!maxMuxedSize) {
      Logger::warning(AVtag) << "Cannot use min_moov_size if max_video_file_chunk is not set" << std::endl;
    }
  }

  // if the internal esimated moov atom size is bigger, we will not use the moov_size reservation process
  int64_t maxMoovSize = minMoovSize;
  if (Parse::populateInt(AVparsetag, config, "max_moov_size", maxMoovSize, false) == Parse::PopulateResult_Ok) {
    Logger::verbose(AVtag) << "Maximum moov atom reserved size : " << maxMoovSize << std::endl;
    if (!maxMuxedSize) {
      Logger::warning(AVtag) << "Cannot use max_moov_size if max_video_file_chunk is not set" << std::endl;
    }
  }

  // CODEC
  std::string avCodec(LIBAV_WRITER_DEFAULT_CODEC);
  if (Parse::populateString(AVparsetag, config, "video_codec", avCodec, false) == Parse::PopulateResult_WrongType) {
    return nullptr;
  }

  // BITRATE
  int avBR = LIBAV_WRITER_DEFAULT_BITRATE;
  if (avCodec != "prores" && avCodec != "mjpeg") {
    if (Parse::populateInt(AVparsetag, config, "bitrate", avBR, false) == Parse::PopulateResult_WrongType) {
      Logger::error(AVtag) << "Parameter 'bitrate' not found in the configuration. Aborting" << std::endl;
      return nullptr;
    }
  }

  // FPS
  double avFR = (double)framerate.num / framerate.den;

  // BITRATE MODE
  std::string avRCMode(LIBAV_WRITER_DEFAULT_BITRATE_MODE);
  if (Parse::populateString(AVparsetag, config, "bitrate_mode", avRCMode, false) == Parse::PopulateResult_WrongType) {
    return nullptr;
  }

  float lazy = 1.0;
  if ((avRCMode == "VBR") && (avCodec == "h264_nvenc")) {
    /* Nvenc can have difficulties to reach the target bitrate in VBR mode.
       using a formula on internet to get a resolution/bitrate relation to enable recording more in one chunk */
    lazy = (float)(avBR * 1.0 / (pow(1.0 * avFR * width * height, 0.75) * 30));
    if (lazy < 1.0) lazy = 1.0;
  }
  /* estimation based on test with some margin, as a value too small will result in a corrupted file
     moov atom contains information for each frame so it size will depend on the number of frames in one chunk
     which depends on the bitrate and framerate */
  moovAtomSize = (int)(maxMuxedSize * (LIBAV_WRITER_DEFAULT_BITRATE * 0.0008 / avBR) * (avFR / 30) * lazy);
  if (moovAtomSize < minMoovSize) moovAtomSize = (int)minMoovSize;
  if ((moovAtomSize > maxMoovSize) || !maxMuxedSize) {
    if (maxMuxedSize && maxMoovSize) {
      Logger::warning(AVtag) << "Estimated moov atom size " << moovAtomSize << " exceed the maximum reserved size "
                             << maxMoovSize << std::endl;
      Logger::warning(AVtag) << "Cancel reserved space for moov atom at the beginning of the file" << std::endl;
    }
    moovAtomSize = 0;
  }

  // -------------- instantiate the runtime objects
  std::vector<Muxer*> muxers;
  // handle multiple muxers
  // XXX TODO FIXME
  // Android crashes if there is no default value assigned
  std::string format = "mp4";
  if (Parse::populateString(AVparsetag, config, "type", format, true) != Parse::PopulateResult_Ok) {
    Logger::error(AVtag) << "Could not create muxer of this type" << std::endl;
    return nullptr;
  }
  std::string filename = "default";
  if (Parse::populateString(AVparsetag, config, "filename", filename, false) != Parse::PopulateResult_Ok) {
    Logger::error(AVtag) << "Could not create muxer : missing filename target" << std::endl;
    return nullptr;
  }

  // CREATE MUXERS
  // make up a new file base name
  std::stringstream fileBaseName;
  fileBaseName << filename;
  if (format != LIBAV_WRITER_LIVE_RTMP && currentImplNumber) {
    fileBaseName << '-' << currentImplNumber;
  }
  Logger::verbose(AVtag) << "Found muxer type '" << format << "' with filename '" << fileBaseName.str() << "'"
                         << std::endl;

  if (format == LIBAV_WRITER_LIVE_RTMP) {
    LiveMuxer* liveMuxer = new LiveMuxer(0, fileBaseName.str(), codecs);
    if (!liveMuxer->openResource(fileBaseName.str())) {
      Logger::error(AVtag) << "Could not initialize RTMP stream" << std::endl;
      delete liveMuxer;
      return nullptr;
    }
    muxers.push_back(liveMuxer);
  } else {
    AVDictionary* muxerConfig = nullptr;
    if (moovAtomSize) {
      Logger::verbose(AVtag) << "moov atom reserved size : " << moovAtomSize << std::endl;
      av_dict_set_int(&muxerConfig, "moov_size", moovAtomSize, 0);
      maxFrameId = (int)(maxMuxedSize * 8.0 * avFR * lazy / avBR);
    }
    FileMuxer* fileMuxer = new FileMuxer(0, format, fileBaseName.str(), codecs, muxerConfig);
    if (!fileMuxer->openResource(fileBaseName.str())) {
      Logger::error(AVtag) << "Could not initialize file" << std::endl;
      delete fileMuxer;
      return nullptr;
    }
    muxers.push_back(fileMuxer);
    av_dict_free(&muxerConfig);
  }

  return new AvMuxer_pimpl(maxMuxedSize, maxFrameId, firstPTS, muxers);
}

AvMuxer_pimpl::AvMuxer_pimpl(const int64_t maxMuxedSize, const int maxFrameId, const mtime_t firstPTS,
                             const std::vector<Muxer*>& mxs)
    : muxers(mxs),
      m_maxMuxedSize(maxMuxedSize),
      m_maxFrameId(maxFrameId),
      m_curFrameId(0),
      m_Status(MuxerThreadStatus::OK),
      m_needsRespawn(false),
      m_firstPTS(firstPTS),
      m_lastPTS(-1) {}

AvMuxer_pimpl::~AvMuxer_pimpl() { close(); }

void AvMuxer_pimpl::start() {
  // Start Muxers
  for (auto muxer : muxers) {
    muxer->start();
  }
}

// -------------------------- Encoding --------------------------

MuxerThreadStatus AvMuxer_pimpl::getStatus() {
  if (m_Status == MuxerThreadStatus::OK) {
    for (Muxer* muxer : muxers) {
      if (muxer->getThreadStatus() != MuxerThreadStatus::OK) {
        m_Status = muxer->getThreadStatus();
        return m_Status;
      }
    }
  }
  return m_Status;
}

void AvMuxer_pimpl::updateRespawnStatus(const bool isvideo) {
  if (m_needsRespawn) {
    return;
  }
  // check if maximum video file chunk size is reached, or if a timeout error occured
  // NOTE: the file size is given by ffmpeg, it does not take into account unflushed data
  bool maxMuxedSizeReached = false;
  bool timeoutError = false;
  bool networkError = false;
  for (Muxer* muxer : muxers) {
    maxMuxedSizeReached = maxMuxedSizeReached || (m_maxMuxedSize && muxer->getMuxedSize() >= m_maxMuxedSize);
    timeoutError = timeoutError || (muxer->getThreadStatus() == MuxerThreadStatus::TimeOutError);
    networkError = networkError || (muxer->getThreadStatus() == MuxerThreadStatus::NetworkError);
  }
  // respawn the implementation if max chunk size is reached or if timeout error on RTMP output stream
  if (maxMuxedSizeReached) {
    Logger::warning(AVtag) << "Maximum muxed size reached, needs reset" << std::endl;
    /* will reset the writer at the next LibavWriter::pushFrame() */
    m_needsRespawn = true;
  }
  // respawn the implementation if timeout error on RTMP output stream occured
  if (timeoutError || networkError) {
    Logger::warning(AVtag) << "Timeout occured on RTMP output, needs reset" << std::endl;
    /* will reset the writer at the next LibavWriter::pushFrame() */
    m_needsRespawn = true;
  }
  if (isvideo) m_curFrameId++;
  // In case effective bitrate is lower than expected this is to avoid "reserved_moov_size is too small"
  if (m_maxFrameId && (m_curFrameId > m_maxFrameId)) {
    Logger::warning(AVtag) << "Maximum frameId reached " << m_curFrameId << ", needs reset" << std::endl;
    m_needsRespawn = true;
  }
}

bool AvMuxer_pimpl::needsRespawn() const { return m_needsRespawn; }

void AvMuxer_pimpl::pushVideoPacket(std::shared_ptr<struct AVPacket> videoPkt) {
  if (videoPkt->size == 0) {
    return;
  }

  Logger::debug(AVtag) << "Pushing video packet of size " << videoPkt->size << " from frame " << videoPkt->pts
                       << " to the muxer" << std::endl;
  for (Muxer* muxer : muxers) {
    muxer->packets.pushPacket(videoPkt, 0);
  }

  updateRespawnStatus(true);
}

void AvMuxer_pimpl::pushAudioPacket(std::shared_ptr<struct AVPacket> audioPkt) {
  if (audioPkt->size == 0) {
    return;
  }

  Logger::debug(AVtag) << "Pushing audio packet of size " << audioPkt->size << " from frame " << audioPkt->pts
                       << " to the muxer" << std::endl;
  for (Muxer* muxer : muxers) {
    muxer->packets.pushPacket(audioPkt, 1);
  }
  updateRespawnStatus(0);
}

void AvMuxer_pimpl::pushMetadataPacket(std::shared_ptr<struct AVPacket> pkt) {
  if (pkt->size == 0) {
    return;
  }

  Logger::debug(AVtag) << "Pushing metadata packet of size " << pkt->size << " from frame " << pkt->pts
                       << " to the muxer" << std::endl;
  for (Muxer* muxer : muxers) {
    muxer->packets.pushPacket(pkt, 2);
  }
  updateRespawnStatus(0);
}

MuxerThreadStatus AvMuxer_pimpl::close() {
  // wait for the muxers
  for (auto& muxer : muxers) {
    muxer->join();
  }
  // Write muxer trailer
  for (auto& muxer : muxers) {
    muxer->writeTrailer();
  }

  // get status before deleting the muxers to retrieve pending error from muxers
  MuxerThreadStatus status = getStatus();
  for (auto muxer : muxers) {
    delete muxer;
  }
  muxers.clear();
  return status;
}

AvMuxer::AvMuxer() : m_config(nullptr), m_pimpl(nullptr) {}

VideoStitch::Status AvMuxer::create(const Ptv::Value& config, unsigned width, unsigned height, FrameRate framerate,
                                    Span<unsigned char>& header, mtime_t videotimestamp, mtime_t audiotimestamp) {
  // register all codecs, demux and protocols
  Util::Libav::checkInitialization();

  std::lock_guard<std::mutex> lk(pimplMu);
  if (m_pimpl) {
    return {Origin::Output, ErrType::RuntimeError, "AvMuxer already instantiated"};
  }

  m_codecs.resize(3);

  /* need to create AVCodec as there is no known AVCodecContext */
  // VIDEO CODEC
  m_codecs[0].id = 0;
  std::string avCodec(LIBAV_WRITER_DEFAULT_CODEC);
  if (Parse::populateString(AVparsetag, config, "video_codec", avCodec, false) == Parse::PopulateResult_WrongType) {
    return {Origin::Output, ErrType::InvalidConfiguration, "AvMuxer \"video_codec\" field error in the configuration"};
  }

  // TODO change design AVEncoder design to deal with const AVCodec *, remove cast
  m_codecs[0].codec = const_cast<AVCodec *>(avcodec_find_encoder_by_name(avCodec.c_str()));
  if (!m_codecs[0].codec) {
    Logger::error(AVtag) << "Missing video codec" << avCodec.c_str() << ", disable output." << std::endl;
    return {Origin::Output, ErrType::InvalidConfiguration, "AvMuxer missing video codec"};
  }

  // AUDIO CODEC
  m_codecs[1].id = 1;
  std::string sampleFormat;
  std::string channelLayout;
  std::string audioCodecName;
  if (Parse::populateString(AVparsetag, config, "audio_codec", audioCodecName, false) != Parse::PopulateResult_Ok) {
    Logger::error(AVtag) << "Audio codec: " << audioCodecName.c_str() << ": invalid audio codec name" << std::endl;
  } else {
    m_codecs[1].codec = const_cast<AVCodec *>(avcodec_find_encoder_by_name(audioCodecName.c_str()));
    if (!m_codecs[1].codec) {
      Logger::error(AVtag) << "Audio codec: " << audioCodecName.c_str() << " not found, disable output." << std::endl;
      return {Origin::Output, ErrType::InvalidConfiguration, "AvMuxer missing audio codec"};
    }

    // SAMPLING RATE
    if (Parse::populateInt(AVparsetag, config, "sampling_rate", m_sampleRate, true) != Parse::PopulateResult_Ok) {
      Logger::error(AVtag) << "Audio codec: " << m_sampleRate << " : invalid sample rate (32000, 44100 or 48000)"
                           << std::endl;
      return {Origin::Output, ErrType::InvalidConfiguration,
              "AvMuxer \"sampling_rate\" field error in the configuration"};
    }
    // SAMPLE FORMAT
    if (Parse::populateString(AVparsetag, config, "sample_format", sampleFormat, true) != Parse::PopulateResult_Ok ||
        av_get_sample_fmt(sampleFormat.c_str()) == 0) {
      Logger::error(AVtag) << "Audio codec: " << sampleFormat.c_str() << " : invalid sample format" << std::endl;
      return {Origin::Output, ErrType::InvalidConfiguration,
              "AvMuxer \"sample_format\" field error in the configuration"};
    }
    // CHANNEL LAYOUT
    if (Parse::populateString(AVparsetag, config, "channel_layout", channelLayout, true) != Parse::PopulateResult_Ok) {
      Logger::error(AVtag) << "Audio codec: " << channelLayout.c_str() << " : invalid channel layout" << std::endl;
      return {Origin::Output, ErrType::InvalidConfiguration,
              "AvMuxer \"channel_layout\" field error in the configuration"};
    }
    if (av_get_channel_layout(channelLayout.c_str()) == 0) {
      Logger::error(AVtag) << "Audio codec: " << channelLayout.c_str() << " : invalid channel layout" << std::endl;
      return {Origin::Output, ErrType::InvalidConfiguration,
              "AvMuxer \"channel_layout\" field error in the configuration"};
    }
  }

  // METADATA CODEC
  m_codecs[2].id = 2;
  std::string metadataCodecName;
  if (Parse::populateString(AVparsetag, config, "metadata_codec", metadataCodecName, false) !=
      Parse::PopulateResult_Ok) {
  } else {
    m_codecs[2].codec = const_cast<AVCodec *>(avcodec_find_encoder_by_name(metadataCodecName.c_str()));
    if (!m_codecs[2].codec) {
      Logger::error(AVtag) << "Metadata codec: " << metadataCodecName.c_str() << " not found, disable output."
                           << std::endl;
      return {Origin::Output, ErrType::InvalidConfiguration,
              "AvMuxer \"metadata_codec\" field error in the configuration"};
    }
  }

  m_config = config.clone();
  m_width = width;
  m_height = height;
  m_framerate = framerate;
  m_header = std::vector<unsigned char>(header.begin(), header.end());
  m_firstPTS = videotimestamp;
  m_audioOffset = videotimestamp - audiotimestamp;
  m_sampleDepth = VideoStitch::Util::sampleFormat(av_get_sample_fmt(sampleFormat.c_str()));
  m_channelLayout = av_get_channel_layout(channelLayout.c_str());
  m_currentImplNumber = 0;

  /* need to create AvCodecContext to share time_base */
  // VIDEO ENCODER
  m_codecs[0].codecContext = avcodec_alloc_context3(m_codecs[0].codec);
  if (!m_codecs[0].codecContext) {
    Logger::error(AVtag) << "Could not allocate encoder context for video codec " << avCodec.c_str()
                         << ", disable output." << std::endl;
    return {Origin::Output, ErrType::RuntimeError, "AvMuxer could not allocate video encoder context"};
  }
  m_codecs[0].codecContext->width = m_width;
  m_codecs[0].codecContext->height = m_height;
  m_codecs[0].codecContext->time_base = {1, 1000000};
  /* extradata contains the SPS/PPS */
  if (m_header.size()) {
    m_codecs[0].codecContext->extradata = (uint8_t*)malloc(m_header.size());
    memcpy(m_codecs[0].codecContext->extradata, m_header.data(), m_header.size());
    m_codecs[0].codecContext->extradata_size = (int)m_header.size();
  }

  // AUDIO ENCODER
  if (m_codecs[1].codec) {
    m_codecs[1].codecContext = avcodec_alloc_context3(m_codecs[1].codec);
    if (!m_codecs[1].codecContext) {
      Logger::error(AVtag) << "Could not allocate encoder context for audio codec " << audioCodecName.c_str()
                           << ", disable output." << std::endl;
      return {Origin::Output, ErrType::RuntimeError, "AvMuxer could not allocate audio encoder context"};
    }
    m_codecs[1].codecContext->sample_rate = m_sampleRate;
    m_codecs[1].codecContext->sample_fmt = VideoStitch::Util::libavSampleFormat(m_sampleDepth);
    m_codecs[1].codecContext->channel_layout = m_channelLayout;
    m_codecs[1].codecContext->channels = av_get_channel_layout_nb_channels(m_channelLayout);
    m_codecs[1].codecContext->time_base = {1, 1000000};
  }

  // METADATA ENCODER
  if (m_codecs[2].codec) {
    m_codecs[2].codecContext = avcodec_alloc_context3(m_codecs[2].codec);
    if (!m_codecs[2].codecContext) {
      Logger::error(AVtag) << "Could not allocate encoder context for metadata codec " << metadataCodecName.c_str()
                           << ", disable output." << std::endl;
      return {Origin::Output, ErrType::RuntimeError, "AvMuxer could not allocate metadata encoder context"};
    }
    m_codecs[2].codecContext->time_base = {1, 1000000};
  }

  m_pimpl.reset(
      AvMuxer_pimpl::create(config, m_codecs, m_width, m_height, m_framerate, m_firstPTS, m_currentImplNumber++));

  if (m_pimpl) {
    return VideoStitch::Status::OK();
  } else {
    return {Origin::Output, ErrType::RuntimeError, "Can't create AvMuxer"};
  }
}

void AvMuxer::destroy() {
  {
    std::lock_guard<std::mutex> lk(pimplMu);
    if (m_pimpl) {
      m_pimpl->close();
    }
    m_pimpl.reset(nullptr);
  }
  for (auto encoder : m_codecs) {
    if (encoder.codecContext) {
      avcodec_free_context(&encoder.codecContext);
    }
  }
}

AvMuxer::~AvMuxer() {
  destroy();
  if (m_config) {
    delete m_config;
  }
}

bool AvMuxer::implReady() {
  if (m_pimpl) {
    MuxerThreadStatus status = m_pimpl->getStatus();

    /* check if we need to reset the implementation */
    if (m_pimpl->needsRespawn()) {
      status = m_pimpl->close();
      m_pimpl.reset(AvMuxer_pimpl::create(*m_config, m_codecs, m_width, m_height, m_framerate, m_firstPTS,
                                          m_currentImplNumber++));
      if (!m_pimpl) {
        Logger::error(AVtag) << "AvMuxer : no implementation is instantiated, respawn failed" << std::endl;
        return false;
      }
    }
    if (status != MuxerThreadStatus::OK) {
      Logger::warning(AVtag) << "AvMuxer encountered error " << (int)status << std::endl;
    }
    return true;
  }
  return false;
}

void AvMuxer::pushVideoPacket(const VideoStitch::IO::Packet& videoPkt) {
  std::lock_guard<std::mutex> lk(pimplMu);
  if (implReady()) {
    // initialize first seen PTS
    if (m_pimpl->firstFrame()) {
      m_pimpl->m_firstPTS = videoPkt.pts;
      Logger::debug(AVtag) << "m_firstPTS: " << m_pimpl->m_firstPTS << std::endl;
    }
    auto pkt = newPacket();
    pkt->data = (uint8_t*)videoPkt.data.begin();
    pkt->size = (int)videoPkt.data.size();
    pkt->pts = videoPkt.pts - m_pimpl->m_firstPTS;
    pkt->dts = videoPkt.dts - m_pimpl->m_firstPTS;
    Logger::verbose(AVtag) << "(VIDEO): date: " << videoPkt.pts << "  pts: " << pkt->pts << std::endl;
    m_pimpl->pushVideoPacket(pkt);
  }
}

void AvMuxer::pushAudioPacket(const VideoStitch::IO::Packet& audioPkt) {
  std::lock_guard<std::mutex> lk(pimplMu);
  if (implReady()) {
    // initialize first seen PTS
    if (m_pimpl->firstFrame()) {
      m_pimpl->m_firstPTS = audioPkt.pts;
      Logger::debug(AVtag) << "m_firstPTS: " << m_pimpl->m_firstPTS << std::endl;
    }

    auto pkt = newPacket();
    pkt->data = (uint8_t*)audioPkt.data.begin();
    pkt->size = (int)audioPkt.data.size();
    pkt->pts = audioPkt.pts + m_audioOffset - m_pimpl->m_firstPTS;
    pkt->dts = audioPkt.dts + m_audioOffset - m_pimpl->m_firstPTS;
    Logger::verbose(AVtag) << "(AUDIO): date: " << audioPkt.pts << "  pts: " << pkt->pts << std::endl;
    m_pimpl->pushAudioPacket(pkt);
  }
}

void AvMuxer::pushMetadataPacket(const VideoStitch::IO::Packet& metaPkt) {
  std::lock_guard<std::mutex> lk(pimplMu);
  if (implReady()) {
    // initialize first seen PTS
    if (m_pimpl->firstFrame()) {
      m_pimpl->m_firstPTS = metaPkt.pts;
      Logger::debug(AVtag) << "m_firstPTS: " << m_pimpl->m_firstPTS << std::endl;
    }

    auto pkt = newPacket();
    pkt->data = (uint8_t*)metaPkt.data.begin();
    pkt->size = (int)metaPkt.data.size();
    pkt->pts = metaPkt.pts - m_pimpl->m_firstPTS;
    pkt->dts = metaPkt.dts - m_pimpl->m_firstPTS;
    Logger::verbose(AVtag) << "(METADATA): date: " << metaPkt.pts << "  pts: " << pkt->pts << std::endl;
    m_pimpl->pushMetadataPacket(pkt);
  }
}

}  // namespace Output
}  // namespace VideoStitch
