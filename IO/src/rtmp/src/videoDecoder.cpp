// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videoDecoder.hpp"
#include "mockEncoder.hpp"
#include "x264Encoder.hpp"

#ifdef SUP_QUICKSYNC
#include "qsvEncoder.hpp"
#endif

#ifdef SUP_NVENC
#include "nvenc.hpp"
#endif

#include "libvideostitch/logging.hpp"

namespace VideoStitch {
namespace Input {

static const std::string DecoderTag("RTMP VideoDecoder");

static const std::string MOCK_STRING("mock");
static const std::string NVDEC_STRING("cuvid");
static const std::string QUICKSYNC_STRING("qsv");

VideoDecoder* createX264Decoder(int width, int height, FrameRate framerate);

VideoDecoder* createMockDecoder(int width, int height, FrameRate framerate);

VideoDecoder* createQSVDecoder(int width, int height, FrameRate framerate);

#ifdef SUP_NVDEC
VideoDecoder* createCuvidDecoder(int width, int height, FrameRate framerate);
#elif SUP_NVDEC_M
VideoDecoder* createNvV4l2Decoder(int width, int height, FrameRate framerate);
#endif

std::string VideoDecoder::typeToString(VideoDecoder::Type decoderType) {
  switch (decoderType) {
    case VideoDecoder::Type::CuVid:
      return "Nvidia";
    case VideoDecoder::Type::QuickSync:
      return "QuickSync";
    case VideoDecoder::Type::Mock:
      return "Mock";
  }
  assert(false);
  return "";
}

PotentialValue<VideoDecoder::Type> VideoDecoder::parseDecoderType(const std::string& decoderType) {
  if (decoderType.empty()) {
#ifdef SUP_QUICKSYNC
    Logger::info(DecoderTag) << "No decoder type set. Falling back to QuickSync." << std::endl;
    return VideoDecoder::Type::QuickSync;
#elif defined(SUP_NVDEC)
    Logger::info(DecoderTag) << "No decoder type set. Falling back to CuVid." << std::endl;
    return VideoDecoder::Type::CuVid;
#elif defined(SUP_NVDEC_M)
    Logger::info(DecoderTag) << "No decoder type set. Falling back to NvDec." << std::endl;
    return VideoDecoder::Type::CuVid;
#else
    Logger::info(DecoderTag) << "No decoder type set. Falling back to QuickSync." << std::endl;
    return VideoDecoder::Type::QuickSync;
#endif
  }

  if (decoderType == QUICKSYNC_STRING) {
    return VideoDecoder::Type::QuickSync;
  }
  if (decoderType == MOCK_STRING) {
    return VideoDecoder::Type::Mock;
  }
  if (decoderType == NVDEC_STRING) {
    return VideoDecoder::Type::CuVid;
  }

  return Status{VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration,
                DecoderTag + ": Unknown decoder type '" + decoderType + "'"};
}

Potential<VideoDecoder> VideoDecoder::createVideoDecoder(int width, int height, FrameRate framerate,
                                                         VideoDecoder::Type decoderType) {
  VideoDecoder* videoDecoder = nullptr;

  switch (decoderType) {
    case VideoDecoder::Type::CuVid:

#ifdef SUP_NVDEC
      videoDecoder = createCuvidDecoder(width, height, framerate);
      break;
#elif SUP_NVDEC_M
      videoDecoder = createNvV4l2Decoder(width, height, framerate);
      break;
#else
      return Status{VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration,
                    DecoderTag + ": " + typeToString(decoderType) + " decoding is not supported"};
#endif

    case VideoDecoder::Type::QuickSync:

#ifdef SUP_QUICKSYNC
      videoDecoder = createQSVDecoder(width, height, framerate);
      break;
#else
      return Status{VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration,
                    DecoderTag + ": " + typeToString(decoderType) + " decoding is not supported"};
#endif

    case VideoDecoder::Type::Mock:
      videoDecoder = createMockDecoder(width, height, framerate);
      break;
  }

  if (!videoDecoder) {
    return Status{VideoStitch::Origin::Output, VideoStitch::ErrType::InvalidConfiguration,
                  DecoderTag + ": Initialization of " + typeToString(decoderType) + " video decoder failed!"};
  }

  return videoDecoder;
}

AddressSpace VideoDecoder::decoderAddressSpace(VideoDecoder::Type decoderType) {
  switch (decoderType) {
    case Type::CuVid:
      return Device;
    case Type::QuickSync:
    case Type::Mock:
      return Host;
  }
  assert(false);
  return Host;
}

void VideoDecoder::demuxHeader(Span<const unsigned char> pkt, mtime_t timestamp, VideoStitch::IO::Packet& avpkt,
                               std::vector<unsigned char>& bitS) {
  std::vector<unsigned char> start_seq;
  start_seq.push_back(0);
  start_seq.push_back(0);
  start_seq.push_back(0);
  start_seq.push_back(1);  // start code

  bitS.clear();

  // sequence parameters set
  uint32_t spsLen = ((uint32_t)(pkt[11]) << 8) + pkt[12];
  std::copy(start_seq.begin(), start_seq.end(), std::back_inserter(bitS));
  std::copy(pkt.data() + 13, pkt.data() + 13 + spsLen, std::back_inserter(bitS));

  // picture parameters set
  std::copy(start_seq.begin(), start_seq.end(), std::back_inserter(bitS));
  uint32_t ppsLen = ((uint32_t)(pkt[13 + spsLen + 1]) << 8) + pkt[13 + spsLen + 2];
  std::copy(pkt.data() + 13 + spsLen + 3, pkt.data() + 13 + spsLen + 3 + ppsLen, std::back_inserter(bitS));

  avpkt.data = Span<unsigned char>(bitS.data(), bitS.size());
  avpkt.pts = (timestamp + ((uint64_t)(pkt[2]) << 16) + ((uint64_t)(pkt[3]) << 8) + ((uint64_t)(pkt[4]))) * 1000;
  avpkt.dts = timestamp * 1000;
}

bool VideoDecoder::demuxPacket(Span<const unsigned char> pkt, mtime_t timestamp, VideoStitch::IO::Packet& avpkt,
                               std::vector<unsigned char>& bitS) {
  std::vector<unsigned char> start_seq;
  start_seq.push_back(0);
  start_seq.push_back(0);
  start_seq.push_back(0);
  start_seq.push_back(1);  // start code

  bitS.clear();

  size_t bytesLeft = pkt.size() - 5;
  const unsigned char* ptr = pkt.data() + 5;
  while (bytesLeft) {
    uint32_t dataSize = ((uint32_t)(*(ptr)) << 24) + ((uint32_t)(*(ptr + 1)) << 16) + ((uint32_t)(*(ptr + 2)) << 8) +
                        (uint32_t)(*(ptr + 3));
    switch (*(ptr + 4) & 0x1f) {
      case 1:
      case 5:
        std::copy(start_seq.begin(), start_seq.end(), std::back_inserter(bitS));
        std::copy(ptr + 4, ptr + 4 + dataSize, std::back_inserter(bitS));
        break;
    }
    bytesLeft -= dataSize + 4;
    ptr += dataSize + 4;
  }

  avpkt.data = Span<unsigned char>(bitS.data(), bitS.size());
  avpkt.pts = (timestamp + ((uint64_t)(pkt[2]) << 16) + ((uint64_t)(pkt[3]) << 8) + ((uint64_t)(pkt[4]))) * 1000;
  avpkt.dts = timestamp * 1000;
  return true;
}

}  // namespace Input
}  // namespace VideoStitch
