// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <fcntl.h>
#include <sys/time.h>

#include <unistd.h>
#include <cassert>
#include <fstream>
#include <ostream>
#include <iostream>
#include <vector>

#define ANDROID_DEBUG

#if (defined ANDROID_DEBUG)
#include <android/log.h>
#else /*	(defined ANDROID_DEBUG)	*/
#define ANDROID_LOG_INFO 0
void __android_log_print(int, const char*, const char*, ...) {}
#endif /*	(defined ANDROID_DEBUG)	*/

#include <OMXAL/OpenMAXAL.h>

#include "libvideostitch/input.hpp"
#include "libvideostitch/logging.hpp"
#include "frameRateHelpers.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <extensionChecker.hpp>

#include "mp4Input.hpp"

static std::string MP4tag("Mp4Reader");
#define CHECK(fn)                                                               \
  {                                                                             \
    media_status_t mErr = (fn);                                                 \
    if (mErr != AMEDIA_OK) {                                                    \
      Logger::warning(MP4tag) << #fn " failed with code " << mErr << std::endl; \
    }                                                                           \
  }

namespace VideoStitch {
namespace Input {

Mp4Reader* Mp4Reader::create(readerid_t id, const std::string& fileName,
                             const Plugin::VSReaderPlugin::Config& runtime) {
  const ProbeResult& probeResult = Mp4Reader::probe(fileName);
  if (!probeResult.valid) {
    return nullptr;
  } else if (probeResult.width != runtime.width || probeResult.height != runtime.height) {
    Logger::error(MP4tag) << "Input size (" << probeResult.width << "x" << probeResult.height
                          << ") is different from expected size (" << runtime.width << "x" << runtime.height << ")"
                          << std::endl;
    return nullptr;
  }

  FrameRate mFrameRate = {30, 1};
  AMediaExtractor* extractor = AMediaExtractor_new();
  AMediaCodec* videoCodec = nullptr;
  int32_t pixelFormat;
  PixelFormat mPixelFormat = PixelFormat::Unknown;
  int32_t mStride;

  int fd = open(fileName.c_str(), O_RDONLY);
  CHECK(AMediaExtractor_setDataSourceFd(extractor, fd, 0, LONG_MAX))
  close(fd);
  int numtracks = (int)AMediaExtractor_getTrackCount(extractor);
  for (int i = 0; i < numtracks; i++) {
    AMediaFormat* format = AMediaExtractor_getTrackFormat(extractor, i);
    Logger::info(MP4tag) << "AMedia_TrackFormat : " << AMediaFormat_toString(format) << std::endl;
    const char* mime;
    if (!AMediaFormat_getString(format, AMEDIAFORMAT_KEY_MIME, &mime)) {
      Logger::info(MP4tag) << "no mime type" << std::endl;
      /*
       *	No mime type?
       */
    } else if (strncmp(mime, "video/", strlen("video/")) == 0) {
      int32_t i_framerate;
      float f_framerate;
      if (AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_FRAME_RATE, &i_framerate)) {
        Logger::info(MP4tag) << "Frame rate : " << i_framerate << " fps" << std::endl;
        mFrameRate = {i_framerate, 1};
      } else if (AMediaFormat_getFloat(format, AMEDIAFORMAT_KEY_FRAME_RATE, &f_framerate)) {
        Logger::info(MP4tag) << "Frame rate : " << f_framerate << " fps" << std::endl;
        mFrameRate = Util::fpsToNumDen(f_framerate);
      }
      CHECK(AMediaExtractor_selectTrack(extractor, i))
      videoCodec = AMediaCodec_createDecoderByType(mime);
      CHECK(AMediaCodec_configure(videoCodec, format, NULL, NULL, 0))
      CHECK(AMediaCodec_start(videoCodec))
      AMediaFormat* mFormat = AMediaCodec_getOutputFormat(videoCodec);
      Logger::info(MP4tag) << "AMedia_OutputFormat : " << AMediaFormat_toString(mFormat) << std::endl;

      AMediaFormat_getInt32(mFormat, AMEDIAFORMAT_KEY_STRIDE, &mStride);
      AMediaFormat_getInt32(mFormat, AMEDIAFORMAT_KEY_COLOR_FORMAT, &pixelFormat);
      switch (pixelFormat) {
        case COLOR_FormatYUV420SemiPlanar:
        case COLOR_QCOM_FormatYUV420SemiPlanar:
        case COLOR_TI_FormatYUV420PackedSemiPlanar:
          mPixelFormat = PixelFormat::NV12;
          break;
        case COLOR_FormatYUV420Planar:
          Logger::warning(MP4tag) << "AMediaFormat YUV420 Planar PixelFormat : " << std::endl;
          mPixelFormat = PixelFormat::YV12;
          break;
        default:
          Logger::warning(MP4tag) << "AMediaFormat unknown PixelFormat : " << pixelFormat << std::endl;
          break;
      }
      CHECK(AMediaFormat_delete(mFormat))
    }
    CHECK(AMediaFormat_delete(format))
  }

  if (videoCodec == nullptr) {
    Logger::error(MP4tag) << "no video track detected" << std::endl;
    return nullptr;
  }

  return new Mp4Reader(id, extractor, videoCodec, mFrameRate, mPixelFormat, mStride, (int32_t)runtime.width,
                       (int32_t)runtime.height);
}

ProbeResult Mp4Reader::probe(const std::string& fileName) {
  int32_t width = -1, height = -1;
  bool valid = false;

  AMediaExtractor* extractor = AMediaExtractor_new();
  int fd = open(fileName.c_str(), O_RDONLY);
  if (fd <= 0) {
    Logger::error(MP4tag) << "Fail to open " << fileName << std::endl;
  }
  CHECK(AMediaExtractor_setDataSourceFd(extractor, fd, 0, LONG_MAX))
  close(fd);
  int numtracks = (int)AMediaExtractor_getTrackCount(extractor);
  for (int i = 0; i < numtracks; i++) {
    AMediaFormat* format = AMediaExtractor_getTrackFormat(extractor, i);
    const char* mime;
    if ((AMediaFormat_getString(format, AMEDIAFORMAT_KEY_MIME, &mime) == true) &&
        (strncmp(mime, "video/", strlen("video/")) == 0)) {
      CHECK(AMediaExtractor_selectTrack(extractor, i))
      AMediaCodec* codec = AMediaCodec_createDecoderByType(mime);
      CHECK(AMediaCodec_configure(codec, format, NULL, NULL, 0))
      CHECK(AMediaCodec_start(codec))
      AMediaFormat* format = AMediaCodec_getOutputFormat(codec);
      if ((AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_WIDTH, &width) == true) &&
          (AMediaFormat_getInt32(format, AMEDIAFORMAT_KEY_HEIGHT, &height) == true)) {
        valid = true;
      }
      CHECK(AMediaCodec_stop(codec))
      CHECK(AMediaCodec_delete(codec))
    }
    CHECK(AMediaFormat_delete(format))
  }
  CHECK(AMediaExtractor_delete(extractor))
  return ProbeResult({valid, false, -1, -1, (int64_t)width, (int64_t)height, false, valid});
}

Mp4Reader::Mp4Reader(readerid_t id, AMediaExtractor* extractor, AMediaCodec* videoCodec, FrameRate frameRate,
                     PixelFormat pixelFormat, int32_t stride, int32_t width, int32_t height)
    : Reader(id),
      VideoReader(width, height, VideoStitch::getFrameDataSize(width, height, pixelFormat), pixelFormat, Host,
                  frameRate, 0, NO_LAST_FRAME, false, nullptr),
      mExtractor(extractor),
      mCodec(videoCodec),
      mWidth(width),
      mHeight(height),
      mPixelFormat(pixelFormat),
      mStride(stride),
      mStarted(false),
      mTimeval({0, 0}) {}

Mp4Reader::~Mp4Reader() {
  CHECK(AMediaCodec_stop(mCodec))
  CHECK(AMediaCodec_delete(mCodec))
  CHECK(AMediaExtractor_delete(mExtractor))
}

bool Mp4Reader::handles(const std::string& filename) {
  __android_log_print(ANDROID_LOG_INFO, "Mp4Reader", "bool Mp4Reader::handles()");
  return (hasExtension(filename, ".mp4") || hasExtension(filename, ".MP4"));
}

ReadStatus Mp4Reader::readFrame(mtime_t& date, unsigned char* data) {
  if (!mStarted) {
    gettimeofday(&mTimeval, nullptr);
    mStarted = true;
  }

  bool gotFrame = false;
  int nb_iter = 0;

  while (!gotFrame) {
    nb_iter++;

    /*	Input	*/
    {
      ssize_t bufidx = AMediaCodec_dequeueInputBuffer(mCodec, 2000);
      if (bufidx >= 0) {
        size_t bufsize;
        uint8_t* buf = AMediaCodec_getInputBuffer(mCodec, bufidx, &bufsize);
        ssize_t sampleSize = AMediaExtractor_readSampleData(mExtractor, buf, bufsize);
        if (sampleSize < 0) {
          float fdiff;
          struct timeval tv;
          gettimeofday(&tv, nullptr);
          fdiff = float(double(((int64_t(tv.tv_sec) * 1000000) + int64_t(tv.tv_usec)) -
                               ((int64_t(mTimeval.tv_sec) * 1000000) + int64_t(mTimeval.tv_usec))) /
                        1000000.0);
          mTimeval = tv;
          Logger::info(MP4tag) << "loop time: " << fdiff << std::endl;
          sampleSize = 0;
          CHECK(AMediaExtractor_seekTo(mExtractor, 0ll, AMEDIAEXTRACTOR_SEEK_CLOSEST_SYNC))
        }
        int64_t presentationTimeUs = AMediaExtractor_getSampleTime(mExtractor);
        CHECK(AMediaCodec_queueInputBuffer(mCodec, bufidx, 0, sampleSize, presentationTimeUs, 0))
        AMediaExtractor_advance(mExtractor);
      } else {
        Logger::debug(MP4tag) << "AMediaCodec_dequeueInputBuffer failed " << bufidx << std::endl;
      }
    }

    /*	Output	*/
    {
      AMediaCodecBufferInfo bufferInfo;
      ssize_t outputBufferIndex = AMediaCodec_dequeueOutputBuffer(mCodec, &bufferInfo, 0);
      if (outputBufferIndex >= 0) {
        size_t bufferSize = 0;
        uint8_t* outputBuffer = AMediaCodec_getOutputBuffer(mCodec, outputBufferIndex, &bufferSize);
#ifdef USE_CUDA
        cudaMemcpy(data, outputBuffer, bufferSize, cudaMemcpyDefault);
#else
        uint8_t* out_ptr = data;
        // luminance plane
        uint8_t* ptrY = outputBuffer;
        for (auto i = 0; i < mHeight; i++) {
          memcpy(out_ptr, ptrY + i * mStride, mWidth);
          out_ptr += mWidth;
        }

        // chroma planes
        uint8_t* ptrUV = outputBuffer + mHeight * mStride;
        /* assume PixelFormat::YV12 chroma stride is mStride/2 for the moment */
        if ((mPixelFormat == PixelFormat::NV12) || (mPixelFormat == PixelFormat::YV12)) {
          for (auto i = 0; i < mHeight / 2; i++) {
            memcpy(out_ptr, ptrUV + i * mStride, mWidth);
            out_ptr += mWidth;
          }
        }
#endif
        date = bufferInfo.presentationTimeUs;
        CHECK(AMediaCodec_releaseOutputBuffer(mCodec, outputBufferIndex, false))
        gotFrame = true;
        if (bufferInfo.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
          Logger::info(MP4tag) << "AMediaCodec_dequeueOutputBuffer EndOfFile detected" << std::endl;
          return ReadStatus::fromCode<ReadStatusCode::EndOfFile>();
        }
      } else if (outputBufferIndex == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
        Logger::debug(MP4tag) << "AMediaCodec_dequeueOutputBuffer AMEDIACODEC_INFO_TRY_AGAIN_LATER. " << std::endl;
        //        return ReadStatus::fromCode<ReadStatusCode::TryAgain>();
      } else {
        Logger::info(MP4tag) << "AMediaCodec_dequeueOutputBuffer failed " << outputBufferIndex << std::endl;
      }
    }
  }

  Logger::verbose(MP4tag) << "reader " << id << " readFrame at time " << date << " after " << nb_iter << " iterations"
                          << std::endl;
  return ReadStatus::OK();
}

Status Mp4Reader::seekFrame(frameid_t /*targetFrame*/) {
  Logger::warning(MP4tag) << "seekFrame not implemented" << std::endl;
  return Status::OK();
}

ReadStatus Mp4Reader::readSamples(size_t, Audio::Samples&) {
  Logger::warning(MP4tag) << "readFrameAudioOnly not implemented" << std::endl;
  return ReadStatus::OK();
}

}  // namespace Input
}  // namespace VideoStitch
