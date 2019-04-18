// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/profile.hpp"

#include <sstream>
#include <algorithm>

namespace VideoStitch {
namespace Output {

class VS_EXPORT MultiTimer {
 public:
  void registerEvent() {
    if (startedRecording) {
      microsecondsPerEvent.push_back(timer.elapsed());
    }

    startedRecording = true;
    timer.reset();
  }

  size_t nbEvent() { return microsecondsPerEvent.size(); }

  std::tuple<size_t, double, double, double> getProfilingInformation() {
    std::vector<uint64_t> profCopy = microsecondsPerEvent;

    if (profCopy.empty()) {
      return std::make_tuple(0, 0, 0, 0);
    }

    auto calcMedianFPS = [&profCopy]() -> double {
      sort(profCopy.begin(), profCopy.end());

      if (profCopy.size() % 2 == 0) {
        return 1000000. / (((double)profCopy[profCopy.size() / 2 - 1] + (double)profCopy[profCopy.size() / 2]) / 2);
      } else {
        return 1000000. / (double)profCopy[profCopy.size() / 2];
      }
    };

    auto calcMeanFPS = [&profCopy]() -> double {
      double sum = 0;
      for (auto t : profCopy) {
        sum += (double)t;
      }
      return (1000000. * (double)profCopy.size() / sum);
    };

    auto calcVariance = [&profCopy](double mean) -> double {
      double temp = 0;
      for (uint64_t t : profCopy) {
        double td = 1000000. / (double)t;
        temp += (mean - td) * (mean - td);
      }
      return sqrt(temp / (double)profCopy.size());
    };

    auto mean = calcMeanFPS();
    auto variance = calcVariance(mean);
    auto median = calcMedianFPS();
    return std::make_tuple(profCopy.size(), mean, median, variance);
  }

  void reset() {
    microsecondsPerEvent.clear();
    startedRecording = false;
  }

 private:
  Util::SimpleTimer timer;
  bool startedRecording = false;
  std::vector<uint64_t> microsecondsPerEvent;
};

class ProfilingWriter : public VideoWriter {
 public:
  ProfilingWriter(const std::string& name, unsigned width, unsigned height, FrameRate framerate)
      : Output(name), VideoWriter(width, height, framerate, VideoStitch::PixelFormat::YV12) {
    outputFrameCopy = malloc(getExpectedFrameSize());
  }
  ProfilingWriter(const std::string& name, unsigned width, unsigned height, FrameRate framerate,
                  VideoStitch::PixelFormat fmt)
      : Output(name), VideoWriter(width, height, framerate, fmt, Device) {
    outputFrameCopy = NULL;
  }

  void pushVideo(const Frame& video) {
    // can't simulate realistic usage, but at the very least, the output writer has to copy the data once
    if (outputFrameCopy) {
      nvtxRangeId_t nvtxId = nvtxRangeStartA("prof_mcpy");
      memcpy(outputFrameCopy, video.planes[0], video.width * video.height);
      nvtxRangeEnd(nvtxId);
    }
    multiTimer.registerEvent();
  }

  void reset() { multiTimer.reset(); }

  double getFps() {
    auto prof = multiTimer.getProfilingInformation();
    return std::get<1>(prof);
  }

  ~ProfilingWriter() {
    auto prof = multiTimer.getProfilingInformation();
    std::stringstream message;
    message << "Profiling output writer measured time for " << std::get<0>(prof)
            << " frames. Mean: " << std::get<1>(prof) << " fps, median: " << std::get<2>(prof)
            << " fps, var: " << std::get<3>(prof) << " fps" << std::endl;
    Logger::get(Logger::Error) << message.str();

    if (outputFrameCopy) {
      free(outputFrameCopy);
    }
  }

 private:
  MultiTimer multiTimer;
  void* outputFrameCopy;
};

}  // namespace Output
}  // namespace VideoStitch
