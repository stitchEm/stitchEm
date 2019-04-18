// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videoPipeline.hpp"

#include "buffer.hpp"

#include "stitchOutput/stitchOutput.hpp"
#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/stream.hpp"
#include "image/unpack.hpp"

#include "libvideostitch/gpu_device.hpp"

namespace VideoStitch {
namespace Core {

VideoPipeline::VideoPipeline(const std::vector<Input::VideoReader*>& readers,
                             const std::vector<PreProcessor*>& preprocs, PostProcessor* postproc)
    : postproc(postproc) {
  GPU::useDefaultBackendDevice();
  int i = 0;
  for (auto r : readers) {
    PreProcessor* pre = nullptr;
    if (!preprocs.empty()) {
      pre = preprocs[i++];
    }
    this->preprocs[r->id] = pre;
    this->readers[r->id] = r;
  }
}

VideoPipeline::~VideoPipeline() {
  for (auto s : streams) {
    s.second.destroy();
  }
  for (auto idb : inputDeviceBuffers) {
    idb.second.release();
  }
}

Potential<VideoPipeline> VideoPipeline::createVideoPipeline(const std::vector<Input::VideoReader*>& readers,
                                                            const std::vector<PreProcessor*>& preprocs,
                                                            PostProcessor* postproc) {
  VideoPipeline* ret = new VideoPipeline(readers, preprocs, postproc);

  Status initStatus = ret->init();

  if (!initStatus.ok()) {
    delete ret;
    ret = nullptr;
    return initStatus;
  }

  return ret;
}

Status VideoPipeline::init() {
  for (auto r : readers) {
    auto stream = GPU::Stream::create();
    if (!stream.ok()) {
      return stream.status();
    }
    streams[r.first] = stream.value();
    switch (r.second->getSpec().addressSpace) {
      case Host: {
        // device frames in original format
        auto idb = GPU::Buffer<unsigned char>::allocate(r.second->getFrameDataSize(), "Input Frames");

        if (!idb.ok()) {
          return idb.status();
        }
        inputDeviceBuffers[r.first] = idb.value();
        break;
      }
      case Device:
        break;
    }
  }

  return Status::OK();
}

Status VideoPipeline::extract(mtime_t date, FrameRate frameRate,
                              std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                              std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());
  std::vector<std::pair<videoreaderid_t, GPU::Surface&>> frames;
  for (auto extract : extracts) {
    // TODO getStitchBuffer should return std::pair of buffer and stream
    GPU::Stream st;
    GPU::Surface& rbB = extract->pimpl->acquireFrame(date, st);
    FAIL_RETURN(
        extraction(inputBuffers.find(extract->pimpl->getSource())->second, extract->pimpl->getSource(), rbB, st));
    extract->pimpl->pushVideo(date);

    std::pair<videoreaderid_t, GPU::Surface&> p((int)frames.size(), rbB);
    frames.push_back(p);
  }

  if (algo != nullptr) {
    algo->onFrame(frames, date, frameRate);
  }

  return Status::OK();
}

Status VideoPipeline::extract(mtime_t date, std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                              ExtractOutput* extract) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());
  GPU::Stream stream;
  GPU::Surface& readbackDevBuffer = extract->pimpl->acquireFrame(date, stream);
  FAIL_RETURN(extraction(inputBuffers.find(extract->pimpl->getSource())->second, extract->pimpl->getSource(),
                         readbackDevBuffer, stream));
  extract->pimpl->pushVideo(date);

  return Status::OK();
}

Status VideoPipeline::extraction(Input::PotentialFrame inputBuffer, int source, GPU::Surface& readbackDevBuffer,
                                 GPU::Stream stream) {
  const Input::VideoReader* reader = readers[source];
  const Input::VideoReader::Spec& spec = reader->getSpec();
  GPU::Buffer<unsigned char> inputDevBuffer;
  if (inputBuffer.status.ok()) {
    switch (inputBuffer.frame.addressSpace()) {
      case Host:
        FAIL_RETURN(
            GPU::memcpyAsync(inputDeviceBuffers[source], inputBuffer.frame.hostBuffer(), spec.frameDataSize, stream));
        inputDevBuffer = inputDeviceBuffers[source];
        break;
      case Device:
        inputDevBuffer = inputBuffer.frame.deviceBuffer();
        break;
    }
    FAIL_RETURN(Image::unpackCommonPixelFormat(spec.format, readbackDevBuffer, inputDevBuffer, spec.width, spec.height,
                                               stream));
  } else {
    // error policy : black frames in case of reader error/EOF
    // XXX TODO FIXME
    // GPU::memsetToZeroAsync(readbackDevBuffer,
    //                       spec.width * spec.height * 4,
    //                       stream);
  }
  return Status::OK();
}
}  // namespace Core
}  // namespace VideoStitch
