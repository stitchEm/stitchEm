// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "flashSync.hpp"

#include "common/thread.hpp"

#include "gpu/2dBuffer.hpp"
#include "gpu/allocator.hpp"
#include "gpu/buffer.hpp"
#include "gpu/hostBuffer.hpp"
#include "gpu/memcpy.hpp"
#include "gpu/stream.hpp"
#include "gpu/surface.hpp"
#include "gpu/image/downsampler.hpp"

#include "image/histogram.hpp"
#include "image/unpack.hpp"

#include "util/registeredAlgo.hpp"

#include "libvideostitch/allocator.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"

#include <memory>
#include <sstream>

namespace VideoStitch {
namespace Synchro {
namespace {
Util::RegisteredAlgo<FlashSyncAlgorithm> registered("flash_synchronization");
}

FlashSyncAlgorithm::FlashSyncAlgorithm(const Ptv::Value* config) : firstFrame(0), lastFrame(1000) {
  if (config != NULL) {
    const Ptv::Value* value = config->has("first_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      firstFrame = (int)value->asInt();
    }
    value = config->has("last_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      lastFrame = (int)value->asInt();
    }
    value = config->has("devices");
    if (value && value->getType() == Ptv::Value::LIST) {
      const std::vector<Ptv::Value*>& devIds = value->asList();
      for (std::vector<Ptv::Value*>::const_iterator d = devIds.begin(); d != devIds.end(); ++d) {
        value = (*d)->has("id");
        if (value && value->getType() == Ptv::Value::INT) {
          devices.push_back((int)value->asInt());
        }
      }
    }
    if (devices.size() == 0) devices.push_back(0);
  }
}

FlashSyncAlgorithm::~FlashSyncAlgorithm() {}

Potential<Ptv::Value> FlashSyncAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                                Util::OpaquePtr**) const {
  std::vector<int> offsetsFrames;
  FAIL_RETURN(doAlign(devices, *pano, offsetsFrames, progress));
  for (readerid_t i = 0; i < (readerid_t)offsetsFrames.size(); ++i) {
    pano->getInput(i).setFrameOffset(offsetsFrames[i]);
  }
  return Potential<Ptv::Value>(Status::OK());
}

const char* FlashSyncAlgorithm::docString =
    "An algorithm that computes frame offsets using the luma histograms to synchronize the inputs.\n"
    "Can be applied pre-calibration.\n"
    "The result is a { \"frames\": list of integer offsets (all >=0, in frames), \"seconds\": list of double offsets "
    "(all >=0.0, in seconds) }\n"
    "which can be used directly as a 'frame_offset' parameter for the 'inputs'.\n";

namespace {
class LumaHistogramTask : public ThreadPool::Task {
 public:
  static Potential<LumaHistogramTask> create(int deviceId, readerid_t inputId, Input::VideoReader* reader,
                                             int64_t width, int64_t height, frameid_t firstFrame, frameid_t lastFrame,
                                             frameid_t& flashFrame, std::vector<Status>& errors,
                                             Util::Algorithm::ProgressReporter* progress) {
    return Potential<LumaHistogramTask>(new LumaHistogramTask(deviceId, inputId, reader, width, height, firstFrame,
                                                              lastFrame, flashFrame, errors, progress));
  }

  ~LumaHistogramTask() { delete reader; }

 private:
  LumaHistogramTask(int deviceId, readerid_t inputId, Input::VideoReader* reader, int64_t width, int64_t height,
                    frameid_t firstFrame, frameid_t lastFrame, frameid_t& flashFrame, std::vector<Status>& errors,
                    Util::Algorithm::ProgressReporter* progress)
      : deviceId(deviceId),
        inputId(inputId),
        reader(reader),
        width(width),
        height(height),
        firstFrame(firstFrame),
        lastFrame(lastFrame),
        flashFrame(flashFrame),
        errors(errors),
        progress(progress) {}

  virtual void run() {
    // cudaSetDevice(deviceId);

    Input::VideoReader::Spec spec = reader->getSpec();

    auto pHostHistograms =
        GPU::HostBuffer<uint32_t>::allocate((lastFrame - firstFrame) * 256, "Luma histograms: host histograms");
    auto pDevHistograms =
        GPU::Buffer<uint32_t>::allocate((lastFrame - firstFrame) * 256, "Luma histograms: device histograms");
    auto pHostBuffer =
        GPU::HostBuffer<unsigned char>::allocate(width * height * sizeof(uint32_t), "Luma histograms: host frame");
    auto pDevBuffer =
        GPU::Buffer<unsigned char>::allocate(width * height * sizeof(uint32_t), "Luma histograms: device frame");
    auto pSurf = Core::OffscreenAllocator::createSourceSurface(width, height, "Luma histograms: device frame");
    auto pGray = GPU::Buffer2D::allocate(width, height, "Luma histograms: grayscale image");
    auto pGrayDown = GPU::Buffer2D::allocate(width / 4, height / 4, "Luma histograms: downscaled grayscale image");
    auto pStream = GPU::Stream::create();

    if (!pHostHistograms.ok()) {
      errors.push_back({Origin::Unspecified, ErrType::OutOfResources, "Couldn't allocate host memory"});
      return;
    }
    if (!pDevHistograms.ok() || !pSurf.ok() || !pStream.ok() || !pGray.ok() || !pGrayDown.ok()) {
      errors.push_back({Origin::GPU, ErrType::OutOfResources, "Couldn't allocate GPU memory"});
      return;
    }
    GPU::Stream stream = pStream.value();
    GPU::HostBuffer<uint32_t> hostHistograms = pHostHistograms.value();
    GPU::Buffer<uint32_t> devHistograms = pDevHistograms.value();
    GPU::HostBuffer<unsigned char> hostBuffer = pHostBuffer.value();
    GPU::Buffer<unsigned char> devBuffer = pDevBuffer.value();
    Core::SourceSurface* surf = pSurf.object();
    GPU::Buffer2D grayscale = pGray.value();
    GPU::Buffer2D grayscaleDown = pGrayDown.value();

    for (frameid_t frame = 0; frame < lastFrame - firstFrame; ++frame) {
      unsigned char* origFrame = nullptr;
      switch (reader->getSpec().addressSpace) {
        case Device:
          origFrame = devBuffer.devicePtr();
          break;
        case Host:
          origFrame = hostBuffer.hostPtr();
          break;
      }
      mtime_t date;
      Input::ReadStatus statusRead = reader->readFrame(date, origFrame);
      if (!statusRead.ok()) {
        errors.push_back({Origin::Input, ErrType::RuntimeError, "Luma histograms: could not read the frame"});
        return;
      }

      // transfer to device if needed
      switch (reader->getSpec().addressSpace) {
        case Host:
          GPU::memcpyAsync(devBuffer, hostBuffer.as_const(), stream);
          break;
        case Device:
          break;
      }

      Image::unpackCommonPixelFormat(spec.format, *surf->pimpl->surface, devBuffer.as_const(), spec.width, spec.height,
                                     stream);
      Image::unpackGrayscale(grayscale, *surf->pimpl->surface, spec.width, spec.height, stream);
      Image::downsample(grayscale, grayscaleDown, stream);
      Image::lumaHistogram(grayscaleDown, frame, devHistograms, stream);
      if (frame % 100 == 0) {
        frameid_t analyzedFirstFrame = firstFrame + frame;
        frameid_t analyzedLastFrame = std::min(analyzedFirstFrame + 99, lastFrame - 1);
        std::stringstream ss;
        ss << "Analyzing frames " << analyzedFirstFrame << "..." << analyzedLastFrame << " of source " << inputId
           << " on GPU " << deviceId;
        if (progress && progress->notify(ss.str(), (100.0 * (double)frame) / (double)(lastFrame - firstFrame))) {
          errors.push_back({Origin::SynchronizationAlgorithm, ErrType::OperationAbortedByUser, "Algorithm cancelled"});
        }
      }
    }
    GPU::memcpyAsync(hostHistograms, devHistograms.as_const(), stream);

    stream.synchronize();

    // look for the frame with the highest 95th percentile of luma
    const int64_t cutoff = (80 * width * height) / 100;
    unsigned bestFrame = 0;
    double bestPercentileCutoffMean = 0;

    for (frameid_t frame = 0; frame < (lastFrame - firstFrame); ++frame) {
      uint32_t* histogram = hostHistograms.hostPtr() + 256 * frame;
      int64_t total = 0;
      int64_t mean = 0;

      for (int32_t luma = 255; luma >= 0; --luma) {
        total += histogram[luma];
        mean += histogram[luma] * luma;

        if (total > cutoff) {
          break;
        }
      }

      const double fmean = (double)mean / (double)total;

      if (fmean > bestPercentileCutoffMean) {
        bestPercentileCutoffMean = fmean;
        bestFrame = frame;
      }
    }
    flashFrame = bestFrame + firstFrame;
    return;
  }

  int deviceId;
  readerid_t inputId;
  Input::VideoReader* reader;
  const int64_t width, height;

  const frameid_t firstFrame, lastFrame;
  frameid_t& flashFrame;

  std::vector<Status>& errors;
  Util::Algorithm::ProgressReporter* progress;
};
}  // namespace

Status FlashSyncAlgorithm::doAlign(const std::vector<int>& devices, const Core::PanoDefinition& pano,
                                   std::vector<int>& frames, ProgressReporter* progress) const {
  if (pano.numInputs() < 2) {
    return Status::OK();
  }
  if (pano.numVideoInputs() != pano.numInputs()) {
    return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration, "Some enabled inputs do not have video"};
  }
  std::vector<int> offsetsInFrames;
  offsetsInFrames.resize(pano.numInputs());
  Input::DefaultReaderFactory readerFactory(firstFrame, lastFrame);

  ThreadPool threadPool((int)pano.numInputs());

  // luma histograms of each of the input sequences
  std::vector<frameid_t> flashFrames(pano.numInputs());
  std::vector<Status> errors;
  for (readerid_t in = 0; in < pano.numInputs(); ++in) {
    const Core::InputDefinition& im = pano.getInput(in);
    if (im.getReaderConfig().getType() != Ptv::Value::STRING) {
      return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration, "Malformed reader configuration"};
    }
    Potential<Input::Reader> reader = readerFactory.create(in, im);
    if (!reader.ok()) {
      return {Origin::SynchronizationAlgorithm, ErrType::SetupFailure, "Cannot create readers"};
    }
    Input::VideoReader* videoReader = reader.release()->getVideoReader();
    if (videoReader) {
      Potential<LumaHistogramTask> task =
          LumaHistogramTask::create(devices[in % devices.size()], in, videoReader, im.getWidth(), im.getHeight(),
                                    firstFrame, lastFrame, flashFrames[in], errors, progress);
      if (!task.status().ok()) {
        return task.status();
      }
      threadPool.tryRun(task.release());
    }
  }
  threadPool.waitAll();

  // Find the smallest offset.
  // TODO make top-level error with sub errors as cause
  if (!errors.empty()) {
    return errors[0];
  }
  unsigned minOffset = UINT_MAX;
  for (readerid_t in = 0; in < pano.numInputs(); ++in) {
    const Core::InputDefinition& im = pano.getInput(in);
    unsigned offset = flashFrames[in] + im.getFrameOffset();
    if (offset < minOffset) {
      minOffset = offset;
    }
  }
  for (readerid_t in = 0; in < pano.numInputs(); ++in) {
    const Core::InputDefinition& im = pano.getInput(in);
    frames.push_back(flashFrames[in] + im.getFrameOffset() - minOffset);
  }
  return Status::OK();
}
}  // namespace Synchro
}  // namespace VideoStitch
