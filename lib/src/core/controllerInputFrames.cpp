// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "controllerInputFrames.hpp"

#include "gpu/allocator.hpp"
#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"

#include "image/unpack.hpp"

namespace VideoStitch {
namespace Core {

namespace {
static Status videoLoadStatus(const Input::ReadStatus& readStatus) {
  switch (readStatus.getCode()) {
    case Input::ReadStatusCode::Ok:
      return Status::OK();
    case Input::ReadStatusCode::ErrorWithStatus:
      return Status{Origin::Input, ErrType::RuntimeError, "Could not load input frames", readStatus.getStatus()};
    case Input::ReadStatusCode::EndOfFile:
      return Status{Origin::Input, ErrType::RuntimeError, "Could not load input frame, reader reported end of stream"};
    case Input::ReadStatusCode::TryAgain:
      return Status{Origin::Input, ErrType::RuntimeError, "Could not load input frame, reader starved"};
  }
  assert(false);
  return Status{Origin::Input, ErrType::ImplementationError, "Could not load input frames, unknown error code"};
}
}  // namespace

template <>
Status ControllerInputFrames<PixelFormat::RGBA, uint32_t>::processFrame(Buffer readerFrame,
                                                                        GPU::HostBuffer<uint32_t> readbackDestination,
                                                                        readerid_t readerID) {
  GPU::Buffer<unsigned char> tmp;
  switch (readerFrame.addressSpace()) {
    case Host:
      FAIL_RETURN(GPU::memcpyAsync(devBuffer, readerFrame.hostBuffer(), readerFrame.hostBuffer().byteSize(), stream));
      tmp = devBuffer;
      break;
    case Device:
      tmp = readerFrame.deviceBuffer();
      break;
  }

  auto reader = readerController->getReader(readerID);
  auto spec = reader->getSpec();
  FAIL_RETURN(Image::unpackCommonPixelFormat(spec.format, *surf->pimpl->surface, tmp.as_const(), spec.width,
                                             spec.height, stream));
  FAIL_RETURN(GPU::memcpyAsync(readbackDestination.hostPtr(), *surf->pimpl->surface, stream));

  return stream.synchronize();
}

template <>
Status ControllerInputFrames<PixelFormat::Grayscale, unsigned char>::processFrame(
    Buffer readerFrame, GPU::HostBuffer<unsigned char> readbackDestination, readerid_t readerID) {
  GPU::Buffer<unsigned char> tmp;
  switch (readerFrame.addressSpace()) {
    case Host:
      FAIL_RETURN(GPU::memcpyAsync(devBuffer, readerFrame.hostBuffer(), readerFrame.hostBuffer().byteSize(), stream));
      tmp = devBuffer;
      break;
    case Device:
      tmp = readerFrame.deviceBuffer();
      break;
  }

  auto reader = readerController->getReader(readerID);
  auto spec = reader->getSpec();
  FAIL_RETURN(Image::unpackCommonPixelFormat(spec.format, *surf->pimpl->surface, tmp.as_const(), spec.width,
                                             spec.height, stream));
  // convert from RGBA into grayscale
  Image::unpackGrayscale(grayscale, *surf->pimpl->surface, spec.width, spec.height, stream);
  FAIL_RETURN(GPU::memcpyAsync(readbackDestination, grayscale, stream));
  return stream.synchronize();
}

template <PixelFormat destinationColor, typename readbackType>
Status ControllerInputFrames<destinationColor, readbackType>::load(
    std::map<readerid_t, PotentialValue<GPU::HostBuffer<readbackType>>>& processedFrames, mtime_t* date) {
  std::map<readerid_t, Input::PotentialFrame> framesFromReader;
  processedFrames.clear();
  std::vector<Audio::audioBlockGroupMap_t> audioBlocks;
  mtime_t tempDate = 0;
  Input::MetadataChunk metadata;
  auto loadStatus = readerController->load(tempDate, framesFromReader, audioBlocks, metadata);
  FAIL_RETURN(videoLoadStatus(std::get<0>(loadStatus)));
  if (date) {
    *date = tempDate;
  }

  for (auto inputFrame : framesFromReader) {
    if (inputFrame.second.status.ok()) {
      auto processStatus = processFrame(inputFrame.second.frame, readbackFrames[inputFrame.first], inputFrame.first);
      processedFrames.insert({inputFrame.first, {processStatus, readbackFrames[inputFrame.first]}});
    } else {
      processedFrames.insert({inputFrame.first, videoLoadStatus(inputFrame.second.status)});
    }
  }

  readerController->releaseBuffer(framesFromReader);
  return Status::OK();
}

template <PixelFormat destinationColor, typename readbackType>
Potential<ControllerInputFrames<destinationColor, readbackType>>
ControllerInputFrames<destinationColor, readbackType>::create(const Core::PanoDefinition* pano) {
  auto cif = new ControllerInputFrames();
  Status initStatus = cif->init(pano);

  if (!initStatus.ok()) {
    delete cif;
    return initStatus;
  }

  return cif;
}

template <PixelFormat destinationColor, typename readbackType>
Status ControllerInputFrames<destinationColor, readbackType>::init(const Core::PanoDefinition* pano) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());

  auto potStream = GPU::Stream::create();
  FAIL_RETURN(potStream.status());
  stream = potStream.value();

  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::createDefault());
  auto potReaderController = ReaderController::create(*pano, *audioPipe, new Input::DefaultReaderFactory(0, -1), 0);

  if (!potReaderController.ok()) {
    readerController = nullptr;
    return potReaderController.status();
  }

  readerController = potReaderController.release();

  FAIL_RETURN(readerController->setupReaders());

  // Allocate buffers for readback.
  int64_t height = readerController->getReaderSpec(0).height;
  int64_t width = readerController->getReaderSpec(0).width;
  for (int k = 0; k < (int)pano->numInputs(); ++k) {
    if (!pano->getInput(k).getIsVideoEnabled()) {
      continue;
    }
    const Input::VideoReader::Spec& spec = readerController->getReaderSpec(k);
    auto potReadback = GPU::HostBuffer<readbackType>::allocate(spec.width * spec.height, "ControllerInputFrames");
    FAIL_RETURN(potReadback.status());
    readbackFrames.push_back(potReadback.value());
    if ((spec.height != height) || (spec.width != width)) {
      return {Origin::Input, ErrType::InvalidConfiguration, "All inputs must have the same size"};
    }
  }

  auto potDevBuffer =
      GPU::Buffer<unsigned char>::allocate(readerController->getReaderSpec(0).frameDataSize, "ControllerInputFrames");
  FAIL_RETURN(potDevBuffer.status());
  devBuffer = potDevBuffer.value();

  PotentialValue<GPU::Buffer2D> potGray = GPU::Buffer2D::allocate(
      readerController->getReaderSpec(0).width, readerController->getReaderSpec(0).height, "Grayscale image");
  FAIL_RETURN(potGray.status());
  grayscale = potGray.value();

  Potential<SourceSurface> potSurf = OffscreenAllocator::createSourceSurface(width, height, "ControllerInputFrames");
  FAIL_RETURN(potSurf.status());
  surf = potSurf.release();

  return Status::OK();
}

template <PixelFormat destinationColor, typename readbackType>
ControllerInputFrames<destinationColor, readbackType>::~ControllerInputFrames() {
  if (readerController) {
    readerController->cleanReaders();
    delete readerController;
  }

  if (devBuffer.wasAllocated()) {
    devBuffer.release();
  }

  grayscale.release();

  if (surf) {
    delete surf;
  }

  for (auto hostBuf : readbackFrames) {
    if (hostBuf.byteSize()) {
      hostBuf.release();
    }
  }

  stream.destroy();
}

template <PixelFormat destinationColor, typename readbackType>
Status ControllerInputFrames<destinationColor, readbackType>::seek(frameid_t fr) {
  return readerController->seekFrame(fr);
}

template class ControllerInputFrames<PixelFormat::RGBA, uint32_t>;
template class ControllerInputFrames<PixelFormat::Grayscale, unsigned char>;

}  // namespace Core
}  // namespace VideoStitch
