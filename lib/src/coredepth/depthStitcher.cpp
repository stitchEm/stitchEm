// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "depthStitcher.hpp"

#include "panoMerger.hpp"

#include "gpu/allocator.hpp"
#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"

#include "common/container.hpp"

#include "libvideostitch/imageMergerFactory.hpp"
#include "libvideostitch/stitchOutput.hpp"
#include "libvideostitch/panoDef.hpp"

namespace VideoStitch {
namespace Core {

template <typename Output>
DepthStitcher<Output>::DepthStitcher(const std::string& name, const PanoDefinition& pano, Eye eye)
    : PanoStitcherImplBase<Output>(name, pano, eye), panoMerger(nullptr) {}

template <typename Output>
DepthStitcher<Output>::~DepthStitcher() {
  deleteAllValues(surfaces);
  for (auto buf : devUnpackTmps) {
    buf.release();
  }
  devUnpackTmps.clear();
  delete panoMerger;
}

template <typename Output>
Status DepthStitcher<Output>::redoSetupImpl(const ImageMergerFactory& /* mergerFactory */,
                                            const ImageWarperFactory& /* warperFactory */,
                                            const ImageFlowFactory& /* flowFactory */,
                                            const std::map<readerid_t, Input::VideoReader*>& /* readers */,
                                            const StereoRigDefinition* /* rigDef */) {
  // TODO input number / size may have changed, redo buffer setup

  return Status::OK();
}

template <typename Output>
Status DepthStitcher<Output>::setupImpl(const ImageMergerFactory& mergerFactory,
                                        const ImageWarperFactory& /* warperFactory */,
                                        const ImageFlowFactory& /* flowFactory */,
                                        const std::map<readerid_t, Input::VideoReader*>& readers,
                                        const StereoRigDefinition*) {
  for (auto reader : readers) {
    const Input::VideoReader::Spec& spec = reader.second->getSpec();
    // TODO handle cubemap
    auto tex = OffscreenAllocator::createSourceSurface(spec.width, spec.height, "InputSurface");
    if (tex.ok()) {
      videoreaderid_t videoInputID = getPano().convertInputIndexToVideoInputIndex(reader.first);
      surfaces[videoInputID] = tex.release();
    } else {
      // TODO leaking already allocated
      return tex.status();
    }

    auto potBuffer = GPU::Buffer<unsigned char>::allocate(spec.frameDataSize, "InputFrame");
    if (potBuffer.ok()) {
      devUnpackTmps.push_back(potBuffer.value());
    } else {
      // TODO leaking already allocated
      return potBuffer.status();
    }
  }

  Potential<PanoMerger> cur = mergerFactory.createDepth(getPano());

  // TODO leaking buffers
  FAIL_RETURN(cur.status());

  panoMerger = cur.release();
  return Status::OK();
}

template <typename Output>
Status DepthStitcher<Output>::setupTexArrayAsync(videoreaderid_t inputID, frameid_t /* frame */,
                                                 const Input::PotentialFrame& inputFrame,
                                                 const InputDefinition& /* inputDef */, GPU::Stream& stream,
                                                 Input::VideoReader* reader, const PreProcessor* /* preprocessor */) {
  GPU::Buffer<unsigned char> inputDevBuffer;
  if (inputFrame.status.ok()) {
    switch (inputFrame.frame.addressSpace()) {
      case Host:
        PROPAGATE_FAILURE_STATUS(GPU::memcpyAsync(devUnpackTmps[inputID], inputFrame.frame.hostBuffer(),
                                                  (size_t)reader->getFrameDataSize(), stream));
        inputDevBuffer = devUnpackTmps[inputID];
        break;
      case Device:
        inputDevBuffer = inputFrame.frame.deviceBuffer();
        break;
    }
    PROPAGATE_FAILURE_STATUS(reader->unpackDevBuffer(*surfaces[inputID]->pimpl->surface, inputDevBuffer, stream));
    // if (preprocessor) {
    //   preprocessor->process(frame, getSurface(), inputDef.getWidth(), inputDef.getHeight(), imId, stream);
    // }
  } else {
    // error policy : black frames in case of reader error/EOF
    // PROPAGATE_FAILURE_STATUS(GPU::memsetToZeroAsync(devOutBuf, inputDef.getWidth() * inputDef.getHeight() * 4,
    // stream));
  }

  return Status::OK();
}

template <typename Output>
Status DepthStitcher<Output>::merge(frameid_t frame, const std::map<readerid_t, Input::PotentialFrame>& inputFrames,
                                    const std::map<readerid_t, Input::VideoReader*>& readers,
                                    const std::map<readerid_t, PreProcessor*>& preprocessors, PanoSurface& pano) {
  GPU::Stream stream = pano.pimpl->stream;

  GPU::memsetToZeroAsync(pano.pimpl->buffer, stream);

  for (videoreaderid_t inputID = 0; inputID < getPano().numVideoInputs(); inputID++) {
    Input::VideoReader* reader = readers.at(inputID);
    const InputDefinition& inputDef = getPano().getInput(inputID);
    PreProcessor* preprocessor = preprocessors.at(inputID);

    FAIL_RETURN(
        setupTexArrayAsync(inputID, frame, inputFrames.find(inputID)->second, inputDef, stream, reader, preprocessor));
  }

  panoMerger->computeAsync(getPano(), pano, surfaces, stream);

  // TODO create event...

  return stream.synchronize();
}

template <typename Output>
ChangeCompatibility DepthStitcher<Output>::getCompatibility(const PanoDefinition& /* pano */,
                                                            const PanoDefinition& /* newPano */) const {
  // TODO
  return SetupCompatibleChanges;
}

// explicit instantiations
template class DepthStitcher<StitchOutput>;

}  // namespace Core
}  // namespace VideoStitch
