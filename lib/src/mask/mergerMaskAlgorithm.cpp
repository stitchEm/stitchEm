// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mergerMaskAlgorithm.hpp"
#include "mergerMaskProgress.hpp"

#include "common/container.hpp"
#include "gpu/memcpy.hpp"
#include "util/registeredAlgo.hpp"

#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/mergerMaskDef.hpp"
#include "libvideostitch/ptv.hpp"

namespace VideoStitch {
namespace MergerMask {

namespace {
Util::RegisteredAlgo<MergerMaskAlgorithm> registered("mask");
}

const char* MergerMaskAlgorithm::docString =
    "An algorithm that optimizes the blending mask and blending order of the input images\n";

MergerMaskAlgorithm::MergerMaskAlgorithm(const Ptv::Value* config)
    : mergerMaskConfig(config),
      rigDef(config->has("rig") ? Core::StereoRigDefinition::create(*config->has("rig")) : nullptr) {}

MergerMaskAlgorithm::~MergerMaskAlgorithm() { deleteAllValues(readers); }

Potential<Ptv::Value> MergerMaskAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                                 Util::OpaquePtr**) const {
  Input::DefaultReaderFactory readerFactory(0, NO_LAST_FRAME);
  deleteAllValues(readers);
  for (readerid_t in = 0; in < pano->numInputs(); ++in) {
    Potential<Input::Reader> reader = readerFactory.create(in, pano->getInput(in));
    FAIL_CAUSE(reader.status(), Origin::BlendingMaskAlgorithm, ErrType::SetupFailure, "Could not create input readers");
    Input::VideoReader* videoReader = reader.release()->getVideoReader();
    if (videoReader) {
      readers[videoReader->id] = videoReader;
    }
  }

  MergerMaskProgress mergerMaskProgress(progress, readers.size(), mergerMaskConfig.useBlendingOrder(),
                                        mergerMaskConfig.useSeam());
  auto potMergerMask = MergerMask::create(readers, *pano, rigDef, mergerMaskConfig, mergerMaskProgress);

  FAIL_RETURN(potMergerMask.status());

  GPU::UniqueBuffer<uint32_t> inputIndexPixelBuffer;
  std::vector<size_t> masksOrder;
  for (readerid_t i = 0; i < pano->numInputs(); i++) {
    masksOrder.push_back(i);
  }
  FAIL_CAUSE(inputIndexPixelBuffer.alloc(pano->getWidth() * pano->getHeight(), "Merger Mask Algorithm"),
             Origin::BlendingMaskAlgorithm, ErrType::SetupFailure, "Cannot allocate input buffer");

  std::unique_ptr<MergerMask> mergerMask(potMergerMask.release());
  // Call the main algorithm to find the mask and blending order
  FAIL_CAUSE(mergerMask->getMergerMasks(inputIndexPixelBuffer.borrow(), masksOrder), Origin::BlendingMaskAlgorithm,
             ErrType::RuntimeError, "Cannot find the mask and blending order");

  auto hostBuffer = GPU::HostBuffer<uint32_t>::allocate(pano->getWidth() * pano->getHeight(), "Merger Mask Algorithm");
  FAIL_CAUSE(hostBuffer.status(), Origin::BlendingMaskAlgorithm, ErrType::SetupFailure,
             "Cannot allocate readback buffer");

  FAIL_CAUSE(GPU::memcpyBlocking(hostBuffer.value(), inputIndexPixelBuffer.borrow_const()),
             Origin::BlendingMaskAlgorithm, ErrType::RuntimeError, "Cannot read back buffer");

  // Setup the result into the pano
  Core::MergerMaskDefinition& mergerMaskDef = pano->getMergerMask();
  mergerMaskDef.setWidth(pano->getWidth());
  mergerMaskDef.setHeight(pano->getHeight());
  mergerMaskDef.setMasksOrder(masksOrder);
  mergerMaskDef.setInputScaleFactor(mergerMaskDef.getInputScaleFactor());
  std::map<videoreaderid_t, std::string> maskInputSpaces;
  FAIL_RETURN(MergerMask::transformMasksFromOutputToEncodedInputSpace(
      *pano, readers, inputIndexPixelBuffer.borrow_const(), maskInputSpaces));

  FAIL_RETURN(mergerMaskDef.setInputIndexPixelData(maskInputSpaces, pano->getWidth(), pano->getHeight(),
                                                   mergerMaskConfig.getFrames()[0]));
  mergerMaskDef.setEnabled(true);
  FAIL_RETURN(hostBuffer.value().release());
  return Potential<Ptv::Value>(Status::OK());
}

}  // namespace MergerMask
}  // namespace VideoStitch
