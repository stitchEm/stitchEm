// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "panoStitcherBase.hpp"

#include "gpu/buffer.hpp"
#include "gpu/memcpy.hpp"
#include "common/angles.hpp"
#include "common/container.hpp"
#include "image/unpack.hpp"
#include "photoTransform.hpp"
#include "stitchOutput/stitchOutput.hpp"
#include "stitchOutput/stereoOutput.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/postprocessor.hpp"
#include "libvideostitch/profile.hpp"
#include "libvideostitch/gpu_device.hpp"

#include <cassert>
#include <fstream>
#include <memory>
#include <sstream>

namespace VideoStitch {
namespace Core {
template <typename Output>
PanoStitcherImplBase<Output>::PanoStitcherImplBase(const std::string& nameValue, const PanoDefinition& panoValue,
                                                   Eye eyeValue)
    : name(nameValue), pano(&panoValue), eye(eyeValue) {}

template <typename Output>
PanoStitcherImplBase<Output>::~PanoStitcherImplBase() {
  deleteAllValues(photoTransforms);
  for (auto stream : streams) {
    stream.second.destroy();
  }
  streams.clear();
}

template <typename Output>
const DevicePhotoTransform& PanoStitcherImplBase<Output>::getPhotoTransform(readerid_t inputId) const {
  assert(0 <= inputId && inputId < (readerid_t)photoTransforms.size());
  return *photoTransforms.at(inputId);
}

template <typename Output>
PanoSurface& acquireFrame(Eye, Output* output, mtime_t date);
template <>
PanoSurface& acquireFrame(Eye, StitchOutput* output, mtime_t date) {
  return output->pimpl->acquireFrame(date);
}
template <>
PanoSurface& acquireFrame(Eye eye, StereoOutput* output, mtime_t date) {
  if (eye == LeftEye) {
    return output->pimpl->acquireLeftFrame(date);
  } else {
    return output->pimpl->acquireRightFrame(date);
  }
}

template <typename Output>
Status pushVideo(Eye, Output* output, mtime_t date);
template <>
Status pushVideo(Eye, StitchOutput* output, mtime_t date) {
  return output->pimpl->pushVideo(date);
}
template <>
Status pushVideo(Eye eye, StereoOutput* output, mtime_t date) {
  return output->pimpl->pushVideo(date, eye);
}

/**
 *
 * - do not load concurrently (avoid concurrent accesses to disk)
 * - do not write while loading
 * - loading, transmitting and mapping are independant for two different images
 * - merging requires the previous and current images to be mapped.
 * - keep frames independant for the moment, i.e. do not start images for next frames during current frame.
 *
 * Image1 | load | map | merge     |                                                           | load |
 * Image2 |      | load     | map   | merge    |
 * Image3 |                 | load | map |XXXXX| merge    |
 * Image4 |                        | load | map |XXXXXXXXX| merge  |
 * Image5 |                               | load       | map   |XXX| merge  |
 *
 *                                                                          | readback | write |
 *
 * The main thread orchestrates the loading, delegating asynchronous logic to cuda streams.
 */
template <typename Output>
Status PanoStitcherImplBase<Output>::stitch(mtime_t date, frameid_t frame, PostProcessor* postprocessor,
                                            std::map<readerid_t, Input::PotentialFrame> inputBuffers,
                                            std::map<readerid_t, Input::VideoReader*> readers,
                                            std::map<readerid_t, PreProcessor*> preprocessors, Output* output) {
  auto stitchProcess = [&]() -> Status {
    FAIL_RETURN(GPU::useDefaultBackendDevice());
    PanoSurface& surf = acquireFrame(eye, output, date);
    Status status = merge(frame, inputBuffers, readers, preprocessors, surf);
    if (status.ok()) {
      if (postprocessor) {
        status = postprocessor->process(surf.pimpl->buffer, *pano, frame, surf.pimpl->stream);
      }
    }
    if (!status.ok()) {
      GPU::memsetToZeroAsync(surf.pimpl->buffer, (size_t)(pano->getWidth() * pano->getHeight() * 4),
                             surf.pimpl->stream);
      // still push the output, to release the panorama frame buffer
      pushVideo(eye, output, date);
    } else {
      status = pushVideo(eye, output, date);
    }
    surf.pimpl->stream.flush();

    return status;
  };

  const Status success = stitchProcess();
  if (!success.ok()) {
    // Signal an error to the output so that it does not wait forever for the next frame.
    Logger::get(Logger::Warning) << "Failed to stitch. Skipping output for frame " << frame << "." << std::endl;
  }

  return success;
}

template <typename Output>
Status PanoStitcherImplBase<Output>::createTransforms(const std::map<readerid_t, Input::VideoReader*>& readers) {
  deleteAllValues(photoTransforms);
  for (auto reader : readers) {
    const InputDefinition& inputDef = pano->getInput(reader.second->id);
    DevicePhotoTransform* photoTransform = DevicePhotoTransform::create(inputDef);
    if (!photoTransform) {
      deleteAllValues(photoTransforms);
      return {Origin::Stitcher, ErrType::SetupFailure,
              "Cannot create transformation for input " + std::to_string(reader.second->id)};
    }
    photoTransforms[reader.second->id] = photoTransform;
  }
  return Status::OK();
}

template <typename Output>
Status PanoStitcherImplBase<Output>::setup(const ImageMergerFactory& mergerFactory,
                                           const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory,
                                           const std::map<readerid_t, Input::VideoReader*>& readers,
                                           const StereoRigDefinition* rig) {
  FAIL_RETURN(GPU::useDefaultBackendDevice());

  if (streams.size() > 0) {
    return {Origin::Stitcher, ErrType::ImplementationError, "Stitcher has already been setup"};
  }

  // Create streams for async operations.
  for (auto r : readers) {
    auto potStream = GPU::Stream::create();
    if (!potStream.ok()) {
      streams.clear();
      return potStream.status();
    }
    streams[r.second->id] = potStream.value();
  }

  PROPAGATE_FAILURE_STATUS(createTransforms(readers));

  // Setup the implementation.
  return setupImpl(mergerFactory, warperFactory, flowFactory, readers, rig);
}

template <typename Output>
Status PanoStitcherImplBase<Output>::redoSetup(const PanoDefinition& newPano, const ImageMergerFactory& mergerFactory,
                                               const ImageWarperFactory& warperFactory,
                                               const ImageFlowFactory& flowFactory,
                                               std::map<readerid_t, Input::VideoReader*> readers,
                                               const StereoRigDefinition* rig) {
  pano = &newPano;
  FAIL_RETURN(GPU::useDefaultBackendDevice());

  PROPAGATE_FAILURE_STATUS(createTransforms(readers));

  // redoSetup the implementation.
  return redoSetupImpl(mergerFactory, warperFactory, flowFactory, readers, rig);
}

template <typename Output>
ChangeCompatibility PanoStitcherImplBase<Output>::worstCompatibility(ChangeCompatibility a, ChangeCompatibility b) {
  switch (a) {
    case IncompatibleChanges:
      return IncompatibleChanges;
    case SetupIncompatibleChanges:
      return b == IncompatibleChanges ? IncompatibleChanges : SetupIncompatibleChanges;
    case SetupCompatibleChanges:
      return b;
  }
  return IncompatibleChanges;
}

template <typename Output>
void PanoStitcherImplBase<Output>::applyRotation(double yaw, double pitch, double roll) {
  interactivePersp *= Matrix33<double>::fromEulerZXY(degToRad(yaw), degToRad(pitch), degToRad(roll));
}

template <typename Output>
void PanoStitcherImplBase<Output>::resetRotation() {
  interactivePersp = Matrix33<double>();
}

template <typename Output>
Quaternion<double> PanoStitcherImplBase<Output>::getRotation() const {
  double yaw, pitch, roll;
  interactivePersp.toEuler(yaw, pitch, roll);
  return Quaternion<double>::fromEulerZXY(yaw, pitch, roll);
}

// explicit instantiations

template class PanoStitcherImplBase<StitchOutput>;
template class PanoStitcherImplBase<StereoOutput>;

}  // namespace Core
}  // namespace VideoStitch
