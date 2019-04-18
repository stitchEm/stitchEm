// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "gpu/buffer.hpp"
#include "gpu/hostBuffer.hpp"
#include "gpu/stream.hpp"
#include "input/inputFrame.hpp"

#include "libvideostitch/audio.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch/matrix.hpp"
#include "libvideostitch/output.hpp"
#include "libvideostitch/quaternion.hpp"

#include <map>

namespace VideoStitch {

namespace Input {
class Reader;
}
namespace Core {

class AlgorithmOutput;
class ImageMergerFactory;
class ImageWarperFactory;
class ImageFlowFactory;
class PanoDefinition;
class DevicePhotoTransform;
class PanoSurface;
class PostProcessor;
class PreProcessor;
class StereoRigDefinition;

class Buffer;

/**
 * An enum that describes the compatibility of changes between to PanoDefinitions.
 */
enum ChangeCompatibility {
  IncompatibleChanges,       // Changes are completely incompatible (e.g. changing an input).
  SetupIncompatibleChanges,  // Changes are compatible, but setup needs to be redon (e.g. resizing an input).
  // TransformIncompatibleChanges,  // Changes are compatible, but transform needs to be re-initialized (e.g. changing
  // the photo response).
  SetupCompatibleChanges  // CHanges are fully compatible.
};

/**
 * @brief Implementation of PanoStitcher.
 */
template <typename Output>
class PanoStitcherImplBase {
 public:
  PanoStitcherImplBase(const std::string& name, const PanoDefinition&, Eye);
  virtual ~PanoStitcherImplBase();

  /**
   * Stitches a full panorama image.
   * @param output Where to write the output.
   * @param readFrame If false, the stitcher will not read the next frame but will restitch the last frame.
   * @return True on success.
   */
  Status stitch(mtime_t date, frameid_t frame, PostProcessor* postprocessor,
                std::map<readerid_t, Input::PotentialFrame> inputBuffers,
                std::map<readerid_t, Input::VideoReader*> readers, std::map<readerid_t, PreProcessor*> preprocessors,
                Output* output);

  /**
   * Returns the photo transform for a given input.
   * @param inputId Input id whose transform to retrieve.
   */
  const DevicePhotoTransform& getPhotoTransform(readerid_t inputId) const;

  /**
   * Computes the level of compatibility between two PanoDefinitions.
   * @param pano reference pano
   * @param newPano new pano
   */
  virtual ChangeCompatibility getCompatibility(const PanoDefinition& pano, const PanoDefinition& newPano) const = 0;

  /**
   * Returns the worst compatibility.
   * @param a compared
   * @param b compared to
   */
  static ChangeCompatibility worstCompatibility(ChangeCompatibility a, ChangeCompatibility b);

  /**
   * The Panostitcher is invalid until it has been setup().
   * pano must live until the PanoStitcher is destroyed.
   */
  PanoStitcherImplBase();

  /**
   * Setup the panoStitcher.
   * @param mergerFactory Used to create mergers. Can be released after the call.
   * Returns true on success.
   */
  Status setup(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
               const ImageFlowFactory& flowFactory, const std::map<readerid_t, Input::VideoReader*>&,
               const StereoRigDefinition* rig = NULL);

  /**
   * Re-setup the panoStitcher. This is less expensive that destroying and setting up, but is not compatible with all
   * changes (see controller::resetPano()); Returns true on success.
   */
  Status redoSetup(const PanoDefinition&, const ImageMergerFactory& mergerFactory,
                   const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory,
                   const std::map<readerid_t, Input::VideoReader*>, const StereoRigDefinition* = NULL);

  // Getters for implementors.

  const PanoDefinition& getPano() const { return *pano; }
  void setPano(const PanoDefinition& p) { pano = &p; }

  Eye getEye() const { return eye; }

  GPU::Stream getStreamForInput(readerid_t inputId) { return streams[inputId]; }

  virtual void applyRotation(double yaw, double pitch, double roll);
  virtual void resetRotation();
  virtual Quaternion<double> getRotation() const;

  const Matrix33<double>& getInteractivePersp() const { return interactivePersp; }

 private:
  /**
   * Actual stitching work.
   */
  virtual Status merge(frameid_t frame, const std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                       const std::map<readerid_t, Input::VideoReader*>& readers,
                       const std::map<readerid_t, PreProcessor*>& preprocessors, PanoSurface& pano) = 0;

  virtual Status setupImpl(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                           const ImageFlowFactory& flowFactory, const std::map<readerid_t, Input::VideoReader*>&,
                           const StereoRigDefinition*) = 0;

  virtual Status redoSetupImpl(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                               const ImageFlowFactory& flowFactory, const std::map<readerid_t, Input::VideoReader*>&,
                               const StereoRigDefinition*) = 0;

  /**
   * Creates the transforms.
   * @return false on error.
   */
  Status createTransforms(const std::map<readerid_t, Input::VideoReader*>&);

 private:
  const std::string name;
  const PanoDefinition* pano;
  std::map<readerid_t, GPU::Stream> streams;
  std::map<readerid_t, DevicePhotoTransform*> photoTransforms;
  Matrix33<double> interactivePersp;  // interactive yaw/pitch/roll
  Eye eye;
  PanoStitcherImplBase& operator=(const PanoStitcherImplBase&);
};
}  // namespace Core
}  // namespace VideoStitch
