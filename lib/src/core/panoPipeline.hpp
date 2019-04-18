// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videoPipeline.hpp"
#include "panoStitcherBase.hpp"

#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/matrix.hpp"
#include "libvideostitch/frame.hpp"
#include "libvideostitch/stitchOutput.hpp"

namespace VideoStitch {
namespace Core {

class PanoDefinition;

/**
 * Implementation of panoramic video pipeline.
 */
class PanoPipeline : public VideoPipeline {
 public:
  typedef StitchOutput Output;
  typedef PanoDeviceDefinition DeviceDefinition;

  virtual ~PanoPipeline();

  /**
   * The PanoPipeline is invalid until it has been setup().
   * pano must live until the PanoStitcher is destroyed.
   */
  static Potential<PanoPipeline> createPanoPipeline(PanoStitcherImplBase<StitchOutput>* stitcher,
                                                    const std::vector<Input::VideoReader*>& readers,
                                                    const std::vector<PreProcessor*>& preprocs,
                                                    PostProcessor* postproc);

  /**
   * Stitches a full panorama image.
   */
  Status stitch(mtime_t date, FrameRate frameRate, std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                StitchOutput* output);

  /**
   * Stitches a full panorama image and extracts the input frames.
   */
  Status stitchAndExtract(mtime_t date, FrameRate frameRate, std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                          StitchOutput* output, std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo);

  /**
   * Returns the photo transform for a given input.
   */
  const DevicePhotoTransform& getPhotoTransform(readerid_t inputId) const {
    return stitcher->getPhotoTransform(inputId);
  }

  /**
   * Computes the level of compatibility between two PanoDefinitions.
   */
  virtual ChangeCompatibility getCompatibility(const PanoDefinition& pano, const PanoDefinition& newPano) const {
    return stitcher->getCompatibility(pano, newPano);
  }

  /**
   * Setup the panoStitcher.
   * Returns true on success.
   */
  Status setup(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
               const ImageFlowFactory& flowFactory, const StereoRigDefinition*);

  /**
   * Re-setup the panoStitcher. This is less expensive that destroying and setting up, but is not compatible with all
   * changes (see controller::resetPano()); Returns true on success.
   */
  Status redoSetup(PanoDefinition& pano, const ImageMergerFactory& mergerFactory,
                   const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory,
                   const StereoRigDefinition*) {
    return stitcher->redoSetup(pano, mergerFactory, warperFactory, flowFactory, readers);
  }

  void setPano(const PanoDefinition& p) { stitcher->setPano(p); }

  virtual void applyRotation(double yaw, double pitch, double roll) { stitcher->applyRotation(yaw, pitch, roll); }

  virtual void resetRotation() { stitcher->resetRotation(); }
  virtual Quaternion<double> getRotation() const { return stitcher->getRotation(); }

  const Matrix33<double>& getInteractivePersp() const { return stitcher->getInteractivePersp(); }

 private:
  PanoPipeline(PanoStitcherImplBase<StitchOutput>*, const std::vector<Input::VideoReader*>&,
               const std::vector<PreProcessor*>&, PostProcessor*);
  PanoStitcherImplBase<StitchOutput>* stitcher;

  PanoPipeline& operator=(const PanoPipeline&);
};

}  // namespace Core
}  // namespace VideoStitch
