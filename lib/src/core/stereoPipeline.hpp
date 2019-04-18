// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "videoPipeline.hpp"
#include "panoStitcherBase.hpp"

#include "libvideostitch/gpu_device.hpp"
#include "libvideostitch/matrix.hpp"
#include "libvideostitch/stitchOutput.hpp"

#include <thread>

namespace VideoStitch {
namespace Core {

class PanoDefinition;

/**
 * Implementation of a stereoscopic video pipeline.
 *
 * Forward asynchronously the request to the left and right
 * Stitchers.
 */
class StereoPipeline : public VideoPipeline {
 public:
  typedef StereoOutput Output;
  typedef StereoDeviceDefinition DeviceDefinition;

  virtual ~StereoPipeline();

  static Potential<StereoPipeline> createStereoPipeline(PanoStitcherImplBase<StereoOutput>* left,
                                                        PanoStitcherImplBase<StereoOutput>* right,
                                                        const std::vector<Input::VideoReader*>& readers,
                                                        const std::vector<PreProcessor*>& preprocs,
                                                        PostProcessor* postproc);
  /**
   * Stitches a full stereoscopic panorama image.
   * @param output Where to write the output.
   * @param readFrame If false, the stitcher will not read the next frame but will restitch the last frame.
   * @return True on success.
   */
  virtual Status stitch(mtime_t date, frameid_t frame, std::map<readerid_t, Input::PotentialFrame>& inputBuffers,
                        StereoOutput* output);

  /**
   * Stitches a full panorama image and extracts the input frames.
   * @param output Where to write the panorama.
   * @param extracts Which frames to write and where to write them.
   * @return false on failure.
   */
  virtual Status stitchAndExtract(mtime_t date, FrameRate frameRate,
                                  std::map<readerid_t, Input::PotentialFrame>& inputBuffers, StereoOutput* output,
                                  std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo);

  /**
   * Computes the level of compatibility between two PanoDefinitions.
   * @param pano reference pano
   * @param newPano new pano
   */
  virtual ChangeCompatibility getCompatibility(const PanoDefinition& pano, const PanoDefinition& newPano) const {
    return leftStitcher->getCompatibility(pano, newPano);
  }

  virtual void applyRotation(double yaw, double pitch, double roll) {
    leftStitcher->applyRotation(yaw, pitch, roll);
    rightStitcher->applyRotation(yaw, pitch, roll);
  }

  virtual void resetRotation() {
    leftStitcher->resetRotation();
    rightStitcher->resetRotation();
  }
  virtual Quaternion<double> getRotation() const { return leftStitcher->getRotation(); }

  /**
   * Setup the panoStitcher.
   * @param mergerFactory Used to create mergers. Can be released after the call.
   * Returns true on success.
   */
  Status setup(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
               const ImageFlowFactory& flowFactory, const StereoRigDefinition*);

  /**
   * Re-setup the panoStitcher. This is less expensive that destroying and setting up, but is not compatible with all
   * changes (see controller::resetPano()); Returns true on success.
   */
  Status redoSetup(const PanoDefinition& pano, const ImageMergerFactory& mergerFactory,
                   const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory,
                   const StereoRigDefinition* rig) {
    FAIL_RETURN(rightStitcher->redoSetup(pano, mergerFactory, warperFactory, flowFactory, readers, rig));
    return leftStitcher->redoSetup(pano, mergerFactory, warperFactory, flowFactory, readers, rig);
  }

  void setPano(const PanoDefinition& p) {
    leftStitcher->setPano(p);
    rightStitcher->setPano(p);
  }

  const Matrix33<double>& getInteractivePersp() const { return leftStitcher->getInteractivePersp(); }

 private:
  StereoPipeline(PanoStitcherImplBase<StereoOutput>* left, PanoStitcherImplBase<StereoOutput>* right,
                 const std::vector<Input::VideoReader*>&, const std::vector<PreProcessor*>&, PostProcessor*);

  PanoStitcherImplBase<StereoOutput>*leftStitcher, *rightStitcher;
  std::thread leftThread, rightThread;

  StereoPipeline& operator=(const StereoPipeline&);
};
}  // namespace Core
}  // namespace VideoStitch
