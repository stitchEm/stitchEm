// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "inputsMap.hpp"
#include "inputsMapCubemap.hpp"

#include "core/panoStitcherBase.hpp"
#ifndef VS_OPENCL
#include "maskinterpolation/inputMaskInterpolation.hpp"
#endif

#include "libvideostitch/matrix.hpp"
#include "libvideostitch/input.hpp"
#include "libvideostitch/output.hpp"

#include <memory>

namespace VideoStitch {
namespace Core {

class ImageMapping;
class ImageMerger;
class ImageWarper;
class InputDefinition;
class OverlayInputDefinition;
class PanoDefinition;
class StereoRigDefinition;
class PreProcessor;
class Transform;
class MergerPair;
class ImageFlow;

/**
 * @brief Implementation of PanoStitcher.
 */
template <typename Output>
class PanoStitcherImplV1 : public PanoStitcherImplBase<Output> {
 public:
  virtual ~PanoStitcherImplV1();

  /**
   * The Panostitcher is invalid until it has been setup().
   * pano must live until the PanoStitcher is destroyed.
   */
  PanoStitcherImplV1(const std::string& name, const PanoDefinition& pano, Eye eye);

 private:
  virtual Status redoSetupImpl(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                               const ImageFlowFactory& flowFactory,
                               const std::map<readerid_t, Input::VideoReader*>& readers, const StereoRigDefinition*);
  virtual Status setupImpl(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                           const ImageFlowFactory& flowFactory,
                           const std::map<readerid_t, Input::VideoReader*>& readers, const StereoRigDefinition*);

  virtual Status merge(frameid_t frame, const std::map<readerid_t, Input::PotentialFrame>& inputFrames,
                       const std::map<readerid_t, Input::VideoReader*>& readers,
                       const std::map<readerid_t, PreProcessor*>& preprocessors, PanoSurface& pano);

  virtual ChangeCompatibility getCompatibility(const PanoDefinition& pano, const PanoDefinition& newPano) const;

  /**
   * Prepares the mappers.
   * @return false on error.
   */
  Status prepareMappers(const StereoRigDefinition* rig);

  /**
   * Common setup code.
   */
  Status setupCommon(const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                     const ImageFlowFactory& flowFactory, const std::map<readerid_t, Input::VideoReader*>& readers,
                     const StereoRigDefinition* rig);
  /**
   * Fills the output buffer with an image in which each pixel is a 32-bit integer,
   * with the i-th bit set if the i-th image contributes to this pixel.
   */
  Status computeSetupImage(const std::map<readerid_t, Input::VideoReader*>&, const StereoRigDefinition*);

  /**
   * Re-compute/load the "key" inputs map at every frame
   */
  Status adaptInputsMap(const frameid_t frameId, std::map<readerid_t, Input::VideoReader*> readers);

  /**
   * Computes the level of compatibility between two InputDefinitions.
   * @param input reference input
   * @param newInput new input
   */
  static ChangeCompatibility getCompatibility(const InputDefinition& input, const InputDefinition& newInput);

  /**
   * Computes the level of compatibility between two OverlayInputDefinitions.
   * @param overlay reference overlay
   * @param newOverlay new overlay
   */
  static ChangeCompatibility getCompatibility(const OverlayInputDefinition& overlay,
                                              const OverlayInputDefinition& newOverlay);

  std::map<readerid_t, ImageMapping*> imageMappings;

  const StereoRigDefinition* rigDef;
  std::shared_ptr<InputsMap> inputsMap;
  std::shared_ptr<InputsMapCubemap> inputsMapCubemap;
#ifndef VS_OPENCL
  std::unique_ptr<MaskInterpolation::InputMaskInterpolation> maskInterpolation;
#endif

  uint32_t alignSize;
  ImageMerger* merger;

  using PanoStitcherImplBase<Output>::getStreamForInput;
  using PanoStitcherImplBase<Output>::getInteractivePersp;
  using PanoStitcherImplBase<Output>::worstCompatibility;
  using PanoStitcherImplBase<Output>::getPano;
};
}  // namespace Core
}  // namespace VideoStitch
