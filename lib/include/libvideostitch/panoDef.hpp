// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "inputDef.hpp"
#include "overlayInputDef.hpp"
#include "object.hpp"
#include "projections.hpp"
#include "controlPointListDef.hpp"

#include <string>

namespace VideoStitch {

class ThreadSafeOstream;

namespace Core {

class MergerMaskDefinition;
class RigDefinition;

/**
 * @brief The panorama definition
 */
class VS_EXPORT PanoDefinition : public Ptv::Object {
 public:
  /**
   * Output format (projection).
   */
  typedef PanoProjection Format;

  /**
   * Parses a PanoDefinition from a PTO file.
   * @param filename The pto filename. If "-", read from stdin.
   * @param sourcePano Calibration data will be added on top of this pano definition instead of creating one from
   * scratch. Optional.
   * @return The parsed PanoDefinition.
   */
  static Potential<PanoDefinition> parseFromPto(const std::string& filename, const PanoDefinition* sourcePano = NULL);

  /**
   * Build from a Ptv::Value.
   * @param value Input value.
   * @return The parsed PanoDefinition.
   */
  static PanoDefinition* create(const Ptv::Value& value);

  /**
   * Creates a copy of the PanoDefinition with changes from the stereo specification.
   * @param panoDiff Input value.
   */
  virtual Potential<PanoDefinition> createStereo(const Ptv::Value& panoDiff) const;

  /**
   * Clones a PanoDefinition java-style.
   * @return A similar PanoDefinition. Ownership is given to the caller.
   */
  virtual PanoDefinition* clone() const;

  Ptv::Value* serialize() const;

  /**
   * Comparison operator.
   */
  virtual bool operator==(const PanoDefinition& other) const;

  /**
   * Validates that the panorama makes sense.
   * @param os The sink for error messages.
   * @return false in case of failure.
   */
  virtual bool validate(std::ostream& os) const;

  /**
   * Returns true if all input masks are valid (see InputDefinition::validateMask()).
   */
  virtual bool validateInputMasks() const;

  virtual ~PanoDefinition();

  /**
   * Return the merger mask.
   */
  virtual const MergerMaskDefinition& getMergerMask() const;

  /**
   * Return the merger mask.
   */
  virtual MergerMaskDefinition& getMergerMask();

  /**
   * Return whether the blending mask is enabled
   */
  virtual bool getBlendingMaskEnabled() const;

  /**
   * Set the blending mask's "enabled"
   */
  virtual void setBlendingMaskEnabled(const bool enabled);

  /**
   * Return whether the blending mask interpolation is enabled
   */
  virtual bool getBlendingMaskInterpolationEnabled() const;

  /**
   * Set the blending mask interpolation
   */
  virtual void setBlendingMaskInterpolationEnabled(const bool enabled);

  /**
   * Return the blending mask's frame Id
   */
  virtual void removeBlendingMaskFrameIds(const std::vector<frameid_t>& frameIds);

  /**
   * Return the blending mask's width
   */
  virtual int64_t getBlendingMaskWidth() const;

  /**
   * Return the blending mask's height
   */
  virtual int64_t getBlendingMaskHeight() const;

  /**
   * Return the blending mask's frame Id
   */
  virtual std::vector<frameid_t> getBlendingMaskFrameIds() const;

  /**
   * Return the bounded frame's Id
   */
  virtual std::pair<frameid_t, frameid_t> getBlendingMaskBoundedFrameIds(const frameid_t frameId) const;

  /**
   * Return the blending order.
   */
  virtual std::vector<size_t> getMasksOrder() const;

  /**
   * Return the blending mask scale factor (in the input space).
   */
  virtual int getBlendingMaskInputScaleFactor() const;

  /**
   * Return inputs map
   */
  virtual std::vector<std::pair<frameid_t, std::map<videoreaderid_t, std::string>>> getInputIndexPixelDataIfValid(
      const frameid_t frameId) const;

  /**
   * Helper function to convert an input index into a video input index
   * @param i Input id.
   */
  virtual videoreaderid_t convertInputIndexToVideoInputIndex(readerid_t i) const;

  /**
   * Helper function to convert an input index into an audio input index
   * @param i Input id.
   */
  virtual audioreaderid_t convertInputIndexToAudioInputIndex(readerid_t i) const;

  /**
   * Helper function to convert a video input index into an input index
   * @param i Video input id.
   */
  virtual readerid_t convertVideoInputIndexToInputIndex(videoreaderid_t i) const;

  /**
   * Helper function to convert an audio input index into an input index
   * @param i Audio input id.
   */
  virtual readerid_t convertAudioInputIndexToInputIndex(audioreaderid_t i) const;

  /**
   * Returns the i-th input.
   * @param i Input id.
   */
  virtual const InputDefinition& getInput(readerid_t i) const;

  /**
   * Returns the i-th input.
   * @param i Input id.
   */
  virtual InputDefinition& getInput(readerid_t i);

  /**
   * Returns the i-th video input (numbering is different from getInput()).
   * @param i Video input id.
   */
  virtual const InputDefinition& getVideoInput(videoreaderid_t i) const;

  /**
   * Returns the i-th video input (numbering is different from getInput()).
   * @param i Video input id.
   */
  virtual InputDefinition& getVideoInput(videoreaderid_t i);

  /**
   * Returns the i-th audio input (numbering is different from getInput()).
   * @param i Audio input id.
   */
  virtual const InputDefinition& getAudioInput(audioreaderid_t i) const;

  /**
   * Returns the i-th audio input (numbering is different from getInput()).
   * @param i Audio input id.
   */
  virtual InputDefinition& getAudioInput(audioreaderid_t i);

  /**
   * Returns the inputs.
   */
  virtual std::vector<std::reference_wrapper<const InputDefinition>> getInputs() const;

  /**
   * Returns the inputs.
   */
  virtual std::vector<std::reference_wrapper<InputDefinition>> getInputs();

  /**
   * Returns the video-enabled inputs.
   */
  virtual std::vector<std::reference_wrapper<const InputDefinition>> getVideoInputs() const;

  /**
   * Returns only the video-enabled inputs.
   */
  virtual std::vector<std::reference_wrapper<InputDefinition>> getVideoInputs();

  /**
   * Returns only the audio-enabled inputs.
   */
  virtual std::vector<std::reference_wrapper<const InputDefinition>> getAudioInputs() const;

  /**
   * Returns the audio-enabled inputs.
   */
  virtual std::vector<std::reference_wrapper<InputDefinition>> getAudioInputs();

  /**
   * Inserts an input at a given position.
   * @param inputDef Input definition to insert. We take ownership.
   * @param i Input id of the input. If < 0, inserts at the last position.
   */
  virtual void insertInput(InputDefinition* inputDef, readerid_t i);

  /**
   * Deletes the input at a given position, and returns it. The caller is responsible for freeing it.
   * @param i Input id of the input. Between 0 and (numInputs() - 1).
   */
  virtual InputDefinition* popInput(readerid_t i);

  /**
   * Deletes the input at a given position. Returns true if the input is successfully deleted.
   * @param i Input id of the input. Between 0 and (numInputs() - 1).
   */
  virtual bool removeInput(readerid_t i);

  /**
   * Returns the number of inputs.
   * @note it considers all types of inputs - if you only need video-enabled or audio-enabled ones, use numVideoInputs()
   * or numAudioInputs()
   */
  virtual readerid_t numInputs() const;

  /**
   * Returns the number of video-enabled inputs.
   * @note use getVideoInputs() to get a vector of video-enabled inputs, filtering out all other ones
   */
  virtual videoreaderid_t numVideoInputs() const;

  /**
   * Returns the number of audio-enabled inputs.
   * @note use getAudioInputs() to get a vector of audio-enabled inputs, filtering out all other ones
   */
  virtual audioreaderid_t numAudioInputs() const;

  /**
   * @brief Get the horizontal fov from input sources
   * @note  The hfov must be the same for all the input sources. Otherwise, a warning is added in the logger and the
   * median is returned.
   * @return The FOV from inputs
   */
  virtual double getHFovFromInputSources() const;

  /**
   * @brief Get the lens format from input sources
   * @note The lens formats must be the same for all the input sources. Otherwise, a warning is added in the logger and
   * the first is returned.
   * @return The lens format from inputs
   */
  InputDefinition::Format getLensFormatFromInputSources() const;

  /**
   * @brief Get the lens category from input sources
   * @note The lens categories must be the same for all the input sources. Otherwise, a warning is added in the logger
   * and the first is returned.
   * @return The lens category from inputs
   */
  InputDefinition::LensModelCategory getLensModelCategoryFromInputSources() const;

  /**
   * Returns calibration control points
   */
  virtual const ControlPointListDefinition& getControlPointListDef() const;

  /**
   * Returns calibration control points
   */
  virtual ControlPointListDefinition& getControlPointListDef();

  /**
   * Returns whether to use the precomputed coordinate buffer for ImageMapping
   */
  virtual bool getPrecomputedCoordinateBuffer() const;

  /**
   * Set whether to use the precomputed coordinate buffer for ImageMapping
   */
  virtual void setPrecomputedCoordinateBuffer(const bool b);

  /**
   * Returns the precomputed coordinate shrink factor for ImageMapping
   */
  virtual double getPrecomputedCoordinateShrinkFactor() const;

  /**
   * Set the precomputed coordinate shrink factor for ImageMapping
   */
  virtual void setPrecomputedCoordinateShrinkFactor(const double b);

  /**
   * Returns the panorama width.
   */
  virtual int64_t getWidth() const;

  /**
   * Returns the panorama height.
   */
  virtual int64_t getHeight() const;

  /**
   * Returns the cubemap edge length.
   */
  virtual int64_t getLength() const;

  /**
   * Returns the list of output postprocessors.
   */
  virtual const Ptv::Value* getPostprocessors() const;

  /**
   * The output projection.
   */
  virtual PanoProjection getProjection() const;

  /**
   * Returns the horizontal field of view.
   */
  virtual double getHFOV() const;

  /**
   * Returns the vertical field of view. The field of view refers to the cropped input ('C' in pts, not 'S')
   */
  virtual double getVFOV() const;

  /**
   * Return the output sphere scale
   * @return the sphere scale
   */
  virtual double getSphereScale() const;

  /**
   * Set the output sphere scale. Natural lower limit can be queried through getMinimumRigSphereRadius().
   * @param scale the new sphere scale
   */
  virtual void setSphereScale(double scale);

  /**
   * Set the calibration cost
   * @param cost final cost returned by the calibration algorithm
   */
  virtual void setCalibrationCost(double cost);

  /**
   * Return the calibration cost
   */
  virtual double getCalibrationCost() const;

  /**
   * @brief setCalibrationInitialHFOV
   * @param hfov used to initialize the calibration algorithm
   */
  virtual void setCalibrationInitialHFOV(double hfov);

  /**
   * Return the hfov value used to initialize the calibration algorithm
   */
  virtual double getCalibrationInitialHFOV() const;

  /**
   * Set the calibration control point list
   * @param list control point list
   */
  virtual void setCalibrationControlPointList(const ControlPointList& list);

  /**
   * Return the calibration control point list
   */
  virtual const ControlPointList& getCalibrationControlPointList() const;

  /**
   * Set the calibration rig presets
   * @param rigDef the rig definition presets. We take ownership.
   */
  virtual void setCalibrationRigPresets(RigDefinition* rigDef);

  /**
   * Get the calibration rig presets
   * @return the calibration rig presets
   */
  virtual const RigDefinition& getCalibrationRigPresets() const;

  /**
   * Get calibration rig presets name
   */
  virtual std::string getCalibrationRigPresetsName() const;

  /**
   * Calibration rig presets are optional, check if PanoDefinition has one
   */
  virtual bool hasCalibrationRigPresets() const;

  /**
   * @brief Checks if a rig information preset is compatible with the current panorama.
   * @param rigValue Rig PTV value
   * @return True if it's compatible
   */
  bool isRigPresetCompatible(const VideoStitch::Ptv::Value* rigValue) const;

  /**
   * Set the "has been deshuffled" flag
   * @param deshuffled boolean indicating whether the PanoDefinition has been deshuffled by the calibration algorithm
   */
  virtual void setHasBeenCalibrationDeshuffled(const bool deshuffled);

  /**
   * Check if the pano has beed deshuffled by the calibration algorithm
   */
  virtual bool hasBeenCalibrationDeshufled() const;

  /**
   * Check if the pano is already synchronized
   */
  bool hasBeenSynchronized() const;

  /**
   * Check if the pano is already calibrated with control points on it
   */
  bool hasCalibrationControlPoints() const;

  /**
   * Checks if the inputs gemoetries where computed
   */
  bool hasBeenCalibrated() const;

  /**
   * Check if the pano photometry is already calibrated
   */
  bool photometryHasBeenCalibrated() const;

  /**
   * Check if the pano input geometries have translations
   */
  bool hasTranslations() const;

  /**
   * Minimum distance of any inputs principal point to the world origin.
   * Natural lower limit to the sphere scale parameter.
   */
  double computeMinimumRigSphereRadius() const;

  /**
   * Returns the PTV name of an output projection.
   */
  static const char* getFormatName(const PanoProjection& fmt);

  /**
   * Convert a panotools int to an PanoDefinition::Format.
   * @param ptFmt a PanoTools identifier representing the format. Can be either an integer (PanoTools) or a string
   * (PTGui)
   * @param fmt VideoStitch output format.
   * @return false of the output format is not supported.
   */
  static bool fromPTFormat(const char* ptFmt, PanoProjection* fmt);
  /**
   * Convert a PTGui format to an PanoDefinition::Format.
   * @param ptFmt a PTGui string representing the format (e.g. 'frectilinear').
   * @param fmt VideoStitch output format.
   * @return false if the input format is not supported.
   */
  static bool fromPTSFormat(const std::string& ptFmt, PanoProjection* fmt);

  /**
   * Sets the width of the pano.
   */
  virtual void setWidth(uint64_t w);

  /**
   * Sets the height of the pano.
   */
  virtual void setHeight(uint64_t h);

  /**
   * Sets the length of an edge of the cubemap.
   */
  virtual void setLength(uint64_t);

  /**
   * green/red/blue compensation factor. (Linear scale, 1.0 means no correction)
   * For hugin, this is always 1. For PTGui, this is computed from red and blue so that the sum of the 3 components is 0
   * in log scale. We convert it on import.
   */
  DECLARE_CURVE(RedCB, double)
  DECLARE_CURVE(GreenCB, double)
  DECLARE_CURVE(BlueCB, double)

  /**
   * Exposure value, log scale.
   */
  DECLARE_CURVE(ExposureValue, double)

  /**
   * Sets the output projection to a constant format, erasing any interpolated formats.
   */
  virtual void setProjection(PanoProjection format);

  /**
   * Sets the horizontal field of view.
   */
  virtual void setHFOV(double hFov);

  /**
   * Sets the horizontal field of view corresponding to the given vertical field of view.
   */
  virtual void setVFOV(double vFov);

  /**
   * Returns the overlays.
   */
  virtual std::vector<std::reference_wrapper<const OverlayInputDefinition>> getOverlays() const;

  /**
   * Returns the overlays.
   */
  virtual std::vector<std::reference_wrapper<OverlayInputDefinition>> getOverlays();

  /**
   * Returns the i-th overlay.
   * @param i Overlay id.
   */
  virtual const OverlayInputDefinition& getOverlay(overlayreaderid_t i) const;

  /**
   * Returns the i-th overlay.
   * @param i Overlay id.
   */
  virtual OverlayInputDefinition& getOverlay(overlayreaderid_t i);

  /**
   * Inserts an overlay at a given position.
   * @param overlayDef overlay definition to insert. We take ownership.
   * @param i overlay id of the overlay. If < 0, inserts at the last position.
   */
  virtual void insertOverlay(OverlayInputDefinition* overlayDef, overlayreaderid_t i);

  /**
   * Deletes the overlay at a given position, and returns it. The caller is responsible for freeing it.
   * @param i overlay id of the overlay. Between 0 and (numOverlays() - 1).
   */
  virtual OverlayInputDefinition* popOverlay(overlayreaderid_t i);

  /**
   * Deletes the overlay at a given position. Returns true if the overlay is successfully deleted.
   * @param i overlay id of the overlay. Between 0 and (numOverlays() - 1).
   */
  virtual bool removeOverlay(overlayreaderid_t i);

  /**
   * Returns the number of Overlays.
   * @note it considers all types of Overlays - if you only need video-enabled or audio-enabled ones, use
   * numVideoOverlays() or numAudioOverlays()
   */
  virtual overlayreaderid_t numOverlays() const;

  /**
   * Global orientation curve.
   */
  DECLARE_CURVE(GlobalOrientation, Quaternion<double>)
  /**
   * Global stabilization curve.
   */
  DECLARE_CURVE(Stabilization, Quaternion<double>)
  DECLARE_CURVE(StabilizationYaw, double)
  DECLARE_CURVE(StabilizationPitch, double)
  DECLARE_CURVE(StabilizationRoll, double)

  /**
   * Compute the optimal resolution for the panorama given the resolution of the inputs and the setup.
   * The idea is to minimize the distortions at the center of the panorama.
   * @returns Sets width and height references to the optimal size.
   */
  virtual void computeOptimalPanoSize(unsigned& width, unsigned& height) const;

  /**
   * @brief resetExposure Resets exposure curve values for all inputs.
   */
  virtual void resetExposure();

  /**
   * @brief Removes control points, removes rig definition and sets inputs FOV to default value.
   */
  void resetCalibration();

  /** Create a PanoProjection from its name
   * @param fmt A string with the name of the PanoProjection, e.g. 'equirectangular'.
   */
  static PanoProjection getFormatFromName(const std::string& fmt);

 protected:
  // Todo: consider extracting an interface to implement decorator
  PanoDefinition();
  PanoDefinition(PanoDefinition&& rhs);

  template <typename T>
  using inserted_iterator = typename std::vector<T>::iterator;

  template <typename T>
  static inserted_iterator<T> safeInsert(std::vector<T>& container, const T& value, readerid_t index) {
    if (index < 0 || index > (readerid_t)container.size()) {
      index = (readerid_t)container.size();
    }
    return container.insert(container.begin() + index, value);
  }

  template <typename T>
  static T safeRemove(std::vector<T>& container, readerid_t index) {
    T result = nullptr;
    if (0 <= index && index < (readerid_t)container.size()) {
      result = container[index];
      container.erase(container.begin() + index);
    }
    return result;
  }

 private:
  Status readParams(char* line);

  /**
   * Disabled, use clone()
   */
  PanoDefinition(const PanoDefinition&) = delete;

  /**
   * Disabled, use clone()
   */
  PanoDefinition& operator=(const PanoDefinition&) = delete;

  Status parseOrientationCurves(const Ptv::Value& value);
  Status parseExposureCurves(const Ptv::Value& value);

 private:
  class Pimpl;
  Pimpl* pimpl;
  friend class MergerPair;
};

}  // namespace Core
}  // namespace VideoStitch
