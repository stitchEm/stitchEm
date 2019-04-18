// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "mutableprojectdefinition.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch-gui/utils/inputlensenum.hpp"
#include "libvideostitch-gui/utils/inputformat.hpp"
#include "libvideostitch-gui/widgets/crop/cropshapeeditor.hpp"
#include "libvideostitch-base/lockingproxy.hpp"

#include <QObject>
#include <QString>
#include <sstream>
#include <QMutex>

class ProjectDefinition;

typedef VideoStitch::Helper::LockingProxy<VideoStitch::Core::PanoDefinition, const ProjectDefinition>
    PanoDefinitionLocked;
typedef VideoStitch::Helper::LockingProxy<VideoStitch::Core::AudioPipeDefinition, const ProjectDefinition>
    AudioPipeDefinitionLocked;
typedef VideoStitch::Helper::LockingProxy<VideoStitch::Core::ImageMergerFactory, const ProjectDefinition>
    ImageMergerFactoryLocked;
typedef VideoStitch::Helper::LockingProxy<VideoStitch::Core::ImageFlowFactory, const ProjectDefinition>
    ImageFlowFactoryLocked;
typedef VideoStitch::Helper::LockingProxy<VideoStitch::Core::ImageWarperFactory, const ProjectDefinition>
    ImageWarperFactoryLocked;
typedef VideoStitch::Helper::LockingProxy<VideoStitch::Core::StereoRigDefinition, const ProjectDefinition>
    StereoRigDefinitionLocked;

namespace VideoStitch {
namespace Core {
class ImageMergerFactory;
class ImageFlowFactory;
class ImageWarperFactory;
}  // namespace Core
}  // namespace VideoStitch

class SignalCompressionCaps;

/**
 * A class describing the project. Thread-safe.
 * Also hold a few runtime-parameters for the project,
 * so not stateless, compared to the underlying MutableProjectDefinition.
 */
class VS_GUI_EXPORT ProjectDefinition : public QObject {
  Q_OBJECT

 public:
  ProjectDefinition();
  virtual ~ProjectDefinition();

  /**
   * @brief Change the state of the project to @modified and notify using the hasBeenModified(bool) signal.
   */
  void setModified(const bool modified = true);

  /**
   * @brief Validates the panorama of the project.
   * @param errsink Stream which contains the error.
   * @returns True = the panorama is valid / false the panorama is invalid.
   */
  bool validatePanorama(std::stringstream &errsink);

  /**
   * @brief Adds a project from a Ptv::Value. If there is an existing project, it will be deleted.
   * @param value Input value.
   * @returns Returns true on success. false on error, back in a clean state.
   */
  bool load(const VideoStitch::Ptv::Value &value);

  /**
   * @brief Close the current project.
   */
  void close();

  /**
   * @brief isInit indicates if a project is successfully loaded.
   * @returns True if is initialized.
   */
  bool isInit() const;

  /**
   * @brief Serialize a GUI project definition.
   * @returns The serialized PTV.
   */
  VideoStitch::Ptv::Value *serialize() const;

  /**
   * Returns the pano definition for the project.
   * @returns Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  PanoDefinitionLocked getPano() const;

  /**
   * @brief Returns a const pano definition pointer for the project.
   * @returns Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  const PanoDefinitionLocked getPanoConst() const;

  /**
   * @brief Returns a const audio pipe definition pointer for the project.
   * @returns Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  AudioPipeDefinitionLocked getAudioPipe() const;

  /**
   * @returns Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  const AudioPipeDefinitionLocked getAudioPipeConst() const;

  /**
   * @brief Returns the stereo rig definition for the project.
   * @returns Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  StereoRigDefinitionLocked getStereoRig() const;

  /**
   * @brief Returns a const stereo rig definition pointer for the project.
   * @returns Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  const StereoRigDefinitionLocked getStereoRigConst() const;

  /**
   * @brief Returns the merger factory for the project.
   * @returns Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  ImageMergerFactoryLocked getImageMergerFactory() const;

  /**
   * @brief Gets the image blender type.
   * @returns Blender type name, if any. Otherwise the default one.
   */
  QString getBlender() const;

  /**
   * @brief Gets the image flow type.
   * @returns Flow type name, if any. Otherwise the default one.
   */
  QString getFlow() const;

  /**
   * @brief Gets the image warper type.
   * @returns Warper type name, if any. Otherwise the default one.
   */
  QString getWarper() const;

  /**
   * @brief Gets the feather values
   * @returns Returns the feather values if any. Default value, otherwise.
   */
  int getFeather() const;

  /**
   * @brief Gets the sphere scale value
   * @returns Returns the sphere scale value if any. Default value, otherwise.
   */
  double getSphereScale() const;

  /**
   * Lower limit to sphereScale parameter
   */
  double computeMinimumSphereScale() const;

  /**
   * Returns the flow factory for the project.
   * @returns Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  ImageFlowFactoryLocked getImageFlowFactory() const;

  /**
   * Returns the flow factory for the project.
   * @returns Non-NULL when isInit() returns true, NULL when isInit() returns false.
   */
  ImageWarperFactoryLocked getImageWarperFactory() const;

  /**
   * @brief Get the number of inputs.
   * @returns The number of inputs.
   */
  readerid_t getNumInputs() const;

  /**
   * @brief Get the list of input names.
   * @note Audio-only input is NOT included.
   * @returns The list of input names.
   */
  QStringList getInputNames() const;

  /**
   * @brief Get input type for video (we ignore the audio only inputs)
   * @returns The video input type.
   */
  VideoStitch::InputFormat::InputFormatEnum getVideoInputType() const;

  /**
   * @brief Checks if the inputs provides audio only
   * @returns True if has audio input
   */
  bool hasAnInputWithAudioOnly() const;

  /**
   * @brief Checks if the projects contains audio
   * @returns True if there is one audio pipe configured
   */
  bool hasAudio() const;

  /**
   * @brief Checks that the inputs are images or procedurals only (and not videos)
   */
  bool hasImagesOrProceduralsOnly() const;

  /**
   * @brief Checks that the inputs are several videos (at least 2 videos, no images and no procedurals)
   */
  bool hasSeveralVideos() const;

  /**
   * @brief Checks that there is at least 2 visual inputs (videos, images or procedurals)
   */
  bool hasSeveralVisualInputs() const;

  /**
   * @brief Get the project lens type for the UI
   * @return The lens type (Rectilinear, FullFrameFisheye, CircularFisheye, Equirectangular)
   */
  InputLensClass::LensType getProjectLensType() const;

  /**
   * @brief Checks if at least one input has a valid crop area
   */
  bool hasCroppedArea() const;

  /**
   * @brief Get the current projection name.
   * @returns The projection name.
   */
  QString getProjection() const;

  /**
   * @brief Get the panorama (uncropped) dimensions.
   */
  void getImageSize(unsigned &width, unsigned &height) const;

  /**
   * @brief Get the panorama cropped dimensions.
   **/
  void getCroppedImageSize(unsigned &width, unsigned &height) const;

  /**
   * @brief Get the horizontal field of view in arc degrees.
   * @return The FOV value.
   */
  double getHFOV() const;

  /**
   * @brief The project is compatible with the lib version.
   * @return True is it's compatible.
   */
  bool hasFileFormatChanged() const;

  /**
   * @brief Updates the project according to the lib version.
   */
  void updateFileFormat();

  /**
   * @brief isDrawingInputNumbers
   * @return The input numbers are shown.
   */
  bool isDrawingInputNumbers() const;

  /**
   * @brief setAudioPipe writes the @audio pipe definition in the project. Thread-safe.
   */
  void setAudioPipe(VideoStitch::Core::AudioPipeDefinition *audioPipeDefinition);

  /**
   * @brief setPano writes the @pano definition in the project. Thread-safe.
   */
  void setPano(VideoStitch::Core::PanoDefinition *pano);

  /**
   * @brief setMergerFactory writes the @mergerFactory definition in the project. Thread-safe.
   */
  void setMergerFactory(VideoStitch::Core::ImageMergerFactory *mergerFactory);

  /**
   * @brief setFlowFactory writes the @flowFactory definition in the project. Thread-safe.
   */
  void setFlowFactory(VideoStitch::Core::ImageFlowFactory *flowFactory);

  /**
   * @brief setWarperFactory writes the @warperFactory definition in the project. Thread-safe.
   */
  void setWarperFactory(VideoStitch::Core::ImageWarperFactory *warperFactory);

  /**
   * @brief setStereoRigDefinition writes the @rigDefinition definition in the project. Thread-safe.
   */
  void setStereoRigDefinition(VideoStitch::Core::StereoRigDefinition *rigDefinition);

  /**
   * @brief Sets the panorama width and height. Updates project data if needed.
   * @param width New panorama width.
   * @param height New panorama height.
   */
  virtual void updateSize(int width, int height);

  /**
   * @brief Sets the Horizontal field of view.
   * @param fov New horizontal field of view.
   */
  void setHFov(double fov);

  /**
   * @brief Sets the sphere scale value.
   * @param sphereScale New sphere scale.
   */
  void setSphereScale(double sphereScale);

  /**
   * @brief Draws the input numbers on the input image
   * @param draw True
   */
  void setDrawInputNumbers(bool draw);

  /**
   * @brief Sets the nex projection of the panorama.
   * @param projection New projection.
   */
  void setProjection(QString projection);

  /**
   * @brief markAsSaved resets the isModified flag of the class to indicate there is no
   * discrepancy between the current project and what you have saved last.
   */
  void markAsSaved();

  /**
   * @brief hasLocalModifications indicates if the current project has modifications since the last
   * markAsSaved() call.
   */
  bool hasLocalModifications() const;

  /**
   * @brief The orientation grid is displayed.
   * @return True if the orientation grid must be displayed
   */
  bool getDisplayOrientationGrid() const;

  /**
   * @brief The project has a rig configuration (Stereo)
   * @return True if has a rig configuration
   */
  bool hasRigConfiguration() const;

  /**
   * @brief Gets the audio delay from the first active audio input
   * @return The delay value
   */
  int getAudioDelay() const;

  /**
   * @brief Sets the crop values for an input.
   * @param input The input number
   * @param crop Crop values
   * @param lensType Project lens type
   * @param applyToAll Apply the same crop to all inputs.
   */
  void setInputCrop(const unsigned int input, const Crop &crop, const InputLensClass::LensType lensType,
                    const bool applyToAll);

  /**
   * @brief Sets a rig configuration to the project.
   * @param orientation The orientation
   * @param geometry The rig geometry
   * @param diameter The rig diamater
   * @param ipd The rig IPD
   * @param leftInputs List of left inputs
   * @param rightInputs List of right inputs
   */
  void setRigConfiguration(const VideoStitch::Core::StereoRigDefinition::Orientation orientation,
                           const VideoStitch::Core::StereoRigDefinition::Geometry geometry, const double diameter,
                           const double ipd, const QVector<int> leftInputs, const QVector<int> rightInputs);

  /**
   * @brief Sets the image blender type.
   * @param merger Merger type
   * @param feather The feather value
   */
  void changeBlendingParameters(QString merger, int feather);

  /**
   * @brief Sets the advanced blender type.
   * @param flow Image flow type
   * @param warper Image warper type
   */
  void changeAdvancedBlendingParameters(const QString &flow, const QString &warper);

  /**
   * @brief Get a list of audio input names.
   * @return List of audio input names.
   */
  QStringList getAudioInputNames() const;

  /**
   * @brief Get a list of video input names.
   * @return List of video input names.
   */
  QStringList getVideoInputNames() const;

 public slots:
  void changeProjection(const QString &projName, const double HFOV);
  void updateMasks();
  void setDisplayOrientationGrid(bool display, bool reqRestitch = true);
  void toggleInputNumberDrawing();
  void setInterPupillaryDistance(double ipd);
  void setAudioDelay(int delay_ms);

 signals:
  void reqReset(SignalCompressionCaps *);
  void reqResetRig(SignalCompressionCaps *);
  void reqUpdateMask(int index, unsigned char *maskData, int width, int height);
  void reqUpdateMask(int index, QImage *mask);
  void maskUpdateDone();
  void reqDisplayWarning(QString);
  void hasBeenModified(bool modified);
  void imagesOrProceduralsOnlyHasChanged(bool hasImagesOrProceduralsOnly);
  void severalVideosOnlyHasChanged(bool hasSeveralVideos);
  void severalVisualInputsHasChanged(bool hasSeveralVisualInputs);
  void reqToggleInputNumbers(bool draw);
  void reqDisplayOrientationGrid(bool display);
  void reqToggleControlPoints(bool display);
  void reqSetAudioDelay(int value);

 protected:
  /**
   * @brief mutex A mutex to protect access to this object.
   * The mutex is recursive in case the user accesses different objects at the same time.
   */
  mutable QMutex mutex;
  SignalCompressionCaps *signalCompressor;

  friend class VideoStitch::Helper::LockingProxy<VideoStitch::Core::PanoDefinition,
                                                 const ProjectDefinition>;  // PanoDefinitionLocked;
  friend class VideoStitch::Helper::LockingProxy<VideoStitch::Core::AudioPipeDefinition,
                                                 const ProjectDefinition>;  // AudioDefinitionLocked;
  friend class VideoStitch::Helper::LockingProxy<VideoStitch::Core::StereoRigDefinition,
                                                 const ProjectDefinition>;  // StereoRigDefinitionLocked;
  friend class VideoStitch::Helper::LockingProxy<VideoStitch::Core::ImageMergerFactory,
                                                 const ProjectDefinition>;  // ImageMergerFactoryLocked;
  friend class VideoStitch::Helper::LockingProxy<VideoStitch::Core::ImageFlowFactory,
                                                 const ProjectDefinition>;  // ImageFlowFactoryLocked;
  friend class VideoStitch::Helper::LockingProxy<VideoStitch::Core::ImageWarperFactory,
                                                 const ProjectDefinition>;  // ImageWarperFactoryLocked;

  /**
   * @brief lock Grab the local instance mutex.
   */
  void lock() const;

  /**
   * @brief lock Release the local instance mutex.
   */
  void unlock() const;

  virtual MutableProjectDefinition *getDelegate() const = 0;

  virtual void createDelegate(const VideoStitch::Ptv::Value &value) = 0;

  virtual void destroyDelegate() = 0;

  void setDefaultPtvValues(const VideoStitch::Ptv::Value &value);

 private:
  QString getInputName(int index) const;
  void copyValuesIntoVector(std::vector<VideoStitch::Ptv::Value *> &vector, const QVector<int> inputList);

 private:
  /**
   * @brief Tells if the project definition has been modified since it was saved (VideoStitcher:::savePTV()).
   */
  bool isModified;
  bool drawInputNumbers;
  bool displayOrientationGrid;
};
