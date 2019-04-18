// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "inputController.hpp"

#include "config.hpp"
#include "allocator.hpp"
#include "input.hpp"
#include "status.hpp"
#include "audio.hpp"
#include "audioPipeDef.hpp"
#include "stitchOutput.hpp"
#include "gpu_device.hpp"
#include "quaternion.hpp"
#include "orah/imuStabilization.hpp"

#include <vector>
#include <functional>

namespace VideoStitch {

namespace Input {
class ReaderFactory;
}

namespace Core {
class InputDefinition;
class ImageMergerFactory;
class ImageWarperFactory;
class ImageFlowFactory;
class PanoDefinition;
class PostProcessor;
class PreProcessor;
class StereoRigDefinition;
template <class Writer>
class StitcherOutput;
typedef class StitcherOutput<Output::VideoWriter> StitchOutput;
typedef class StitcherOutput<Output::StereoWriter> StereoOutput;
template <class StitcherOutput>
class Stitcher;
typedef class Stitcher<StitchOutput> PanoStitcher;
typedef class Stitcher<StereoOutput> StereoStitcher;

template <typename Stitcher>
struct DeviceDef;

/**
 * Controller deleter (deleting has to be done through Controller::deleteController).
 */
template <typename Controller>
class VS_EXPORT ControllerDeleter {
 public:
  /**
   * Deletes the controller.
   * @param controller Controller to delete.
   */
  void operator()(Controller* controller) const { deleteController(controller); }
};

/**
 * @brief A controller class that is used to orchestrate stitching.
 *
 * A controller creates and owns stitchers that can be run independently on different GPUs,
 * while reading in a thread-safe way from the same inputs.
 * Stitchers created from a controller are called the 'children' of this controller.
 */
template <class Out, class DeviceDef>
class VS_EXPORT StitcherController : public virtual InputController {
 public:
  /**
   */
  typedef Out Output;                  ///< Output
  typedef DeviceDef DeviceDefinition;  ///< DeviceDefinition

  typedef Potential<StitcherController, ControllerDeleter<StitcherController>>
      PotentialController;                    ///< PotentialController
  typedef Potential<Output> PotentialOutput;  ///< PotentialOutput

  /**
   * Creates a stitcher for the controller's panorama on the default device.
   * The stitcher must be deleted using deleteStitcher().
   */
  virtual Status createStitcher() = 0;

  /**
   * Deletes a stitcher created using createStitcher() on the controller, for the default device.
   */
  virtual void deleteStitcher() = 0;

  /**
   * Install a single audio callback without removing the ones already installed.
   * As usual, ownership is transferred to the Controller, forget about this pointer
   * afterward.
   * @return false if the identifier is not unique and the callback couldn't be installed.
   */
  virtual bool addAudioOutput(std::shared_ptr<VideoStitch::Output::AudioWriter>) = 0;

  /**
   * Remove an audio callback by its identifier.
   * @return false if this callback is not installed yet.
   */
  virtual bool removeAudioOutput(const std::string&) = 0;

  /**
   * Creates an Output that writes synchronously to a writer.
   * This means that calls to extract() using this Output will block until
   * the underlying writer is done writing the frame.
   * Takes ownership of the writer.
   * @param source The index of the source input.
   * @param writer The writer to output frames to.
   */
  virtual Potential<ExtractOutput> createBlockingExtractOutput(
      int source, std::shared_ptr<SourceSurface> surf, std::shared_ptr<SourceRenderer> renderer,
      std::shared_ptr<VideoStitch::Output::VideoWriter> writer) = 0;

  /**
   * Creates an Output that writes asynchronously to a writer.
   * Takes ownership of the writer.
   * @param source The index of the source input.
   * @param writer The writer to output frames to.
   */
  virtual Potential<ExtractOutput> createAsyncExtractOutput(
      int source, const std::vector<std::shared_ptr<SourceSurface>>& surf, std::shared_ptr<SourceRenderer> renderer,
      std::shared_ptr<VideoStitch::Output::VideoWriter> writer) const = 0;

  /**
   * A dummy output.
   */
  virtual Potential<Output> createBlockingStitchOutput(std::shared_ptr<PanoSurface> surf) {
    std::vector<std::shared_ptr<typename Output::Writer>> writers;
    std::vector<std::shared_ptr<PanoRenderer>> renderers;
    return createBlockingStitchOutput(surf, renderers, writers);
  }

  /**
   * Creates an Output that writes synchronously to a writer.
   * This means that calls to stitch() using this Output will block until
   * the underlying writer is done writing the frame.
   * Takes ownership of the writer.
   * @param writer The writer to output frames to.
   * @note This Output is NOT thread safe. It is meant to be used by only one stitcher.
   */
  virtual Potential<Output> createBlockingStitchOutput(std::shared_ptr<PanoSurface> surf,
                                                       std::shared_ptr<typename Output::Writer> writer) {
    std::vector<std::shared_ptr<typename Output::Writer>> writers(1, writer);
    std::vector<std::shared_ptr<PanoRenderer>> renderers;
    return createBlockingStitchOutput(surf, renderers, writers);
  }

  /**
   * Same as above, but adds a Renderer in the loop instead of a host callback.
   * @param renderer The renderer which will get the uploaded texture.
   */
  virtual Potential<Output> createBlockingStitchOutput(std::shared_ptr<PanoSurface> surf,
                                                       std::shared_ptr<PanoRenderer> renderer) {
    std::vector<std::shared_ptr<typename Output::Writer>> writers;
    std::vector<std::shared_ptr<PanoRenderer>> renderers(1, renderer);
    return createBlockingStitchOutput(surf, renderers, writers);
  }

  /**
   * Same as above, but tees the output to several writers.
   * @param writers The writers to output frames to. We take ownership of the writers.
   */
  virtual Potential<Output> createBlockingStitchOutput(
      std::shared_ptr<PanoSurface> surf, const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
      const std::vector<std::shared_ptr<typename Output::Writer>>& writers) = 0;

  /**
   * Creates an Output that writes asynchronously to a writer in a separate thread.
   * This means that calls to stitch() using this Output will block only while the frame is copied,
   * or when the frame buffer is full. This allows hiding some of the writer processing time by interleaving
   * stitching (GPU + I/O) and writing (CPU + I/O).
   * Takes ownership of the writer.
   *
   * @param writer The writer to output frames to.
   * @note This Output is thread-safe. Note, however, that to avoid starvation, frames that arrive with an advance of
   * more than numBufferedFrames will block the stitch() call.
   */
  virtual Potential<Output> createAsyncStitchOutput(const std::vector<std::shared_ptr<PanoSurface>>& surf,
                                                    std::shared_ptr<typename Output::Writer> writer) {
    std::vector<std::shared_ptr<typename Output::Writer>> writers(1, writer);
    std::vector<std::shared_ptr<PanoRenderer>> renderers;
    return createAsyncStitchOutput(surf, renderers, writers);
  }

  /**
   * Same as above, but adds a Renderer in the loop instead of a host callback.
   * @param renderer The renderer which will get the uploaded texture.
   */
  virtual Potential<Output> createAsyncStitchOutput(const std::vector<std::shared_ptr<PanoSurface>>& surf,
                                                    std::shared_ptr<PanoRenderer> renderer) {
    std::vector<std::shared_ptr<typename Output::Writer>> writers;
    std::vector<std::shared_ptr<PanoRenderer>> renderers(1, renderer);
    return createAsyncStitchOutput(surf, renderers, writers);
  }

  /**
   * Same as above, but tees the output to several writers.
   * @param writers The writers to output frames to. We take ownership of the writers.
   */
  virtual Potential<Output> createAsyncStitchOutput(
      const std::vector<std::shared_ptr<PanoSurface>>& surf,
      const std::vector<std::shared_ptr<PanoRenderer>>& renderers,
      const std::vector<std::shared_ptr<typename Output::Writer>>& writers) const = 0;

  /**
   * Stitches a full panorama image.
   * @param output Where to write the panorama.
   * @param readFrame If false, the stitcher will not read the next frame but will restitch the last frame.
   * @return True on success.
   */
  virtual ControllerStatus stitch(Output* output, bool readFrame = true) = 0;

  /**
   * Returns the merger factory.
   */
  virtual const ImageMergerFactory& getMergerFactory() const = 0;

  /**
   * Returns the warper factory.
   */
  virtual const ImageWarperFactory& getWarperFactory() const = 0;

  /**
   * Returns the flow factory.
   */
  virtual const ImageFlowFactory& getFlowFactory() const = 0;

  /**
   * @param output Which frame to write and where to write it.
   * @param readFrame If false, the stitcher will not read the next frame but will export the current input frames.
   */
  virtual ControllerStatus extract(ExtractOutput* output, bool readFrame) = 0;

  /**
   * Extracts the current input frame for several inputs.
   * @param extracts outputs
   * @param algo Reserved, do not use.
   * @param readFrame If false, the stitcher will not read the next frame but will export the current input frames.
   */
  virtual ControllerStatus extract(std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo, bool readFrame) = 0;

  /**
   * Stitches a full panorama image and extracts the input frames.
   * @param output Where to write the panorama.
   * @param extracts Which frames to write and where to write them.
   * @param algo Reserved, do not use.
   * @param readFrame If false, the stitcher will not read the next frame but will export the current input frames.
   * @return false on failure.
   */
  virtual ControllerStatus stitchAndExtract(Output* output, std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo,
                                            bool readFrame) = 0;

  /**
   * Returns the panorama definition.
   */
  virtual const PanoDefinition& getPano() const = 0;

  /**
   * Returns the stereo rig definition.
   */
  virtual const StereoRigDefinition* getRig() const = 0;

  /**
   * Checks whether a pano definition change would be compatible
   * @return returns true if Ok, false if the changes are incompatible
   */
  virtual bool isPanoChangeCompatible(const PanoDefinition& newPano) const = 0;

  /**
   * Applies set of changes represented by panoramaUpdater to current panorama and resets the panorama.
   * @param panoramaUpdater Function that applies some changes on top of panorama.
   * @return returns true if Ok, false if changes application or reset failed
   */
  virtual Status updatePanorama(
      const std::function<Potential<PanoDefinition>(const PanoDefinition&)>& panoramaUpdater) = 0;

  /**
   * Basically - thread safe resetPanorama, that operates through creating updater function and calling overload above.
   * @param panorama New panorama.
   * @return returns true if Ok, false if changes application or reset failed
   */
  virtual Status updatePanorama(const PanoDefinition& panorama) = 0;

  /**
   * Modifies the rig definition. Note that this trigger a setup.
   * This is currently not thread-safe with respect to stitching, please make sure that you don't call that while
   * stitching.
   * @param newRig New rig.
   */
  virtual Status resetRig(const StereoRigDefinition& newRig) = 0;

  /**
   * Modifies the merger factory.
   * @param newMergerFactory New merger factory. If not compatible, an error is returned, and the controller is not
   * modified.
   * @param redoSetupNow Whether to redo the setup for existing stitchers. If this is false, the setup will happen the
   * next time resetPano() is called.
   */
  virtual Status resetMergerFactory(const ImageMergerFactory& newMergerFactory, bool redoSetupNow) = 0;

  virtual Status resetWarperFactory(const ImageWarperFactory& newWarperFactory, bool redoSetupNow) = 0;

  virtual Status resetFlowFactory(const ImageFlowFactory& newFlowFactory, bool redoSetupNow) = 0;

  /**
   * Seeks the given frame. Will block if frames are being stitched.
   * Further stitching will resume from this frame for children stitchers.
   * @param frame The frame to seek to.
   * @return False if at least one reader failed to seek. In that case, clients should seek again to make sure that the
   * readers are at a consistent frame.
   */
  virtual Status seekFrame(frameid_t frame) = 0;

  /**
   * Returns the current controller's frame number.
   */
  virtual frameid_t getCurrentFrame() const = 0;

  /**
   * Rotate the panorama using Euler angles (@a yaw, @a pitch @a roll).
   * VideoStitch uses the Body 2-1-3 Euler angle parameterization.
   * Angles are in degrees.
   * Successive rotations do not supersede one another ; they are cumulatives.
   */
  virtual void applyRotation(double yaw, double pitch, double roll) = 0;

  /**
   * Reset the interactive orientation of the panorama.
   */
  virtual void resetRotation() = 0;

  /**
   * Get the current interactive orientation of the panorama.
   */
  virtual Quaternion<double> getRotation() const = 0;

  /**
   * Set the current interactive sphere scale of the panorama.
   */
  virtual void setSphereScale(const double sphereScale) = 0;

  /**
   * Preprocessor setter.
   * @note The current i-th preprocessor will be deleted and replaced by @a p.
   * @note ownership of @a p is transferred to the controller.
   */
  virtual void setPreProcessor(int i, PreProcessor* p) = 0;

  /**
   * Postprocessor setter.
   * @note The current postprocessor will be deleted and replaced by @a p.
   * @note ownership of @a p is transferred to the controller.
   */
  virtual void setPostProcessor(PostProcessor* p) = 0;

  /**
   * AudioPreprocessor setter.
   * @param name Name of the audio preprocessor to activate
   * @param gr Group of the audio preprocessor to be applied
   */
  virtual void setAudioPreProcessor(const std::string& name, groupid_t gr) = 0;

  /**
   * Enables/disables preprocessing.
   */
  virtual void enablePreProcessing(bool value) = 0;

  /**
   * Enables/disables metadata processing.
   */
  virtual void enableMetadataProcessing(bool value) = 0;

  /**
   * @return Is an audio signal available.
   */
  virtual bool hasAudio() const = 0;

  /**
   * Audio input setter.
   * @note The current audio input will be replaced by the one selected.
   * @param name of the audio input selected
   */
  virtual void setAudioInput(const std::string& name) = 0;

  /**
   * Audio delay setter.
   * @note It will set a delay for the current audio input.
   * @param delay in ms
   */
  virtual Status setAudioDelay(double delay_ms) = 0;

  /**
   * Apply a new audio pipe definition for processors
   * @param the new audio pipe definition to be applied
   */
  virtual Status applyAudioProcessorParam(const AudioPipeDefinition& newAudioPipe) = 0;

  /**
   * Checks whether a VuMeter processor can be found for this input
   */
  virtual bool hasVuMeter(const std::string& inputName) const = 0;

  virtual std::vector<double> getPeakValues(const std::string& inputName) const = 0;

  virtual std::vector<double> getRMSValues(const std::string& inputName) const = 0;

  /**
   * IMU Stabilization accessors
   */
  virtual const Quaternion<double> getUserOrientation() = 0;
  virtual void setUserOrientation(const Quaternion<double>& q) = 0;
  virtual void updateUserOrientation(const Quaternion<double>& q) = 0;
  virtual void resetUserOrientation() = 0;

  /**
   * @brief enables (true) or disables (false) the IMU stabilization
   *  It always resets previous user defined orientation
   */
  virtual void enableStabilization(bool value) = 0;
  virtual bool isStabilizationEnabled() = 0;

  virtual Stab::IMUStabilization& getStabilizationIMU() = 0;

  /**
   * Returns the maximum reader latency in ms.
   */
  virtual mtime_t getLatency() const = 0;

  virtual Status addSink(const Ptv::Value* config) = 0;
  virtual void removeSink() = 0;

 protected:
  ~StitcherController() {}

  /**
   * Modifies the pano definition. Note that this may trigger a setup for some changes.
   * This is currently not thread-safe with respect to stitching, please make sure that you don't call that while
   * stitching. For performance reasons, this does not change the current state of the readers. Therefore, if you change
   * frame offsets, you'll have to seekFrame() to the current frame to resynchronize the readers.
   * @param newPano New pano. If not compatible, an error is returned, and the controller is not modified.
   * @return returns true if Ok, false if the changes are incompatible
   */
  virtual Status resetPano(const PanoDefinition& newPano) = 0;
};

typedef StitcherController<StitchOutput, PanoDeviceDefinition> Controller;
typedef StitcherController<StereoOutput, StereoDeviceDefinition> StereoController;

typedef Potential<Controller, ControllerDeleter<Controller>> PotentialController;
typedef Potential<StitchOutput> PotentialStitchOutput;

typedef Potential<StereoController, ControllerDeleter<StereoController>> PotentialStereoController;
typedef Potential<StereoOutput> PotentialStereoOutput;

/**
 * Creates a monoscopic panorama controller, which must be deleted using deleteController().
 * @param pano The panorama definition. Will be copied.
 * @param mergerFactory Factory to create mergers.
 * @param readerFactory Used to create readers. Ownership is transferred to the controller.
 * @param audioPipe Audio pipeline definition.
 * @return A new controller.
 */
Potential<Controller, ControllerDeleter<Controller>> VS_EXPORT createController(
    const PanoDefinition& pano, const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
    const ImageFlowFactory& flowFactory, Input::ReaderFactory* readerFactory, const AudioPipeDefinition& audioPipe);

/**
 * Deletes a controller.
 * @param controller The controller to delete. Invalid after the call.
 */
void VS_EXPORT deleteController(Controller* controller);

/**
 * Creates a stereoscopic controller, which must be deleted using deleteController().
 * @param pano The panorama definition. Will be copied.
 * @param rig The rig definition. Will be copied.
 * @param mergerFactory Factory to create mergers.
 * @param readerFactory Used to create readers. Ownership is transferred to the controller.
 * @return A new controller.
 */
Potential<StereoController, ControllerDeleter<StereoController>> VS_EXPORT createController(
    const PanoDefinition& pano, const StereoRigDefinition& rig, const ImageMergerFactory& mergerFactory,
    const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory, Input::ReaderFactory* readerFactory);

/**
 * Deletes a stereoscopic controller.
 * @param controller The controller to delete. Invalid after the call.
 */
void VS_EXPORT deleteController(StereoController* controller);
}  // namespace Core
}  // namespace VideoStitch
