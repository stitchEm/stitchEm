// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "inputControllerImpl.hpp"

#include "audio/audioPipeline.hpp"
#include "exposure/metadataProcessor.hpp"

#include "libvideostitch/orah/imuStabilization.hpp"
#include "libvideostitch/controller.hpp"
#include "libvideostitch/preprocessor.hpp"
#include "libvideostitch/postprocessor.hpp"

#include <list>
#include <mutex>

#ifdef ANDROID__GNUSTL
#define timed_mutex mutex
#define try_lock_for(a) try_lock()
#endif

#ifdef _MSC_VER
#pragma warning(push)
// using virtual inheritance of InputController on purpose
#pragma warning(disable : 4250)
#endif

namespace VideoStitch {

namespace Core {

class ImageMergerFactory;
class Mutex;
class PreProcessor;

/**
 * @brief Controller implementation.
 */
template <typename VideoPipeline>
class ControllerImpl
    : public InputControllerImpl,
      public StitcherController<typename VideoPipeline::Output, typename VideoPipeline::DeviceDefinition> {
  using typename StitcherController<typename VideoPipeline::Output, typename VideoPipeline::DeviceDefinition>::Output;
  using typename StitcherController<typename VideoPipeline::Output,
                                    typename VideoPipeline::DeviceDefinition>::DeviceDefinition;
  typedef typename Output::Writer Writer;

  using typename StitcherController<Output, DeviceDefinition>::PotentialOutput;
  using typename StitcherController<Output, DeviceDefinition>::PotentialController;

 public:
  /**
   * Factory function to create a Controller with the @pano PanoDefinition and the @readerFactory input ReaderFactory.
   * @param pano The panorama to stitch.
   * @param mergerFactory Factory to create mergers.
   * @param readerFactory The factory for reader. We take ownership.
   * @param preprocessors Vector of the preprocessors.
   * @param rig Stereo rig definition.
   */
  static PotentialController create(const PanoDefinition& pano, const AudioPipeDefinition& audioPipe,
                                    const ImageMergerFactory& mergerFactory, const ImageWarperFactory& warperFactory,
                                    const ImageFlowFactory& flowFactory, Input::ReaderFactory* readerFactory,
                                    const StereoRigDefinition* rig = nullptr);

  virtual ~ControllerImpl();

  virtual Status createStitcher() override;
  virtual void deleteStitcher() override;

  virtual bool addAudioOutput(std::shared_ptr<VideoStitch::Output::AudioWriter>) override;
  virtual bool removeAudioOutput(const std::string&) override;
  virtual void setAudioInput(const std::string& inputName) override;

  virtual Potential<ExtractOutput> createBlockingExtractOutput(
      int source, std::shared_ptr<SourceSurface>, std::shared_ptr<SourceRenderer> renderer,
      std::shared_ptr<VideoStitch::Output::VideoWriter> writer) override;

  virtual Potential<ExtractOutput> createAsyncExtractOutput(
      int source, const std::vector<std::shared_ptr<SourceSurface>>&, std::shared_ptr<SourceRenderer> renderer,
      std::shared_ptr<VideoStitch::Output::VideoWriter> writer) const override;

  virtual PotentialOutput createBlockingStitchOutput(std::shared_ptr<PanoSurface>,
                                                     const std::vector<std::shared_ptr<PanoRenderer>>& renderer,
                                                     const std::vector<std::shared_ptr<Writer>>& writers) override;

  virtual PotentialOutput createAsyncStitchOutput(const std::vector<std::shared_ptr<PanoSurface>>&,
                                                  const std::vector<std::shared_ptr<PanoRenderer>>& renderer,
                                                  const std::vector<std::shared_ptr<Writer>>& writers) const override;

  virtual ControllerStatus stitch(Output* output, bool readFrame) override;

  virtual ControllerStatus extract(ExtractOutput* output, bool readFrame) override;

  virtual ControllerStatus extract(std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo,
                                   bool readFrame) override;

  virtual ControllerStatus stitchAndExtract(Output* output, std::vector<ExtractOutput*> extracts, AlgorithmOutput* algo,
                                            bool readFrame) override;

  virtual const PanoDefinition& getPano() const override { return *pano; }

  virtual const StereoRigDefinition* getRig() const override { return rig; }

  virtual bool isPanoChangeCompatible(const PanoDefinition& newPano) const override;

  virtual const ImageMergerFactory& getMergerFactory() const override { return *mergerFactory; }

  virtual const ImageWarperFactory& getWarperFactory() const override { return *warperFactory; }

  virtual const ImageFlowFactory& getFlowFactory() const override { return *flowFactory; }

  virtual int getCurrentFrame() const override { return readerController->getCurrentFrame(); }

  virtual Status updatePanorama(
      const std::function<Potential<PanoDefinition>(const PanoDefinition&)>& panoramaUpdater) override;
  virtual Status updatePanorama(const PanoDefinition& panorama) override;

  virtual Status resetRig(const StereoRigDefinition& newRig) override;
  virtual Status applyAudioProcessorParam(const AudioPipeDefinition& newAudioPipe) override;

  virtual Status resetMergerFactory(const ImageMergerFactory& newMergerFactory, bool redoSetupNow) override;

  virtual Status resetWarperFactory(const ImageWarperFactory& newWarperFactory, bool redoSetupNow) override;

  virtual Status resetFlowFactory(const ImageFlowFactory& newFlowFactory, bool redoSetupNow) override;

  virtual void applyRotation(double yaw, double pitch, double roll) override;

  virtual void resetRotation() override;
  virtual Quaternion<double> getRotation() const override;

  virtual void setSphereScale(const double sphereScale) override;

  ReaderController& getReaderCtrl() const { return *readerController; }

  std::vector<PreProcessor*> getPreProcessors() const { return preprocessors; }

  /**
   * Preprocessor accessor.
   * @returns The i-th preprocessor, or NULL if no preprocessor.
   * @note Inputs must be locked by the calling thread.
   */
  PreProcessor* getPreProcessor(int i) const;

  /**
   * Preprocessor setter.
   * @note Inputs must be locked by the calling thread.
   * @note The current i-th preprocessor will be deleted and replaced by @p.
   * @note @p ownership is tranferred to the ControllerImpl.
   */
  void setPreProcessor(int i, PreProcessor* p) override;

  /**
   * @brief enablePreProcessing
   * @param value enables preprocessing according to boolean value
   */
  void enablePreProcessing(bool value) override;

  /**
   * Enables/disables metadata processing.
   */
  virtual void enableMetadataProcessing(bool value) override;

  /**
   * Postprocessor accessor.
   */
  PostProcessor* getPostProcessor() const;

  /**
   * Returns true if preprocessing is enabled.
   */
  bool isPreProcessingEnabled() const { return preProcessingEnabled; }

  /**
   * Postprocessor setter.
   * @note The current postprocessor will be deleted and replaced by @p.
   * @note @p ownership is tranferred to the ControllerImpl.
   */
  void setPostProcessor(PostProcessor* p) override;

  /**
   * AudioPreprocessor setter.
   * @param name Name of the audio preprocessor to setup.
   * @param gr Group  id of the audio preprocessor to be applied.
   */
  void setAudioPreProcessor(const std::string& name, groupid_t gr) override {
    readerController->setupAudioPreProc(name, gr);
  }

  /**
   * Audio accessors.
   */
  bool hasAudio() const override { return audioPipe->hasAudio(); }

  Status seekFrame(frameid_t date) override { return readerController->seekFrame(date); }

  virtual bool hasVuMeter(const std::string& inputName) const override;

  virtual std::vector<double> getPeakValues(const std::string& inputName) const override;

  virtual std::vector<double> getRMSValues(const std::string& inputName) const override;

  virtual Status setAudioDelay(double delay_ms) override;

  /**
   * IMU Stabilization accessors
   */
  const Quaternion<double> getUserOrientation() override;
  void setUserOrientation(const Quaternion<double>& q) override;
  void updateUserOrientation(const Quaternion<double>& q) override;
  void resetUserOrientation() override;

  /**
   * @brief enables (true) or disables (false) the IMU stabilization
   *  It always resets previous user defined orientation
   */
  void enableStabilization(bool value) override;
  virtual bool isStabilizationEnabled() override;

  Stab::IMUStabilization& getStabilizationIMU() override;

  virtual mtime_t getLatency() const override;

  virtual Status addSink(const Ptv::Value* config) override;
  virtual void removeSink() override;

 protected:
  /**
   * Create a controller. Call init() after creation.
   * @param pano The panorama to stitch.
   * @param mergerFactory The factory for mergers.
   * @param readers Vector of the intput readers.
   * @param preprocessors Vector of the preprocessors.
   * @param maxStitchers Maximum number of stitchers that can live in parallel.
   * @param options controller options.
   * @param frameid_t Initial frame offset.
   */
  ControllerImpl(const PanoDefinition& pano, Audio::AudioPipeline* audioPipe, const ImageMergerFactory& mergerFactory,
                 const ImageWarperFactory& warperFactory, const ImageFlowFactory& flowFactory,
                 ReaderController* readerController, std::vector<PreProcessor*> preprocessors, PostProcessor* postproc,
                 const StereoRigDefinition*);

  virtual Status resetPano(const PanoDefinition& newPano) override;

 private:
  PanoDefinition* pano;
  const StereoRigDefinition* rig;

  std::timed_mutex panoramaUpdateLock;

  const ImageMergerFactory* mergerFactory;
  const ImageWarperFactory* warperFactory;
  const ImageFlowFactory* flowFactory;

  bool setupPending;

  std::vector<PreProcessor*> preprocessors;
  std::atomic<bool> preProcessingEnabled;
  std::atomic<bool> metadataProcessingEnabled;

  PostProcessor* postprocessor;

  mutable std::mutex stitcherMutex;  // protects pipeline.
  Audio::AudioPipeline* audioPipe;
  VideoPipeline* videoPipe;

  Stab::IMUStabilization stabilizationAlgorithm;
  Quaternion<double> qUserOrientation;
  bool stabilizationEnabled;

  Exposure::MetadataProcessor exposureProcessor;
};

#define PROPAGATE_CONTROLLER_FAILURE_STATUS(call) \
  {                                               \
    const ControllerStatus status = (call);       \
    if (!status.ok()) {                           \
      return status;                              \
    }                                             \
  }
#define FAIL_CONTROLLER_RETURN PROPAGATE_CONTROLLER_FAILURE_STATUS
}  // namespace Core
}  // namespace VideoStitch

#ifdef _MSC_VER
#pragma warning(pop)
#endif
