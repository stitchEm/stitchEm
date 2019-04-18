// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "videostitcher.hpp"

#include "libvideostitch-gui/caps/signalcompressioncaps.hpp"
#include "libvideostitch-gui/mainwindow/LibLogHelpers.hpp"

#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/stitchOutput.hpp"
#include <QDir>

template <typename Controller>
VideoStitcher<Controller>::VideoStitcher(QObject* parent, typename Controller::DeviceDefinition& cudaDeviceDef,
                                         ProjectDefinition& project, Controller& controller,
                                         typename Controller::Output& stitchOutput,
                                         std::vector<VideoStitch::Core::ExtractOutput*>& extractsOutputs,
                                         ActivableAlgorithmOutput& algoOutput, QReadWriteLock& setupLk)
    : VideoStitcherSignalSlots(parent),
      cudaDeviceDefinition(cudaDeviceDef),
      project(project),
      controller(controller),
      stitchOutput(stitchOutput),
      algoOutput(algoOutput),
      extractsOutputs(extractsOutputs),
      setupLock(setupLk) {}

template <typename Controller>
VideoStitcher<Controller>::~VideoStitcher() {}

template <typename Controller>
VideoStitch::Status VideoStitcher<Controller>::init() {
  setupLock.lockForWrite();
  auto status = controller.createStitcher();
  setupLock.unlock();
  return status;
}

template <typename Controller>
void VideoStitcher<Controller>::closeProject() {
  setupLock.lockForWrite();
  controller.deleteStitcher();
  setupLock.unlock();
}

// ---------------------------- Setup ---------------------------------------

template <typename Controller>
void VideoStitcher<Controller>::resetMerger() {
  // reset right now, redo setup
  const VideoStitch::Core::ImageMergerFactory* mergerFactory = project.getImageMergerFactory().get();
  setupLock.lockForWrite();
  VideoStitch::Status s = controller.resetMergerFactory(*mergerFactory, true);
  setupLock.unlock();
  if (!s.ok()) {
    showError(VideoStitch::Core::ControllerStatus::fromError({VideoStitch::Origin::Stitcher,
                                                              VideoStitch::ErrType::SetupFailure,
                                                              "Could not apply the changes to the image merger", s}));
    closeProject();
  } else {
    restitch();
  }
}

template <typename Controller>
void VideoStitcher<Controller>::resetAdvancedBlending() {
  // reset right now, redo setup
  const VideoStitch::Core::ImageFlowFactory* flowFactory = project.getImageFlowFactory().get();
  setupLock.lockForWrite();
  VideoStitch::Status statusFlow = controller.resetFlowFactory(*flowFactory, true);
  setupLock.unlock();
  if (!statusFlow.ok()) {
    showError(VideoStitch::Core::ControllerStatus::fromError(
        {VideoStitch::Origin::Stitcher, VideoStitch::ErrType::SetupFailure,
         VideoStitcherSignalSlots::tr("Could not apply the changes to the image flow").toStdString(), statusFlow}));
    closeProject();
  }

  const VideoStitch::Core::ImageWarperFactory* warperFactory = project.getImageWarperFactory().get();
  setupLock.lockForWrite();
  VideoStitch::Status statusWarper = controller.resetWarperFactory(*warperFactory, true);
  setupLock.unlock();
  if (!statusWarper.ok()) {
    showError(VideoStitch::Core::ControllerStatus::fromError(
        {VideoStitch::Origin::Stitcher, VideoStitch::ErrType::SetupFailure,
         VideoStitcherSignalSlots::tr("Could not apply the changes to the image warper").toStdString(), statusWarper}));
    closeProject();
  }

  restitch();
}

// -------------------------- Stitching -------------------------------------

template <typename Controller>
void VideoStitcher<Controller>::stitch() {
  stitchInternal(true, false, true);
}

template <typename Controller>
void VideoStitcher<Controller>::extract() {
  stitchInternal(true, true, false);
}

template <typename Controller>
void VideoStitcher<Controller>::stitchAndExtract() {
  stitchInternal(true, true, true);
}

template <typename Controller>
void VideoStitcher<Controller>::restitch() {
  stitchInternal(false, false, true);
}

template <typename Controller>
void VideoStitcher<Controller>::reextract() {
  stitchInternal(false, true, false);
}

template <typename Controller>
void VideoStitcher<Controller>::restitchAndExtract() {
  stitchInternal(false, true, true);
}

template <typename Controller>
void VideoStitcher<Controller>::stitchInternal(bool readFrame, bool extract, bool stitch) {
  setupLock.lockForRead();
  std::vector<VideoStitch::Core::ExtractOutput*> inputFrames;
  if (extract) {
    inputFrames = extractsOutputs;
  }
  VideoStitch::Core::AlgorithmOutput* alg = nullptr;
  if (!algoOutput.toggle.test_and_set()) {
    alg = algoOutput.algoOutput;
  }

  VideoStitch::Core::ControllerStatus stitchStatus;
  if (!stitch) {
    // only extract input images
    stitchStatus = controller.extract(inputFrames, alg, readFrame);
  } else {
    stitchStatus = controller.stitchAndExtract(&stitchOutput, inputFrames, alg, readFrame);
  }
  setupLock.unlock();

  if (!stitchStatus.ok()) {
    showError(stitchStatus);
  }
}

template <typename Controller>
void VideoStitcher<Controller>::showError(const VideoStitch::Core::ControllerStatus status) {
  emit notifyErrorMessage(status, false);
}

// -------------------------- Orientation ------------------------------------

template <typename Controller>
VideoStitch::Quaternion<double> VideoStitcher<Controller>::getRotation() const {
  Q_ASSERT(QThread::currentThread() == this->thread());
  return controller.getRotation();
}

template <typename Controller>
void VideoStitcher<Controller>::resetOrientation() {
  setupLock.lockForWrite();
  controller.resetRotation();
  setupLock.unlock();
}

template <typename Controller>
VideoStitch::Quaternion<double> VideoStitcher<Controller>::getCurrentOrientation() const {
  Q_ASSERT(QThread::currentThread() == this->thread());
  return controller.getRotation();
}

template <typename Controller>
void VideoStitcher<Controller>::rotatePanorama(double yaw, double pitch, double roll) {
  setupLock.lockForWrite();
  controller.applyRotation(yaw, pitch, roll);
  setupLock.unlock();
}

// -------------------------- Snapshots -------------------------------

namespace {
// Instantiate Stereo and Mono stitchers
template <typename T>
VideoStitch::Potential<T> makeOutputWriter(VideoStitch::Ptv::Value& outputConfig, unsigned int width,
                                           unsigned int height, VideoStitch::FrameRate framerate);

template <>
VideoStitch::Potential<VideoStitch::Output::VideoWriter> makeOutputWriter(VideoStitch::Ptv::Value& outputConfig,
                                                                          unsigned int width, unsigned int height,
                                                                          VideoStitch::FrameRate framerate) {
  auto pot = VideoStitch::Output::create(outputConfig, "snapshot", width, height, framerate);
  FAIL_RETURN(pot.status());
  return pot.release()->getVideoWriter();
}

template <>
VideoStitch::Potential<VideoStitch::Output::StereoWriter> makeOutputWriter(VideoStitch::Ptv::Value& outputConfig,
                                                                           unsigned int width, unsigned int height,
                                                                           VideoStitch::FrameRate framerate) {
  VideoStitch::Potential<VideoStitch::Output::Output> writer =
      VideoStitch::Output::create(outputConfig, "snapshot", width, height, framerate);
  return VideoStitch::Output::StereoWriter::createComposition(
      writer.release()->getVideoWriter(), VideoStitch::Output::StereoWriter::VerticalLayout, AddressSpace::Host);
}
}  // namespace

template <typename Controller>
void VideoStitcher<Controller>::onSnapshotPanorama(const QString& filename) {
  const QString currentDir = QDir::currentPath();
  const QFileInfo fInfo = QFileInfo(filename);
  QDir::setCurrent(fInfo.absolutePath());
  std::unique_ptr<VideoStitch::Ptv::Value> outputConfig(VideoStitch::Ptv::Value::emptyObject());
  const auto tempByteArray = QString(fInfo.path() + QDir::separator() + fInfo.baseName());

  outputConfig->get("filename")->asString() = tempByteArray.toStdString();
  outputConfig->get("type")->asString() = fInfo.completeSuffix().toStdString();
  outputConfig->get("numbered_digits")->asInt() = 0;

  setupLock.lockForRead();

  VideoStitch::Core::PanoSurface* surf = nullptr;
  typename Controller::Output::Writer* writer = nullptr;

  if (project.getPanoConst()->getProjection() == VideoStitch::Core::PanoProjection::Equirectangular) {
    VideoStitch::Potential<typename Controller::Output::Writer> potWriter =
        makeOutputWriter<typename Controller::Output::Writer>(*outputConfig, project.getPanoConst()->getWidth(),
                                                              project.getPanoConst()->getHeight(),
                                                              controller.getFrameRate());
    // Output Writer creation error
    if (!potWriter.ok()) {
      setupLock.unlock();
      showError(VideoStitch::Core::ControllerStatus(
          {VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
           VideoStitcherSignalSlots::tr("Panorama snapshot failed").toStdString(), potWriter.status()}));
      return;
    }
    writer = potWriter.release();

    VideoStitch::Potential<VideoStitch::Core::PanoSurface> potSurf =
        VideoStitch::Core::OffscreenAllocator::createPanoSurface(
            project.getPanoConst()->getWidth(), project.getPanoConst()->getHeight(), "onSnapshotPanorama");
    if (!potSurf.ok()) {
      showError(VideoStitch::Core::ControllerStatus(
          {VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
           VideoStitcherSignalSlots::tr("Panorama snapshot failed").toStdString(), potSurf.status()}));
      return;
    }
    surf = potSurf.release();
  } else if (project.getPanoConst()->getProjection() == VideoStitch::Core::PanoProjection::Cubemap ||
             project.getPanoConst()->getProjection() == VideoStitch::Core::PanoProjection::EquiangularCubemap) {
    VideoStitch::Potential<typename Controller::Output::Writer> potWriter =
        makeOutputWriter<typename Controller::Output::Writer>(*outputConfig, project.getPanoConst()->getLength() * 3,
                                                              project.getPanoConst()->getLength() * 2,
                                                              controller.getFrameRate());
    // Output Writer creation error
    if (!potWriter.ok()) {
      setupLock.unlock();
      showError(VideoStitch::Core::ControllerStatus(
          {VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
           VideoStitcherSignalSlots::tr("Panorama snapshot failed").toStdString(), potWriter.status()}));
      return;
    }
    writer = potWriter.release();

    VideoStitch::Potential<VideoStitch::Core::CubemapSurface> potSurf =
        VideoStitch::Core::OffscreenAllocator::createCubemapSurface(
            project.getPanoConst()->getLength(), "onSnapshotPanorama",
            project.getPanoConst()->getProjection() == VideoStitch::Core::PanoProjection::EquiangularCubemap);
    if (!potSurf.ok()) {
      showError(VideoStitch::Core::ControllerStatus(
          {VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
           VideoStitcherSignalSlots::tr("Panorama snapshot failed").toStdString(), potSurf.status()}));
      return;
    }
    surf = potSurf.release();
  } else {
    showError(
        VideoStitch::Core::ControllerStatus({VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
                                             VideoStitcherSignalSlots::tr("Panorama snapshot failed").toStdString()}));
    return;
  }

  VideoStitch::Potential<typename Controller::Output> output =
      controller.createBlockingStitchOutput(std::shared_ptr<VideoStitch::Core::PanoSurface>(surf),
                                            std::shared_ptr<typename Controller::Output::Writer>(writer));
  // Stitcher writer creation error
  if (!output.ok()) {
    setupLock.unlock();
    showError(VideoStitch::Core::ControllerStatus(
        {VideoStitch::Origin::Output, VideoStitch::ErrType::SetupFailure,
         VideoStitcherSignalSlots::tr("Panorama snapshot failed").toStdString(), output.status()}));
    return;
  }

  std::unique_ptr<typename Controller::Output> releasedOutput(output.release());
  VideoStitch::Status status;
  status = controller.stitch(releasedOutput.get(), false).getStatus();
  setupLock.unlock();
  QDir::setCurrent(currentDir);

  // Stitching the output error
  if (!status.ok()) {
    showError(VideoStitch::Core::ControllerStatus(
        {VideoStitch::Origin::Stitcher, VideoStitch::ErrType::RuntimeError,
         VideoStitcherSignalSlots::tr("Panorama snapshot failed").toStdString(), status}));
    return;
  }
  emit snapshotPanoramaExported();
}

template <typename Controller>
void VideoStitcher<Controller>::onSnapshotSources(std::vector<VideoStitch::Ptv::Value*> outputConfigs) {
  VS_TH_ASSERT();
  VideoStitch::Status status = VideoStitch::Status::OK();
  setupLock.lockForRead();
  for (uint output = 0; output < outputConfigs.size(); ++output) {
    auto outputConfig = outputConfigs[output];

    auto extract = [&]() -> VideoStitch::Status {
      VideoStitch::Potential<VideoStitch::Output::Output> writer =
          VideoStitch::Output::create(*outputConfig, "jpg", outputConfig->get("width")->asInt(),
                                      outputConfig->get("height")->asInt(), controller.getFrameRate());
      FAIL_RETURN(writer.status());
      auto surf = VideoStitch::Core::OffscreenAllocator::createSourceSurface(
          project.getPanoConst()->getInput(output).getWidth(), project.getPanoConst()->getInput(output).getHeight(),
          "onSnapshotSources");
      FAIL_RETURN(surf.status());
      std::shared_ptr<VideoStitch::Output::VideoWriter> sharedWriter(writer.release()->getVideoWriter());
      VideoStitch::Potential<VideoStitch::Core::ExtractOutput> extOut = controller.createBlockingExtractOutput(
          output, std::shared_ptr<VideoStitch::Core::SourceSurface>(surf.release()), nullptr, sharedWriter);
      FAIL_RETURN(extOut.status());
      return controller.extract(extOut.object(), false).getStatus();
    };
    VideoStitch::Status extractStatus = extract();
    if (status.ok()) {
      status = extractStatus;
    }
  }
  setupLock.unlock();
  // Report one error with all the file names:
  if (!status.ok()) {
    showError(VideoStitch::Core::ControllerStatus({VideoStitch::Origin::Stitcher, VideoStitch::ErrType::RuntimeError,
                                                   VideoStitcherSignalSlots::tr("Input snapshot failed").toStdString(),
                                                   status}));
  }
  for (auto output : outputConfigs) {
    delete output;
  }
}

// explicit instantiations
template class VideoStitcher<VideoStitch::Core::Controller>;
template class VideoStitcher<VideoStitch::Core::StereoController>;
