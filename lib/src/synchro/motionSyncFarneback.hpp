// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "common/thread.hpp"
#include "motion/affineMotion.hpp"
#include "motion/opticalFlow.hpp"

#include "gpu/stream.hpp"
#include "gpu/buffer.hpp"
#include "gpu/hostBuffer.hpp"

#include "libvideostitch/algorithm.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"

#include <opencv2/core.hpp>
#include <atomic>

namespace VideoStitch {
namespace Synchro {

/**
 * @brief Convert a cv::Mat flow into a Motion::ImageSpace::MotionVectorField
 * @param flow : expected to be CV_32FC2 matrix
 * @param field : will be filled
 * @return true if everything is OK, false otherwise
 */
bool cvFlow2MotionVectorField(const cv::Mat& flow, Motion::ImageSpace::MotionVectorField& field);

class MotionEstimationTaskFarneback : public ThreadPool::Task {
 public:
  /**
   * @brief MotionEstimationTaskFarneback
   *
   * It is responsible to fill @param motionModels and @param magnitudes, processing
   * only inputs which indices are in @param inputIndexesToProcess
   *
   * @param progress : progressbar
   * @param firstFrame : first frame to process (included)
   * @param lastFrame : last frame to process (included)
   * @param nbInputs : number of input sources (number of inputs in pano)
   * @param readers : same size as @param nbInputs. Only indices in @param inputIndexesToProcess will be processed
   * @param inputDefs : same size as @param nbInputs. Only indices in @param inputIndexesToProcess will be processed
   * @param inputIndexesToProcess : vector containing the list of inputs to process (indices in @param readers)
   * @param motionModels : will be filled iff not empty
   * @param magnitudes : will be filled iff not empty
   * @param opticalFlowFields : will be filled iff not empty
   * @param numberOfProcessedFrames : same size as @param nbInputs. Only indices in @param inputIndexesToProcess will be
   * incremented
   * @param frameCounter : shared between all tasks
   * @param cancellation : shared between all tasks
   * @param failure : shared between all tasks
   * @param modelLock : mutex to access the progressbar / add in @param motionModels
   * @param downScaleFactor : size factor by which the input will be downscaled. Must be >= 1
   */
  MotionEstimationTaskFarneback(Util::Algorithm::ProgressReporter* progress, int64_t firstFrame, int64_t lastFrame,
                                std::size_t nbInputs, const std::vector<std::shared_ptr<Input::VideoReader> >& readers,
                                const std::vector<const Core::InputDefinition*>& inputDefs,
                                const std::vector<std::size_t>& inputIndexesToProcess,
                                std::vector<Motion::AffineMotionModelEstimation::MotionModel>& motionModels,
                                std::vector<std::vector<double> >& magnitudes,
                                std::vector<std::queue<VideoStitch::Motion::OpticalFlow> >& opticalFlowFields,
                                std::vector<int>& numberOfProcessedFrames, std::atomic<int>& frameCounter,
                                std::atomic<int>& cancellation, std::atomic<int>& failure, std::mutex& modelLock,
                                int downScaleFactor = 1, std::size_t maxFlowFieldSize = 0,
                                bool filterSmallMotions = false);

  ~MotionEstimationTaskFarneback();

  virtual void run();

 public:
  Status status;

 private:
  /**
   * @brief Load the next frame, resize it and put it in bufferResizedSecondFrame
   * @return OK if loading was succesful
   */
  Status loadFrame(Input::VideoReader& reader);

  Util::Algorithm::ProgressReporter* progress;

  int64_t firstFrame;
  int64_t lastFrame;
  const std::size_t nbInputs;
  const std::vector<std::shared_ptr<Input::VideoReader> > readers;
  const std::vector<const Core::InputDefinition*> inputDefs;
  const std::vector<std::size_t> inputIndexesToProcess;
  std::vector<Motion::AffineMotionModelEstimation::MotionModel>& motionModels;
  std::vector<std::vector<double> >& magnitudes;
  std::vector<std::queue<VideoStitch::Motion::OpticalFlow> >& opticalFlowFields;
  std::vector<int>& numberOfProcessedFrames;
  std::atomic<int>& frameCounter;
  std::atomic<int>& cancellation;
  std::atomic<int>& failure;

  std::vector<unsigned char> bufferResizedFirstFrame;
  std::vector<unsigned char> bufferResizedSecondFrame;
  std::size_t height;
  std::size_t width;
  cv::Mat flow;

  //// IOs
  GPU::HostBuffer<unsigned char> hostBuffer;
  GPU::Buffer<unsigned char> devBuffer;
  ////

  std::mutex& modelLock;
  int downScaleFactor;
  std::size_t maxFlowFieldSize;
  bool filterSmallMotions;
};

}  // end namespace Synchro
}  // end namespace VideoStitch
