// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "motionSync.hpp"

#include "motionSyncFarneback.hpp"
#include "sequencePeaks.hpp"

#include "motion/opticalFlow.hpp"
#include "common/graph.hpp"
#include "common/thread.hpp"
#include "util/registeredAlgo.hpp"

#include "libvideostitch/logging.hpp"
#include "libvideostitch/inputDef.hpp"
#include "libvideostitch/inputFactory.hpp"
#include "libvideostitch/ptv.hpp"

#include <algorithm>
#include <atomic>
#include <sstream>
#include <set>
#include <numeric>

#define VERBOSE_DEBUG_MOTIONSYNC 0
#define USE_GOOGLE_METHOD_FOR_MST 0

#if VERBOSE_DEBUG_MOTIONSYNC
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif
#include <iostream>
#include <fstream>
#endif

namespace VideoStitch {
namespace Synchro {

namespace {
Util::RegisteredAlgo<MotionSyncAlgorithm> registered("motion_synchronization");

/*
 * Exhaustive search for frame offsets maximizing the correlation between some
 * of the global motion model components:
 * - translation magnitude
 * - roll
 *
 * See http://crcv.ucf.edu/papers/Spencer_Shah_accv_2004.pdf
 */
struct translationMagnitude {
  double operator()(const Matrix33<double>& h) const {
    return sqrt((h(0, 2) * h(0, 2) + h(1, 2) * h(1, 2)) / (h(2, 2) * h(2, 2)));
  }
};

struct rollMotion {
  double operator()(const Matrix33<double>& h) const { return h(0, 1) - h(1, 0); }
};

template <typename Measure>
void measure(const Motion::AffineMotionModelEstimation::MotionModel& hom, std::vector<double>& measure) {
  int numFrames = (int)hom.size();
  measure.reserve(numFrames);
  Measure msr;
  for (auto h = hom.begin(); h != hom.end(); ++h) {
    if (h->second.first) {
      measure.push_back(msr(h->second.second));
    } else {
      measure.push_back(0);
    }
  }
}

double phi(std::vector<double>::const_iterator x, std::vector<double>::const_iterator xend,
           std::vector<double>::const_iterator y, std::vector<double>::const_iterator yend) {
  double phi = 0.0;
  do {
    phi += (*x) * (*y);
  } while (x++ != xend && y++ != yend);
  return phi;
}

std::vector<offset_t> normalizedCrossCorrelation(const std::vector<double>& xMeasure,
                                                 const std::vector<double>& yMeasure) {
  int numFrames = (int)std::min(xMeasure.size(), yMeasure.size());
  int searchInterval = numFrames / 3;
  int corrInterval = numFrames - searchInterval - 1;

  if (!numFrames || !searchInterval) {
    Logger::get(Logger::Error) << "Motion Synchro: unable to compute normalized cross correlation" << std::endl;
    std::vector<offset_t> emptyResult;
    return emptyResult;
  }

  // self-correlation for normalized cross-correlation
  // consider all sequences [frame , frame+corrInterval]
  // for frame in [firstFrame , firstFrame+searchInterval]
  std::vector<double> xSelfCorr, ySelfCorr;
  std::vector<double>::const_iterator x = xMeasure.begin();
  std::vector<double>::const_iterator xend = x + corrInterval;
  std::vector<double>::const_iterator y = yMeasure.begin();
  std::vector<double>::const_iterator yend = y + corrInterval;
  for (; xend != xMeasure.end() && yend != yMeasure.end(); ++x, ++y, ++xend, ++yend) {
    xSelfCorr.push_back(phi(x, xend, x, xend));
    ySelfCorr.push_back(phi(y, yend, y, yend));
  }

  // offset maximizing the cross-correlation between the two sequences
  // limit the search to [ firstFrame-searchInterval ; firstFrame+searchInterval ]
  std::vector<offset_t> correlations;
  for (int offset = -searchInterval; offset < 0; ++offset) {
    double crossCorrelation = phi(xMeasure.begin() - offset, xMeasure.begin() - offset + corrInterval, yMeasure.begin(),
                                  yMeasure.begin() + corrInterval);
    crossCorrelation /= sqrt(xSelfCorr[-offset] * ySelfCorr[0]);
    correlations.push_back(std::make_pair(1 - std::abs(crossCorrelation), offset));
  }
  for (int offset = 0; offset <= searchInterval; ++offset) {
    double crossCorrelation = phi(xMeasure.begin(), xMeasure.begin() + corrInterval, yMeasure.begin() + offset,
                                  yMeasure.begin() + offset + corrInterval);
    crossCorrelation /= sqrt(xSelfCorr[0] * ySelfCorr[offset]);
    correlations.push_back(std::make_pair(1 - std::abs(crossCorrelation), offset));
  }
  return correlations;
}

}  // namespace

MotionSyncAlgorithm::MotionSyncAlgorithm(const Ptv::Value* config)
    : firstFrame(0), lastFrame(1000), rollWeight(0.), translationWeight(0.), flowMedianMagnitudeDifferenceWeight(1.) {
  if (config != NULL) {
    const Ptv::Value* value = config->has("first_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      firstFrame = value->asInt();
    }
    value = config->has("last_frame");
    if (value && value->getType() == Ptv::Value::INT) {
      lastFrame = value->asInt();
    }
    value = config->has("devices");
    if (value && value->getType() == Ptv::Value::LIST) {
      const std::vector<Ptv::Value*>& devIds = value->asList();
      for (std::vector<Ptv::Value*>::const_iterator d = devIds.begin(); d != devIds.end(); ++d) {
        value = (*d)->has("id");
        if (value && value->getType() == Ptv::Value::INT) {
          devices.push_back((int)value->asInt());
        }
      }
    }
    if (devices.size() == 0) devices.push_back(0);
  }
}

MotionSyncAlgorithm::~MotionSyncAlgorithm() {}

Potential<Ptv::Value> MotionSyncAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter* progress,
                                                 Util::OpaquePtr**) const {
  std::vector<int> offsetsFrames;
  std::vector<double> costs;

  if (pano->numVideoInputs() != pano->numInputs()) {
    return {Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration, "Some enabled inputs do not have video"};
  }

  if (firstFrame == lastFrame) {
    return Status{Origin::SynchronizationAlgorithm, ErrType::InvalidConfiguration,
                  "Not enough input frames. Please define a working sequence for this algorithm using the marquees in "
                  "the timeline."};
  }

  if (!useSpencerShahFeatures() && !useFlowMedianMagnitudeDifferencesFeatures()) {
    return Status{Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "No feature type is selected"};
  }

  bool success = true;

  FAIL_RETURN(doAlignUsingFarneback(*pano, offsetsFrames, costs, success, progress));

  if (offsetsFrames.size() != (size_t)pano->numVideoInputs()) {
    return Status{Origin::SynchronizationAlgorithm, ErrType::RuntimeError, "Inconsistent number of video inputs"};
  }

  for (readerid_t i = 0; i < (readerid_t)offsetsFrames.size(); ++i) {
    pano->getInput(i).setFrameOffset(offsetsFrames[i]);
    pano->getInput(i).setSynchroCost(costs[i]);
  }

  Ptv::Value* returnValue = Ptv::Value::emptyObject();
  returnValue->get("lowConfidence")->asBool() = !success;

  return Potential<Ptv::Value>(returnValue);
}

const char* MotionSyncAlgorithm::docString =
    "An algorithm that computes frame offsets using the global motion estimates to synchronize the inputs.\n"
    "Can be applied pre-calibration.\n"
    "The result is a { \"frames\": list of integer offsets (all >=0, in frames), \"seconds\": list of double offsets "
    "(all >=0.0, in seconds) }\n"
    "which can be used directly as a 'frame_offset' parameter for the 'inputs'.\n";

Status MotionSyncAlgorithm::doAlignUsingFarneback(const Core::PanoDefinition& pano, std::vector<int>& frames,
                                                  std::vector<double>& costs, bool& success,
                                                  ProgressReporter* progress) const {
  if (pano.numVideoInputs() < 2) {
    return Status::OK();
  }

  auto videoInputs = pano.getVideoInputs();

  if ((!useSpencerShahFeatures()) && (!useFlowMedianMagnitudeDifferencesFeatures())) {
    return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "No feature type selected"};
  }

  std::mutex modelMutex;
  std::atomic<int> frameCounter(0);
  std::atomic<int> cancellation(0);
  std::atomic<int> failure(0);

  int numCores =
      getNumCores() < static_cast<int>(pano.numVideoInputs()) ? getNumCores() : static_cast<int>(pano.numVideoInputs());
  ThreadPool workerThreadPool(numCores);

  std::vector<Motion::AffineMotionModelEstimation::MotionModel> motionModels;
  if (useSpencerShahFeatures()) {
    motionModels.resize(pano.numVideoInputs());
  }

  std::vector<int> numberOfProcessedFrames(pano.numVideoInputs());

  Input::DefaultReaderFactory readerFactory((int)firstFrame, (int)lastFrame);
  std::vector<std::shared_ptr<Input::VideoReader> > readers;
  std::vector<const Core::InputDefinition*> inputDefs;

  frameid_t realLastFrame = static_cast<frameid_t>(lastFrame);
  std::vector<std::vector<std::size_t> > vectInputsPerCore(numCores);
  for (videoreaderid_t indexSource = 0; indexSource < pano.numVideoInputs(); ++indexSource) {
    vectInputsPerCore[indexSource % numCores].push_back(indexSource);

    const Core::InputDefinition* im = &(videoInputs.at(indexSource).get());
    Potential<Input::Reader> reader = readerFactory.create(indexSource, *im);
    FAIL_CAUSE(reader.status(), Origin::Input, ErrType::SetupFailure, "Could not create reader factory");
    Input::VideoReader* videoReader = reader.release()->getVideoReader();
    if (videoReader) {
      readers.push_back(std::shared_ptr<Input::VideoReader>(videoReader));
    }
    frameid_t lastFrameCurrentReader = videoReader->getLastFrame() - videoInputs.at(indexSource).get().getFrameOffset();
    if (lastFrameCurrentReader < realLastFrame) {
      realLastFrame = lastFrameCurrentReader;
    }
    inputDefs.push_back(im);
  }
  if (realLastFrame < static_cast<frameid_t>(lastFrame)) {
    std::ostringstream oss;
    oss << "Last frame out of range. Updating last frame from " << lastFrame << " to " << realLastFrame << std::endl;
    Logger::get(Logger::Warning) << oss.str() << std::endl;
    lastFrame = realLastFrame;
  }

  std::vector<std::vector<double> > magnitudes;
  if (useFlowMedianMagnitudeDifferencesFeatures()) {
    magnitudes.resize(pano.numVideoInputs());
    for (std::size_t i = 0; i < magnitudes.size(); ++i) {
      magnitudes[i].resize(lastFrame - firstFrame);
    }
  }

  int w = static_cast<int>(readers.front()->getWidth());
  int h = static_cast<int>(readers.front()->getHeight());
  int minDim = w < h ? w : h;

  int minSize = 32;
  int downscaleFactor = 1;
  while ((minDim / (2 * downscaleFactor)) > minSize) {
    downscaleFactor *= 2;
  }

  std::vector<std::queue<VideoStitch::Motion::OpticalFlow> > opticalFlowFields;

  for (int indexCore = 0; indexCore < numCores; ++indexCore) {
    std::unique_ptr<MotionEstimationTaskFarneback> taskFarneback(new MotionEstimationTaskFarneback(
        progress, firstFrame, lastFrame, pano.numVideoInputs(), readers, inputDefs, vectInputsPerCore[indexCore],
        motionModels, magnitudes, opticalFlowFields, numberOfProcessedFrames, frameCounter, cancellation, failure,
        modelMutex, downscaleFactor));
    if (taskFarneback == nullptr) {
      return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError,
              "Could not initialize the motion estimation task"};
    }
    workerThreadPool.tryRun(taskFarneback.release());
  }

  workerThreadPool.waitAll();

  if (cancellation) {
    return Status{Origin::SynchronizationAlgorithm, ErrType::OperationAbortedByUser, "Algorithm cancelled"};
  }

  if (failure) {
    return Status{Origin::SynchronizationAlgorithm, ErrType::RuntimeError,
                  "Unable to load a frame. Check Error Log for more information"};
  }

  // Make sure the number of frames processed is consistent
  int nbFramesFirstSource = numberOfProcessedFrames.front();
  int nbTotalFrames = 0;
  for (std::size_t i = 0; i < numberOfProcessedFrames.size(); ++i) {
    if (numberOfProcessedFrames[i] != nbFramesFirstSource) {
      std::ostringstream oss;
      oss << "Inconsistent number of frames amongst inputs: ";
      oss << nbTotalFrames << " != " << numberOfProcessedFrames[i] << "  (for input " << i << ")";
      return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, oss.str()};
    }
    nbTotalFrames += numberOfProcessedFrames[i];
  }
  if (nbTotalFrames != frameCounter) {
    std::ostringstream oss;
    oss << "Inconsistent number of frames: ";
    oss << nbTotalFrames << " != " << frameCounter;
    return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, oss.str()};
  }

  for (videoreaderid_t i = 0; i < pano.numVideoInputs(); ++i) {
    if (useFlowMedianMagnitudeDifferencesFeatures()) {
      magnitudes[i].resize(frameCounter / pano.numVideoInputs());
    }
  }

  std::vector<std::vector<std::vector<double> > > features;
  std::vector<double> featuresWeights;

  Status statusFeatures = computeFeaturesForNCC(pano.numVideoInputs(), frameCounter / pano.numVideoInputs(), magnitudes,
                                                motionModels, features, featuresWeights);
  if (!statusFeatures.ok()) {
    Logger::get(Logger::Error)
        << "MotionSyncAlgorithm::doAlignUsingFarneback(): error during computation of features for NCC" << std::endl;
    return statusFeatures;
  }

  return alignGivenFeatures(features, featuresWeights, pano, frames, costs, success);
}

Status MotionSyncAlgorithm::computeFeaturesForNCC(
    std::size_t nbInputs, std::size_t nbFrames, const std::vector<std::vector<double> >& magnitudes,
    const std::vector<Motion::AffineMotionModelEstimation::MotionModel>& motionModels,
    std::vector<std::vector<std::vector<double> > >& features, std::vector<double>& featuresWeights) const {
  features.clear();
  featuresWeights.clear();

  std::vector<std::vector<double> > measuresRoll, measuresTranslation, measuresMedianFlow;
  if (useSpencerShahFeatures()) {
    if (motionModels.size() != nbInputs) {
      return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Invalid motion model size"};
    }
    measuresRoll.resize(nbInputs);
    measuresTranslation.resize(nbInputs);
  }
  if (useFlowMedianMagnitudeDifferencesFeatures()) {
    if (magnitudes.size() != nbInputs) {
      return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Invalid magnitude size"};
    }
    measuresMedianFlow.resize(nbInputs);
  }

  for (std::size_t i = 0; i < nbInputs; i++) {
    if (useSpencerShahFeatures()) {
      if (motionModels[i].size() != nbFrames) {
        std::ostringstream oss;
        oss << "Inconsistent number of features in motionModels ";
        oss << i << ": " << motionModels[i].size() << " !:= " << nbFrames;
        return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, oss.str()};
      }
      measure<rollMotion>(motionModels[i], measuresRoll[i]);
      measure<translationMagnitude>(motionModels[i], measuresTranslation[i]);
    }
    if (useFlowMedianMagnitudeDifferencesFeatures()) {
      if (magnitudes[i].size() != nbFrames) {
        std::ostringstream oss;
        oss << "Inconsistent number of features in magnitudes ";
        oss << i << ": " << magnitudes[i].size() << " !:= " << nbFrames;
        return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, oss.str()};
      }
      measuresMedianFlow[i].resize(nbFrames);
      std::adjacent_difference(magnitudes[i].begin(), magnitudes[i].end(), measuresMedianFlow[i].begin());
    }
  }

  if (useSpencerShahFeatures()) {
    features.push_back(measuresRoll);
    features.push_back(measuresTranslation);
    featuresWeights.push_back(rollWeight);
    featuresWeights.push_back(translationWeight);
  }
  if (useFlowMedianMagnitudeDifferencesFeatures()) {
    features.push_back(measuresMedianFlow);
    featuresWeights.push_back(flowMedianMagnitudeDifferenceWeight);
  }

  return Status::OK();
}

Status MotionSyncAlgorithm::alignGivenFeatures(const std::vector<std::vector<std::vector<double> > >& features,
                                               const std::vector<double>& featureWeights,
                                               const Core::PanoDefinition& pano, std::vector<int>& frames,
                                               std::vector<double>& costs, bool& success) {
  auto videoInputs = pano.getVideoInputs();

  success = true;

  if (features.empty()) {
    return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Empty feature vector"};
  }

  if (features.size() != featureWeights.size()) {
    return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Inconsistent feature weight sizes"};
  }

  // Minimum spanning tree for all correlations
  std::vector<Graph<videoreaderid_t, int>::WeightedEdge> edgesPeaks;
  std::vector<Graph<videoreaderid_t, int>::WeightedEdge> edgesGoogleInput;

  // convolution kernel used to smooth the NCC
  std::vector<float> convKernel;
  Status status = initGaussianKernel1D(2.f, 5, convKernel);
  if (!status.ok()) {
    return status;
  }

  // structure  nbInputs x nbInputs x nbOffsets to keep all the NCC costs
  std::vector<std::vector<std::vector<double> > > allOffsets(pano.numVideoInputs());
  for (std::size_t indexInput = 0; indexInput < allOffsets.size(); ++indexInput) {
    allOffsets[indexInput].resize(allOffsets.size());
  }
  int minimumOffset = std::numeric_limits<int>::max();

  std::size_t sizeNCC = 0;
  for (videoreaderid_t i = 0; i < pano.numVideoInputs() - 1; i++) {
    for (videoreaderid_t j = i + 1; j < pano.numVideoInputs(); j++) {
      std::vector<std::vector<offset_t> > offsetsVects(features.size());
      for (std::size_t featureTypeIndex = 0; featureTypeIndex < features.size(); ++featureTypeIndex) {
        offsetsVects[featureTypeIndex] =
            normalizedCrossCorrelation(features[featureTypeIndex][i], features[featureTypeIndex][j]);
        std::vector<offset_t>& offsetsCurrentFeat = offsetsVects[featureTypeIndex];

        if (offsetsCurrentFeat.empty()) {
          return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, "Offset computation failed"};
        }
        if (sizeNCC == 0) {
          sizeNCC = offsetsCurrentFeat.size();
        }
        if (offsetsCurrentFeat.size() != sizeNCC) {
          std::ostringstream oss;
          oss << "Size of NCCs are inconsistent across inputs / features types";
          oss << "   " << offsetsCurrentFeat.size() << "!= " << sizeNCC;
          return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError, oss.str()};
        }
      }

      std::vector<offset_t> offsetsMerge(sizeNCC);
      for (std::size_t indexOffset = 0; indexOffset < offsetsMerge.size(); ++indexOffset) {
        double costMerge = 0.;
        int offsetVal = 0;

        for (std::size_t featureTypeIndex = 0; featureTypeIndex < offsetsVects.size(); ++featureTypeIndex) {
          const offset_t& currentPair = offsetsVects[featureTypeIndex][indexOffset];
          if (featureTypeIndex == 0) {
            offsetVal = currentPair.second;
          }
          if (offsetVal != currentPair.second) {
            return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError,
                    "Inconsistent offset values between NCCs vectors"};
          }
          costMerge += currentPair.first * featureWeights[featureTypeIndex];
        }

        offsetsMerge[indexOffset].first = costMerge;
        offsetsMerge[indexOffset].second = offsetVal;
      }

      // smooth
      std::vector<offset_t> smoothedCorrelations;
      status = smoothUsingKernel(offsetsMerge, convKernel, smoothedCorrelations);
      if (!status.ok()) {
        return status;
      }

      // make sure minimum offset is consistent
      if (minimumOffset == std::numeric_limits<int>::max()) {
        minimumOffset = smoothedCorrelations.front().second;
      }
      if (minimumOffset != smoothedCorrelations.front().second) {
        return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError,
                "Cross-correlation sequences minimum offsets are inconsistant: " + std::to_string(minimumOffset) +
                    " != " + std::to_string(smoothedCorrelations.front().second)};
      }

      // store current pairwise costs into allOffsets structure
      allOffsets[i][j].resize(smoothedCorrelations.size());
      for (std::size_t indexCorr = 0; indexCorr < smoothedCorrelations.size(); ++indexCorr) {
        allOffsets[i][j][indexCorr] = smoothedCorrelations[indexCorr].first;
      }

      // find the most relevant peaks
      std::vector<offset_t> peaks;
      status = findMinPeaks(smoothedCorrelations, 3, 0.75f, 0.99f, peaks);
      if (!status.ok()) {
        return status;
      }

      // add the most relevant peaks into a graph
      for (std::size_t currentPeakIndex = 0; currentPeakIndex < peaks.size(); ++currentPeakIndex) {
        const offset_t& currentPair = peaks[currentPeakIndex];
        edgesPeaks.push_back(Graph<videoreaderid_t, int>::WeightedEdge(currentPair.first, i, j, currentPair.second));
        if (currentPeakIndex == 0) {
          edgesGoogleInput.push_back(
              Graph<videoreaderid_t, int>::WeightedEdge(currentPair.first, i, j, currentPair.second));
        }
      }
    }
  }

  std::vector<std::vector<Graph<videoreaderid_t, int>::WeightedEdge> > vectGraphs;
  std::vector<std::vector<Graph<videoreaderid_t, int>::WeightedEdge> > vectGraphsNextStep;

  std::set<Graph<videoreaderid_t, int> > setConsistentGraphs;
  std::pair<std::set<Graph<videoreaderid_t, int> >::iterator, bool> retGraph;
  std::set<Graph<videoreaderid_t, int> > setExploredGraphs;

  vectGraphs.push_back(edgesPeaks);
  for (std::size_t iterIndex = 0; iterIndex < 15; ++iterIndex) {
    for (std::size_t indexVectEdges = 0; indexVectEdges < vectGraphs.size(); ++indexVectEdges) {
      Graph<videoreaderid_t, int> currentGraph(vectGraphs[indexVectEdges]);
      std::vector<Graph<videoreaderid_t, int>::WeightedEdge> currentMST = currentGraph.mst();
      if (currentMST.size() != (size_t)pano.numVideoInputs() - 1) {
        continue;
      }

      std::vector<Graph<videoreaderid_t, int>::WeightedEdge> allConsistentEdges;
      status = getAllConsistentEdges(currentMST, pano.numVideoInputs(), allOffsets, minimumOffset, allConsistentEdges);
      if (!status.ok()) {
        return status;
      }

      Graph<videoreaderid_t, int> currentConsistentGraph(allConsistentEdges);
      setConsistentGraphs.insert(currentConsistentGraph);

      for (std::size_t indexEdgeInMST = 0; indexEdgeInMST < currentMST.size(); ++indexEdgeInMST) {
        std::vector<Graph<videoreaderid_t, int>::WeightedEdge> edges = vectGraphs[indexVectEdges];
        edges.erase(std::remove(edges.begin(), edges.end(), currentMST[indexEdgeInMST]), edges.end());

        Graph<videoreaderid_t, int> candidateGraph(edges);
        retGraph = setExploredGraphs.insert(candidateGraph);
        if (retGraph.second) {
          vectGraphsNextStep.push_back(edges);
        }
      }
      if (vectGraphsNextStep.size() > 10000) {
        break;
      }
    }
    vectGraphs.swap(vectGraphsNextStep);
    vectGraphsNextStep.clear();
#if VERBOSE_DEBUG_MOTIONSYNC
    std::cout << "Compute graph candidates iteration: " << iterIndex
              << "  : nb candidates: " << setConsistentGraphs.size() << "  size explore: " << vectGraphs.size()
              << std::endl;
#endif
  }

  std::vector<std::pair<double, std::size_t> > vectPairsCostsConsistentGraphs;
  std::vector<double> vectCostsConsistentGraphs;

  std::vector<Graph<videoreaderid_t, int> > vectConsistentGraphs(setConsistentGraphs.begin(),
                                                                 setConsistentGraphs.end());
  for (std::size_t indexGraph = 0; indexGraph < vectConsistentGraphs.size(); ++indexGraph) {
    double costCurrentGraph = vectConsistentGraphs[indexGraph].sumCostsAllWeigths();
    vectPairsCostsConsistentGraphs.push_back(std::make_pair(costCurrentGraph, indexGraph));
    vectCostsConsistentGraphs.push_back(costCurrentGraph);
  }

  std::sort(vectPairsCostsConsistentGraphs.begin(), vectPairsCostsConsistentGraphs.end());

  // For the time being we only use the best candidate.
  // We can use this list generate to alterative alignements if necessary
  std::size_t indexBestGraph = vectPairsCostsConsistentGraphs.front().second;
  std::vector<Graph<videoreaderid_t, int>::WeightedEdge> mst = vectConsistentGraphs.at(indexBestGraph).mst();
  if (mst.size() != (size_t)pano.numVideoInputs() - 1) {
    return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError,
            "Inconsistency: mst size != nbInputs - 1 : " + std::to_string(mst.size()) +
                "!= " + std::to_string(pano.numVideoInputs() - 1)};
  }

#if USE_GOOGLE_METHOD_FOR_MST
  std::cout << "use google method" << std::endl;
  Graph<videoreaderid_t, int> graphGoogleInput(edgesGoogleInput);
  std::vector<Graph<videoreaderid_t, int>::WeightedEdge> edgesGoogleReweighted;
  reweightGraphUsingOffsetConsistencyCriterion(graphGoogleInput, pano.numVideoInputs(), edgesGoogleReweighted);
  Graph<videoreaderid_t, int> graphGoogleReweighted(edgesGoogleReweighted);
  mst = graphGoogleReweighted.mst();

  if (mst.size() != pano.numVideoInputs() - 1) {
    std::stringstream msg;
    msg << "Inconsistency: mst size != nbInputs - 1 : " << mst.size() << " != " << pano.numVideoInputs() - 1;
    return {Origin::SynchronizationAlgorithm, ErrType::RuntimeError, msg.str()};
  }
#endif

  success = isHighConfidenceMST(mst, minimumOffset, allOffsets);

  std::vector<Graph<videoreaderid_t, int>::WeightedEdge> allConsistentEdges;
  Status statusConsistentEdges =
      getAllConsistentEdges(mst, pano.numVideoInputs(), allOffsets, minimumOffset, allConsistentEdges);
  if (!statusConsistentEdges.ok()) {
    Logger::get(Logger::Error) << "Could not retrieve the graph of consistent edges" << std::endl;
    return statusConsistentEdges;
  }
  Graph<videoreaderid_t, int> consistentGraph(allConsistentEdges);

  std::unordered_map<videoreaderid_t, double> mapCosts = consistentGraph.averageCostsOfIncidentEdges();
  std::unordered_map<videoreaderid_t, double>::iterator itMapCosts;
  costs.resize(pano.numVideoInputs(), 1.0);
  for (int indexVertex = 0; indexVertex < static_cast<int>(pano.numVideoInputs()); ++indexVertex) {
    itMapCosts = mapCosts.find(indexVertex);
    if (itMapCosts == mapCosts.end()) {
      return {Origin::SynchronizationAlgorithm, ErrType::ImplementationError,
              "Could not find the cost for vertex: " + std::to_string(indexVertex)};
    }
    costs[indexVertex] = itMapCosts->second;
#if VERBOSE_DEBUG_MOTIONSYNC
    std::cout << "Cost for vertex " << indexVertex << ": " << itMapCosts->second;
#endif
  }

  std::vector<int> offsetsInFrames;
  offsetsInFrames.resize(pano.numVideoInputs());

  // compute offsets relative to the first input
  offsetsInFrames[0] = 0;
  for (videoreaderid_t j = 1; j < pano.numVideoInputs(); ++j) {
    offsetsInFrames[j] = INT_MAX;
  }
  for (unsigned i = 0; i < mst.size(); ++i) {
    // for each edge, if the src vertex is explored and the dst vertex is unknown, set the offset
    for (std::vector<Graph<videoreaderid_t, int>::WeightedEdge>::const_iterator j = mst.begin(); j != mst.end(); ++j) {
      if (offsetsInFrames[j->getFirst()] != INT_MAX && offsetsInFrames[j->getSecond()] == INT_MAX) {
        offsetsInFrames[j->getSecond()] = offsetsInFrames[j->getFirst()] + j->getPayload();
      } else if (offsetsInFrames[j->getFirst()] == INT_MAX && offsetsInFrames[j->getSecond()] != INT_MAX) {
        offsetsInFrames[j->getFirst()] = offsetsInFrames[j->getSecond()] - j->getPayload();
      }
    }
  }

  // Find the smallest offset.
  int minOffset = 1 << 20;
  for (size_t i = 0; i < offsetsInFrames.size(); ++i) {
    const Core::InputDefinition& im = videoInputs.at(i).get();
    int offset = offsetsInFrames[i] + im.getFrameOffset();
    if (offset < minOffset) {
      minOffset = offset;
    }
  }
  frames.resize(offsetsInFrames.size());
  for (size_t i = 0; i < offsetsInFrames.size(); ++i) {
    const Core::InputDefinition& im = videoInputs.at(i);
    frames[i] = offsetsInFrames[i] + im.getFrameOffset() - minOffset;
  }

  return Status::OK();
}
}  // end namespace Synchro
}  // end namespace VideoStitch
