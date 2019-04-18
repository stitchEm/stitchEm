// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Deshuffling methods for calibration

#include "calibration.hpp"

#include "calibrationUtils.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/logging.hpp"
#include <common/container.hpp>

#include <cmath>

namespace VideoStitch {
namespace Calibration {

bool Calibration::applyPermutation(Core::PanoDefinition* pano, const bool preserveReadersOrder,
                                   const std::vector<videoreaderid_t>& permutation) {
  assert(pano->numVideoInputs() == (videoreaderid_t)permutation.size());
  if (pano->numVideoInputs() != (videoreaderid_t)permutation.size()) {
    return false;
  }

  // Check that the permutation is compatible with the input sizes
  for (videoreaderid_t i = 0; i < static_cast<videoreaderid_t>(permutation.size()); ++i) {
    if (cameras[i]->getWidth() != static_cast<size_t>(pano->getVideoInput(permutation[i]).getWidth()) ||
        cameras[i]->getHeight() != static_cast<size_t>(pano->getVideoInput(permutation[i]).getHeight())) {
      return false;
    }
  }

  // Get the reader configs in the original order.
  std::vector<Ptv::Value*> readerConfigs(permutation.size());
  for (videoreaderid_t i = 0; i < static_cast<videoreaderid_t>(permutation.size()); ++i) {
    readerConfigs[i] = pano->getVideoInput(i).getReaderConfig().clone();
  }

  if (preserveReadersOrder) {
    // Deshuffle the input and camera geometries but keep the readers at the same position

    // Get the inputs and cameras in the original order.
    std::vector<Core::InputDefinition*> inputDefs(permutation.size());
    std::vector<readerid_t> inputPositions(permutation.size());
    for (videoreaderid_t i = static_cast<videoreaderid_t>(permutation.size() - 1); i >= 0; --i) {
      readerid_t inputid = pano->convertVideoInputIndexToInputIndex(i);
      inputDefs[i] = pano->popInput(inputid);
      inputPositions[i] = inputid;
    }
    auto initialCameras = cameras;

    for (videoreaderid_t i = 0; i < static_cast<videoreaderid_t>(permutation.size()); ++i) {
      // Change the reader config of the input first before inserting the input
      // because if pano is a DeferredUpdater, it will insert a clone of the input definition
      inputDefs[permutation[i]]->setReaderConfig(readerConfigs[i]);
      pano->insertInput(inputDefs[permutation[i]], inputPositions[i]);
      cameras[i] = initialCameras[permutation[i]];
    }
  } else {
    // Deshuffle the readers, input pictures and matchedpoints but keep the input geometries and cameras at the same
    // position

    RigCvImages permutedRigImages(rigInputImages.size());
    for (videoreaderid_t i = 0; i < static_cast<videoreaderid_t>(permutation.size()); ++i) {
      pano->getVideoInput(i).setReaderConfig(readerConfigs[permutation[i]]);
      permutedRigImages[i] = rigInputImages[permutation[i]];
    }
    rigInputImages.swap(permutedRigImages);

    // Deshuffle the current sets of control points
    auto lambdaFactorize = [&](std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList>& map) {
      std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> permutedMatchedPoints;
      for (auto& it : map) {
        for (auto& cp : it.second) {
          cp.index0 = permutation[cp.index0];
          cp.index1 = permutation[cp.index1];
        }
        permutedMatchedPoints[{permutation[it.first.first], permutation[it.first.second]}] = it.second;
      }
      map.swap(permutedMatchedPoints);
    };
    lambdaFactorize(matchedpoints_map);
    lambdaFactorize(configmatches_map);
  }

  return true;
}

Status Calibration::deshuffleInputs(Core::PanoDefinition& pano) {
  if (matchedpoints_map.empty()) {
    return {Origin::CalibrationAlgorithm, ErrType::AlgorithmFailure,
            "Deshuffling of inputs failed - no matched control points given. Please, rotate your rig and repeat the "
            "process."};
  }
  // prepare permutation array
  std::vector<videoreaderid_t> permutation;
  for (videoreaderid_t i = 0; i < pano.numVideoInputs(); ++i) {
    permutation.push_back(i);
  }
#if __ANDROID__
  const double numPermutations = tgamma(pano.numVideoInputs() + 1); /* factorial of pano.numVideoInputs() */
#else
  const double numPermutations = std::tgamma(pano.numVideoInputs() + 1); /* factorial of pano.numVideoInputs() */
#endif
  std::vector<videoreaderid_t> bestPermutation = permutation;
  double bestCost = std::numeric_limits<double>::max();
  int bestPermutationIndex = 0;
  auto initialmatchedpoints = matchedpoints_map;
  auto initialRigImages = rigInputImages;

  int permutationIndex = 0;

  std::stringstream report;
  if (!analyzeKeypointsConnectivity(matchedpoints_map, (videoreaderid_t)cameras.size(), &report)) {
    // non-connected camera case
    return {Origin::CalibrationAlgorithm, ErrType::AlgorithmFailure,
            report.str() +
                "Not enough control points were found, inputs are not fully connected to each other. Please, rotate "
                "your rig and repeat the process."};
  }

  /**
   * Go through all permutations, changing the readers, matched points and input pictures, keeping the input geometries
   * constant
   */
  while (std::next_permutation(permutation.begin(), permutation.end())) {
    // clone the initial panorama
    const std::unique_ptr<Core::PanoDefinition> shuffledPano(pano.clone());
    // restore the initial matchedppoints_map
    matchedpoints_map = initialmatchedpoints;
    // restore the initial rig input images
    rigInputImages = initialRigImages;
    double cost;

    // enable progress bar
    progress.enable();
    FAIL_RETURN(progress.add(CalibrationProgress::deshuffle / numPermutations, "Reordering the inputs"));
    // disable progressbar to avoid reporting filtering progress
    progress.disable();

    if (!applyPermutation(shuffledPano.get(), false, permutation)) {
      continue;
    }

    if (!filterControlPoints(pano.getSphereScale(), &cost).ok()) {
      continue;
    }
    if (!analyzeKeypointsConnectivity(globalmatches_map, (videoreaderid_t)cameras.size())) {
      // non-connected camera case
      continue;
    }
    if (bestCost > cost) {
      bestCost = cost;
      bestPermutation = permutation;
      bestPermutationIndex = permutationIndex;
    }
    permutationIndex++;
  }
  // reenable progressbar
  progress.enable();

  if (bestCost == std::numeric_limits<double>::max()) {
    return {Origin::CalibrationAlgorithm, ErrType::AlgorithmFailure, "Deshuffling of inputs failed"};
  }

  std::stringstream message;
  message << "Best deshuffling permutation " << containerToString(bestPermutation) << ", cost " << bestCost
          << std::endl;
  Logger::get(Logger::Info) << message.str();

  // Apply best permutation
  // Restore the initial matchedppoints_map
  matchedpoints_map.swap(initialmatchedpoints);
  // restore the initial rig input images
  rigInputImages.swap(initialRigImages);
  if (!applyPermutation(&pano, calibConfig.isInDeshufflePreserveReadersOrder(), bestPermutation)) {
    return {Origin::CalibrationAlgorithm, ErrType::ImplementationError,
            "Deshuffling of inputs failed - cannot apply permutation"};
  }

  // Sets the deshuffled flag, if we found a permutation other than identity
  pano.setHasBeenCalibrationDeshuffled(bestPermutationIndex != 0);

  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch
