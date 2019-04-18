// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "keypointMatcher.hpp"

namespace VideoStitch {
namespace Calibration {

using namespace cv;

KeypointMatcher::KeypointMatcher(const double ratio, const bool cross_check, const bool bruteForce)
    : nndrRatio(ratio), crossCheck(cross_check) {
  if (bruteForce) {
    matcher = makePtr<BFMatcher>(cv::NORM_HAMMING);
  } else {
    Ptr<flann::IndexParams> indexParams = makePtr<flann::KDTreeIndexParams>();
    Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>(100);

    indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
    searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);

    matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
  }
}

Status KeypointMatcher::match(const unsigned int frameNumber, const std::pair<unsigned int, unsigned int> &inputPair,
                              const KPList &keypointsA, const DescriptorList &descriptorsA, const KPList &keypointsB,
                              const DescriptorList &descriptorsB, Core::ControlPointList &matchedControlPoints) const {
  if (!matcher) {
    return {Origin::CalibrationAlgorithm, ErrType::SetupFailure, "Failed to instantiate keypoint matcher"};
  }

  std::vector<std::vector<DMatch> > matches, crossMatches;
  std::vector<DMatch> expectedCrossMatches;

  if (keypointsA.size() == 0 || keypointsB.size() == 0) {
    matchedControlPoints.clear();
    return {Origin::CalibrationAlgorithm, ErrType::SetupFailure, "Unable to find any key points"};
  }

  /*Match A and B*/
  matcher->knnMatch(descriptorsA, descriptorsB, matches, 2);

  if (crossCheck) {
    /*Cross-check only the points that survive the ratio test*/
    KPList crossKeypoints;
    DescriptorList crossDescriptors;
    std::vector<double> scores;

    for (size_t i = 0; i < matches.size(); ++i) {
      if (matches[i].size() < 2) {
        continue;
      }

      const DMatch &m1 = matches[i][0];
      const DMatch &m2 = matches[i][1];

      /*Check the matching quality*/
      if (m1.distance <= (float)nndrRatio * m2.distance) {
        crossKeypoints.push_back(keypointsB[m1.trainIdx]);
        crossDescriptors.push_back(descriptorsB.row(m1.trainIdx));
        expectedCrossMatches.push_back(DMatch(static_cast<int>(expectedCrossMatches.size()), m1.queryIdx, m1.distance));
        /*Keep track of the score, for the cross check*/
        const double score = (m2.distance > 0.f) ? m1.distance / m2.distance : 0.f;
        scores.push_back(score);
      }
    }

    if (!crossDescriptors.empty()) {
      matcher->knnMatch(crossDescriptors, descriptorsA, crossMatches, 1);
    }

    for (const auto &expectedMatch : expectedCrossMatches) {
      /*Make sure that m1 holds the best matched pair in the cross_matches too*/
      assert(crossMatches[expectedMatch.queryIdx][0].queryIdx == expectedMatch.queryIdx);
      if (crossMatches[expectedMatch.queryIdx][0].trainIdx != expectedMatch.trainIdx) {
        /*Did not find the reverse match in the cross_matches list, continue*/
        continue;
      }

      const double x0 = keypointsA[expectedMatch.trainIdx].pt.x;
      const double y0 = keypointsA[expectedMatch.trainIdx].pt.y;
      const double x1 = crossKeypoints[expectedMatch.queryIdx].pt.x;
      const double y1 = crossKeypoints[expectedMatch.queryIdx].pt.y;

      matchedControlPoints.push_back(Core::ControlPoint(inputPair.first, inputPair.second, x0, y0, x1, y1, frameNumber,
                                                        -1.0, scores[expectedMatch.queryIdx]));

      std::stringstream message;
      message << "Adding matched control point: " << inputPair.first << " " << inputPair.second << " "
              << " score: " << scores[expectedMatch.queryIdx] << " " << x0 << " " << y0 << " " << x1 << " " << y1
              << std::endl;
      Logger::get(Logger::Verbose) << message.str() << std::flush;
    }
  } else {
    for (size_t i = 0; i < matches.size(); ++i) {
      if (matches[i].size() < 2) {
        continue;
      }

      const DMatch &m1 = matches[i][0];
      const DMatch &m2 = matches[i][1];

      /*Check the matching quality*/
      const double score = (m2.distance > 0.f) ? m1.distance / m2.distance : 0.f;

      if (m1.distance <= (float)nndrRatio * m2.distance) {
        const double x0 = keypointsA[m1.queryIdx].pt.x;
        const double y0 = keypointsA[m1.queryIdx].pt.y;
        const double x1 = keypointsB[m1.trainIdx].pt.x;
        const double y1 = keypointsB[m1.trainIdx].pt.y;

        matchedControlPoints.push_back(
            Core::ControlPoint(inputPair.first, inputPair.second, x0, y0, x1, y1, frameNumber, -1.0, score));

        std::stringstream message;
        message << "Adding matched control point: " << inputPair.first << " " << inputPair.second << " "
                << " score: " << score << " " << x0 << " " << y0 << " " << x1 << " " << y1 << std::endl;
        Logger::get(Logger::Verbose) << message.str() << std::flush;
      }
    }
  }
  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch
