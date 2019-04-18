// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "autoCrop.hpp"

#include "gpu/memcpy.hpp"
#include "util/pngutil.hpp"
#include "util/geometryProcessingUtils.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <random>
#include <stack>
#include <vector>

#ifndef CERESLIB_UNSUPPORTED
#if _MSC_VER
// To disable warnings on the external ceres library
#pragma warning(push)
#pragma warning(disable : 4127)
#include <ceres/ceres.h>
#pragma warning(pop)
#else
#include <ceres/ceres.h>
#endif
#endif

//#define AUTOCROP_DEBUG

#ifdef AUTOCROP_DEBUG
#ifdef NDEBUG
#error "This is not supposed to be included in non-debug mode."
#endif

#include "util/pnm.hpp"
#include "util/debugUtils.hpp"
#endif

namespace VideoStitch {
namespace AutoCrop {

static const int rows[4] = {-1, 0, 0, 1};
static const int cols[4] = {0, -1, 1, 0};

template <typename T>
bool AutoCrop::DistanceFromCircleCost::operator()(const T* const x, const T* const y,
                                                  const T* const m,  // r = m^2
                                                  T* residual) const {
  // Since the radius is parameterized as m^2, unpack m to get r.
  T r = *m * *m;
  // Get the position of the sample in the circle's coordinate system.
  T xp = xx_ - *x;
  T yp = yy_ - *y;
  // I use the following cost:
  //
  residual[0] = ww_ * (r - sqrt(xp * xp + yp * yp));
  // which is the distance of the sample from the circle. This works
  // reasonably well, but the sqrt() adds strong nonlinearities to the cost function.

  // A different cost, residual[0] = r*r - xp*xp - yp*yp;
  // which while not strictly a distance in the metric sense
  // (it has units distance^2) it can produce more robust fits when there
  // are outliers. This is because the cost surface is more convex.

  // I tested both functions and the first one seems to give better results
  return true;
}

AutoCrop::AutoCrop(const AutoCropConfig& config) : autoCropConfig(config) {}

AutoCrop::~AutoCrop() {}

Status AutoCrop::setupImage(const cv::Mat& inputImage) {
  if (inputImage.rows == 0 || inputImage.cols == 0) {
    return {Origin::CropAlgorithm, ErrType::InvalidConfiguration, "Input image dimensions are zero"};
  }
  inputCvImage = inputImage.clone();
  cv::Mat blurredImage;
  // First, perform gaussian filter on the input image
  cv::GaussianBlur(
      inputCvImage, blurredImage,
      cv::Size((int)autoCropConfig.getGaussianBlurKernelSize(), (int)autoCropConfig.getGaussianBlurKernelSize()),
      autoCropConfig.getGaussianBlurSigma(), 0);
  cv::cvtColor(blurredImage, inputLabImage, CV_BGR2Lab);

  cv::Size downSize = cv::Size(inputLabImage.cols, inputLabImage.rows);
  while (downSize.width > 512 && downSize.height > 512) {
    downSize /= 2;
  }

  cv::resize(inputLabImage, downLabImage, downSize);
  ratio = cv::Size2f(float(inputImage.cols) / downLabImage.cols, float(inputImage.rows) / downLabImage.rows);
  inputSize = cv::Size(downLabImage.cols, downLabImage.rows);
  inputColors.resize(inputSize.width * inputSize.height, cv::Vec3b(0, 0, 0));
  for (int j = 0; j < downLabImage.cols; j++) {
    for (int i = 0; i < downLabImage.rows; i++) {
      inputColors[i * inputSize.width + j] = downLabImage.at<cv::Vec3b>(cv::Point(j, i));
    }
  }

#ifdef AUTOCROP_DEBUG
  std::vector<unsigned char> dumpVector(inputColors.size() * 4);
  for (int i = 0; i < inputColors.size(); i++) {
    dumpVector[4 * i] = inputColors[i][0];
    dumpVector[4 * i + 1] = inputColors[i][1];
    dumpVector[4 * i + 2] = inputColors[i][2];
    dumpVector[4 * i + 3] = 255;
  }
  Util::PngReader writer;
  writer.writeRGBAToFile("inputImage.png", inputSize.width, inputSize.height, &dumpVector.front());
#endif
  return Status::OK();
}

Status AutoCrop::findCropCircle(const cv::Mat& inputImage, cv::Point3i& circle) {
  circle = cv::Point3i(0, 0, 0);
  // Prepare image: downscale, turn to LAB
  FAIL_RETURN(setupImage(inputImage));

  // Image binarization
  binaryLabels.clear();
  findValidPixel((int)autoCropConfig.getNeighborThreshold(), (int)autoCropConfig.getDifferenceThreshold());

  // Remove all small connected components
  removeSmallDisconnectedComponent();

  // Find all border pixels
  std::vector<cv::Point> points;
  FAIL_RETURN(findBorderPixels(points));

  // Find the convex hull and perform sampling
  std::vector<cv::Point> convexHullPoints;
  std::vector<float> convexHullPointWeights;
  FAIL_RETURN(findConvexHullBorder(downLabImage, points, convexHullPoints, convexHullPointWeights));

  // Find the inscribed circle
  cv::Point3d c(0, 0, 0);
  FAIL_RETURN(findInscribedCircleCeres(convexHullPoints, convexHullPointWeights, c));
  cv::Point3d coarseCircle =
      cv::Point3d(c.x * ratio.width, c.y * ratio.height, c.z * (ratio.width + ratio.height) / 2.0f);
#ifdef AUTOCROP_DEBUG
  { dumpCircleFile(coarseCircle, "initCoarse"); }
#endif

  // Use the coarse circle as initialization for the refined circle
  cv::Point3d refinedCircle;
  FAIL_RETURN(findRefinedCircle(coarseCircle, refinedCircle));
  circle = cv::Point3i((int)std::round(refinedCircle.x), (int)std::round(refinedCircle.y),
                       (int)std::round(refinedCircle.z * autoCropConfig.getScaleRadius()));

  // Based on the assumption that the four corners are not covered by the circle,
  // have a test to reject a lens if the circle is out of bounds.
  const int radiusSqr = circle.z * circle.z;
  std::vector<cv::Point> corners = {cv::Point(0, 0), cv::Point(inputImage.cols - 1, 0),
                                    cv::Point(inputImage.cols - 1, inputImage.rows - 1),
                                    cv::Point(0, inputImage.rows - 1)};
  cv::Point circleCenter(circle.x, circle.y);
  for (auto corner : corners) {
    if (Util::GeometryProcessing::norm2(corner, circleCenter) < radiusSqr) {
      return {Origin::CropAlgorithm, ErrType::InvalidConfiguration, "Invalid circle detected"};
    }
  }
  return Status::OK();
}

void AutoCrop::findFineScalePoints(const std::vector<cv::Point>& circlePoints,
                                   std::vector<cv::Point>& fineTuneCirclePoints, const cv::Vec2f& direction) const {
  // From the coarse circle found in the first step, need to find a fine scale point set
  // For every point p in coarse circle, draw a random line in the direction "direction" ranging from
  // -autoCropConfig.getFineTuneMarginSize() to autoCropConfig.getFineTuneMarginSize()
  // (with p stays at position 0)
  // The fine scale point of the input p is the point with the minimum gradient in the direction "direction"
  const int fineTuneSize = (int)autoCropConfig.getFineTuneMarginSize();
  cv::Vec3b black(0, 0, 0);
  const float normalizedValue = 255.0f;
  for (size_t i = 0; i < circlePoints.size(); i++) {
    bool first = true;
    cv::Vec3b color0(0, 0, 0), color1(0, 0, 0);
    float bestCost = 0.0f;
    cv::Point bestPoint(0, 0);
    for (int j = -fineTuneSize; j <= fineTuneSize; j++) {
      const cv::Point point =
          cv::Point((int)(circlePoints[i].x + direction[0] * j), (int)(circlePoints[i].y + direction[1] * j));
      // If the point stays inside the image
      if (Util::GeometryProcessing::insideImage(point, inputCvImage)) {
        color1 = inputLabImage.at<cv::Vec3b>(point);
        if (!first) {
          cv::Vec3b intensity = inputCvImage.at<cv::Vec3b>(point);
          const float cost =
              1.0f * ((float)std::sqrt(Util::GeometryProcessing::norm2(color0, color1))) / normalizedValue +
              0.01f * ((float)std::sqrt(Util::GeometryProcessing::norm2(black, intensity))) / normalizedValue;
          if (cost > bestCost) {
            bestCost = cost;
            bestPoint = point;
          }
        }
        color0 = color1;
        first = false;
      }
    }
    // Make sure a point is only picked if it is good enough
    if (bestCost > 0.01) {
      fineTuneCirclePoints.push_back(bestPoint);
    }
  }
}

Status AutoCrop::findRefinedCircle(const cv::Point3d& inputCircle, cv::Point3d& refinedCircle) {
  std::vector<cv::Point> circlePoints;
  Util::GeometryProcessing::getUniformSampleOnCircle(autoCropConfig.getConvexHullSampledCount(),
                                                     cv::Size(inputCvImage.cols, inputCvImage.rows), inputCircle,
                                                     circlePoints);
#ifdef AUTOCROP_DEBUG
  {
    std::vector<unsigned char> dumpVector(inputLabImage.cols * inputLabImage.rows, 0);
    for (int i = 0; i < circlePoints.size(); i++)
      if (Util::GeometryProcessing::insideImage(circlePoints[i], inputCvImage)) {
        const int index = circlePoints[i].y * inputLabImage.cols + circlePoints[i].x;
        dumpVector[index] = 255;
      }
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>("border_fineSampledPoint.png", dumpVector, inputLabImage.cols,
                                                     inputLabImage.rows);
  }
#endif

  std::vector<cv::Point> fineTuneCirclePoints;
  // Find the fine scale points in the horizontal and vertical direction
  // Theorectically, adding more directions should improve the final result
  findFineScalePoints(circlePoints, fineTuneCirclePoints, cv::Vec2f(1, 0));
  findFineScalePoints(circlePoints, fineTuneCirclePoints, cv::Vec2f(0, 1));
  FAIL_RETURN(removeOutliers(fineTuneCirclePoints));

#ifdef AUTOCROP_DEBUG
  {
    std::vector<unsigned char> dumpVector(inputLabImage.cols * inputLabImage.rows, 0);
    for (int i = 0; i < fineTuneCirclePoints.size(); i++) {
      const int index = fineTuneCirclePoints[i].y * inputLabImage.cols + fineTuneCirclePoints[i].x;
      dumpVector[index] = 255;
    }
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>("border_finePointVector.png", dumpVector, inputLabImage.cols,
                                                     inputLabImage.rows);
  }
#endif

  refinedCircle = inputCircle;
  // Add 4 synthetic points at the border
  std::vector<cv::Point> borderPoints = {
      cv::Point(inputLabImage.cols / 2, 0), cv::Point(inputLabImage.cols / 2, inputLabImage.rows - 1),
      cv::Point(0, inputLabImage.rows / 2), cv::Point(inputLabImage.cols - 1, inputLabImage.rows / 2)};
  for (auto point : borderPoints) {
    if (Util::GeometryProcessing::pointInsideCircle(point, inputCircle)) {
      fineTuneCirclePoints.push_back(point);
    }
  }

  // Find the convex hull and perform sampling
  std::vector<cv::Point> convexHullPoints;
  std::vector<float> convexHullPointWeights;
  FAIL_RETURN(findConvexHullBorder(inputLabImage, fineTuneCirclePoints, convexHullPoints, convexHullPointWeights,
                                   &borderPoints));

  FAIL_RETURN(findInscribedCircleCeres(convexHullPoints, convexHullPointWeights, refinedCircle, 500));
  return Status::OK();
}

cv::Point3d AutoCrop::getInitialCircle(const std::vector<cv::Point>& points) const {
  // Take the first point as the one that are not on the borders
  int firstPointIndex = -1;
  for (size_t i = 0; i < points.size(); i++) {
    if (!Util::GeometryProcessing::onBorder(points[i], inputSize)) {
      firstPointIndex = (int)i;
      break;
    }
  }

  // Take the second point as the furthest from the first point
  int furthestDistance = 0;
  int secondPointIndex = -1;
  for (size_t i = 0; i < points.size(); i++) {
    int dist = Util::GeometryProcessing::norm2(points[firstPointIndex], points[i]);
    if (dist > furthestDistance) {
      furthestDistance = dist;
      secondPointIndex = (int)i;
    }
  }

  if (firstPointIndex >= 0 && secondPointIndex >= 0) {
    const cv::Point center = (points[firstPointIndex] + points[secondPointIndex]) / 2;
    double x = (double)center.x;
    double y = (double)center.y;
    double r = (sqrt(double((center.x - points[firstPointIndex].x) * (center.x - points[firstPointIndex].x) +
                            (center.y - points[firstPointIndex].y) * (center.y - points[firstPointIndex].y))));
    return cv::Point3d(x, y, r);
  } else {
    return cv::Point3d(inputSize.width / 2, inputSize.height / 2, inputSize.width / 2);
  }
}

// https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/circle_fit.cc
Status AutoCrop::findInscribedCircleCeres(const std::vector<cv::Point>& convexHullPoints,
                                          const std::vector<float>& convexHullPointWeights, cv::Point3d& circle,
                                          const int num_iterations) const {
  if (circle.z <= 0) {
    circle = getInitialCircle(convexHullPoints);
  }
  double x = (double)circle.x;
  double y = (double)circle.y;
  double r = (double)circle.z;
  // Parameterize r as m^2 so that it can't be negative.
  double m = sqrt(r);

#ifndef CERESLIB_UNSUPPORTED
  ceres::Problem problem;
  // Configure the loss function.
  ceres::LossFunction* loss = new ceres::CauchyLoss(0.15);
  // Add the residuals.
  for (size_t i = 0; i < convexHullPoints.size(); i++) {
    if (Util::GeometryProcessing::onBorder(convexHullPoints[i], inputSize)) {
      continue;
    }
    const double xx = convexHullPoints[i].x;
    const double yy = convexHullPoints[i].y;
    const double ww = convexHullPointWeights[i];
    ceres::CostFunction* cost =
        new ceres::AutoDiffCostFunction<DistanceFromCircleCost, 1, 1, 1, 1>(new DistanceFromCircleCost(xx, yy, ww));
    problem.AddResidualBlock(cost, loss, &x, &y, &m);
  }

  // Build and solve the problem.
  ceres::Solver::Options options;
  options.max_num_iterations = num_iterations;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.logging_type = ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (summary.termination_type != ceres::CONVERGENCE && summary.termination_type != ceres::NO_CONVERGENCE) {
    return {Origin::CropAlgorithm, ErrType::AlgorithmFailure,
            "Unable to find a matching circle. The solver did not converge."};
  }
  if (!summary.IsSolutionUsable()) {
    return {Origin::CropAlgorithm, ErrType::AlgorithmFailure,
            "Unable to find a matching circle. The solver did not find a usable solution."};
  }
  // Recover r from m.
  r = m * m;
  circle = cv::Point3d(x, y, r);
  return Status::OK();
#else
  return {Origin::CropAlgorithm, ErrType::UnsupportedAction,
          "Unable to find a matching circle. The ceres::Solver is not available."};
#endif
}

Status AutoCrop::findConvexHullBorder(const cv::Mat& image, const std::vector<cv::Point>& points,
                                      std::vector<cv::Point>& convexHullPoints,
                                      std::vector<float>& convexHullPointWeights,
                                      const std::vector<cv::Point>* borderPoints) const {
  const cv::Size size(image.cols, image.rows);
  std::vector<int> hull;
  cv::convexHull(cv::Mat(points), hull, true);
  if (hull.size() <= 3) {
    return {Origin::CropAlgorithm, ErrType::AlgorithmFailure,
            "Unable to find a valid convex hull. Hull size: " + std::to_string(hull.size())};
  }
  // Draw the convex hull
  cv::Mat convexHullMat(size.height, size.width, CV_8U);
  convexHullMat.setTo(0);
  cv::Point pt0 = points[hull.back()];
  for (size_t i = 0; i < hull.size(); i++) {
    cv::Point pt = points[hull[i]];
    if ((!Util::GeometryProcessing::pointInVector(pt, borderPoints) &&
         !Util::GeometryProcessing::pointInVector(pt0, borderPoints)) ||
        (!borderPoints)) {
      cv::line(convexHullMat, pt0, pt, cv::Scalar(255), 1, cv::LINE_4);
    }
    pt0 = pt;
  }

  // Find the non zero pixel
  std::vector<cv::Point> tracedConvexHullPoints;
  for (int j = 1; j < convexHullMat.cols - 1; j++)
    for (int i = 1; i < convexHullMat.rows - 1; i++)
      if (convexHullMat.at<unsigned char>(cv::Point(j, i)) > 0) {
        tracedConvexHullPoints.push_back(cv::Point(j, i));
      }
  if (!tracedConvexHullPoints.size()) {
    return {Origin::CropAlgorithm, ErrType::AlgorithmFailure,
            "Unable to find a valid convex hull. No traced convex hull points."};
  }
  // Sample a fixed number of points
  convexHullPoints.clear();
  std::default_random_engine gen(0);
  std::uniform_int_distribution<int> di(0, (int)(tracedConvexHullPoints.size() - 1));

  for (size_t i = 0; i < autoCropConfig.getConvexHullSampledCount(); i++) {
    const int randIndex = di(gen);
    convexHullPoints.push_back(tracedConvexHullPoints[randIndex]);
  }

  convexHullPointWeights.clear();
  const std::vector<cv::Point> neighborOffsets{cv::Point(-1, 0), cv::Point(1, 0), cv::Point(0, 1), cv::Point(0, -1)};
  for (size_t i = 0; i < convexHullPoints.size(); i++) {
    float diff = 0.0f;
    float weight = 0.0f;
    cv::Vec3b color0 = image.at<cv::Vec3b>(convexHullPoints[i]);
    for (size_t j = 0; j < neighborOffsets.size(); j++)
      if (Util::GeometryProcessing::insideImage(convexHullPoints[i] + neighborOffsets[j], image)) {
        cv::Vec3b color1 = image.at<cv::Vec3b>(convexHullPoints[i] + neighborOffsets[j]);
        diff += 1.0f * ((float)std::sqrt(Util::GeometryProcessing::norm2(color0, color1))) / 255.0f;
        weight += 1.0f;
      }
    float pointWeight = std::max(0.01f, weight > 0 ? diff / weight : 0.0f);
    convexHullPointWeights.push_back(pointWeight);
  }
#ifdef AUTOCROP_DEBUG
  {
    static int count = 0;
    std::vector<unsigned char> dumpVector(size.width * size.height, 0);
    for (int i = 0; i < convexHullPoints.size(); i++) {
      const int index = convexHullPoints[i].y * size.width + convexHullPoints[i].x;
      dumpVector[index] = 255;
    }
    std::string filename;
    if (count == 0) {
      filename = "border_convexhullVector.png";
    } else {
      filename = "border_convexhullRefinement.png";
    }
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>(filename, dumpVector, size.width, size.height);
    count++;
  }
#endif
  return Status::OK();
}

Status AutoCrop::findBorderPixels(std::vector<cv::Point>& points) const {
  points.clear();
  for (int i = 0; i < inputSize.width; i++)
    for (int j = 0; j < inputSize.height; j++) {
      if (binaryLabels[j * inputSize.width + i] > 0) {
        if (i == 0 || i == inputSize.width - 1 || j == 0 || j == inputSize.height - 1) {
          points.push_back(cv::Point(i, j));
        } else {
          for (int t = 0; t < 4; t++) {
            const cv::Point nextPoint = cv::Point(i + rows[t], j + cols[t]);
            if (nextPoint.x >= 0 && nextPoint.x < inputSize.width && nextPoint.y >= 0 &&
                nextPoint.y < inputSize.height) {
              if (binaryLabels[nextPoint.y * inputSize.width + nextPoint.x] == 0) {
                points.push_back(cv::Point(i, j));
                break;
              }
            }
          }
        }
      }
    }

#ifdef AUTOCROP_DEBUG
  {
    std::vector<unsigned char> dumpVector(inputSize.width * inputSize.height * 4, 0);
    for (int i = 0; i < points.size(); i++) {
      const int index = points[i].y * inputSize.width + points[i].x;
      dumpVector[4 * index] = 255;
      dumpVector[4 * index + 1] = 255;
      dumpVector[4 * index + 2] = 255;
      dumpVector[4 * index + 3] = 255;
    }
    Util::PngReader writer;
    writer.writeRGBAToFile("border.png", inputSize.width, inputSize.height, &dumpVector.front());
  }
#endif

  if (points.size() < 3) {
    return {Origin::CropAlgorithm, ErrType::AlgorithmFailure, "Unable to find the borders of the binary image"};
  }
  return Status::OK();
}

void AutoCrop::findValidPixel(const int moveThreshold, const int differenceThreshold) {
  binaryLabels.resize(inputColors.size(), 255);
  const int cornerBlockSize = 5;
  for (int i = 0; i <= cornerBlockSize; i++)
    for (int j = 0; j <= cornerBlockSize; j++) {
      findConnectedComponent<cv::Vec3b, unsigned char>(255, 0, moveThreshold, differenceThreshold, cv::Point(i, j),
                                                       inputSize, inputColors, binaryLabels);
      findConnectedComponent<cv::Vec3b, unsigned char>(255, 0, moveThreshold, differenceThreshold,
                                                       cv::Point(i, inputSize.height - 1 - j), inputSize, inputColors,
                                                       binaryLabels);
      findConnectedComponent<cv::Vec3b, unsigned char>(255, 0, moveThreshold, differenceThreshold,
                                                       cv::Point(inputSize.width - 1 - i, inputSize.height - 1 - j),
                                                       inputSize, inputColors, binaryLabels);
      findConnectedComponent<cv::Vec3b, unsigned char>(255, 0, moveThreshold, differenceThreshold,
                                                       cv::Point(inputSize.width - 1 - i, j), inputSize, inputColors,
                                                       binaryLabels);
    }
#ifdef AUTOCROP_DEBUG
  {
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>("binaryPixel.png", binaryLabels, inputSize.width,
                                                     inputSize.height);
  }
#endif
}

Status AutoCrop::removeOutliers(std::vector<cv::Point>& points) const {
  std::vector<cv::Point> refinedPoints;
  const int neighborCountThreshold = 10;
  const int neighborSizeThreshold = 10 * 10;
  for (size_t i = 0; i < points.size(); i++) {
    int neighborCount = 0;
    for (size_t j = 0; j < points.size(); j++) {
      if (Util::GeometryProcessing::norm2(points[i], points[j]) < neighborSizeThreshold) {
        neighborCount++;
        if (neighborCount >= neighborCountThreshold) {
          break;
        }
      }
    }
    if (neighborCount >= neighborCountThreshold) {
      refinedPoints.push_back(points[i]);
    }
  }
  points = refinedPoints;
  if (points.size() < 3) {
    return {Origin::CropAlgorithm, ErrType::AlgorithmFailure, "There are too many outliers"};
  } else {
    return Status::OK();
  }
}

void AutoCrop::removeSmallDisconnectedComponent() {
  int componentLabel = 0;
  std::vector<int> disconnectedComponentLabels(binaryLabels.size(), -1);
  std::vector<int> componentCounts;
  for (int i = 0; i < inputSize.width; i++)
    for (int j = 0; j < inputSize.height; j++) {
      if ((disconnectedComponentLabels[j * inputSize.width + i] < 0) && (binaryLabels[j * inputSize.width + i] > 0)) {
        int componentCount = findConnectedComponent<unsigned char, int>(
            -1, componentLabel, 0, 0, cv::Point(i, j), inputSize, binaryLabels, disconnectedComponentLabels);
        componentCounts.push_back(componentCount);
        componentLabel++;
      }
    }

  const int componentThreshold = int(0.01f * inputSize.width * inputSize.height);
  for (int i = 0; i < inputSize.width; i++)
    for (int j = 0; j < inputSize.height; j++) {
      int id = disconnectedComponentLabels[j * inputSize.width + i];
      if (id >= 0 && componentCounts[id] >= componentThreshold) {
        binaryLabels[j * inputSize.width + i] = 255;
      } else {
        binaryLabels[j * inputSize.width + i] = 0;
      }
    }
#ifdef AUTOCROP_DEBUG
  {
    Debug::dumpMonochromeDeviceBuffer<Debug::linear>("binaryPixel_no_small.png", binaryLabels, inputSize.width,
                                                     inputSize.height);
  }
#endif
}

Status AutoCrop::dumpCircleFile(const cv::Point3i circle, const std::string& inputFilename) const {
  cv::Mat outputImage = inputCvImage.clone();
  const cv::Point center(circle.x, circle.y);
  const int radius = circle.z;
  cv::Scalar colorCenter;
  cv::Scalar colorCircle;
  if (outputImage.channels() > 1) {
    colorCenter = cv::Scalar(0, 255, 255);
    colorCircle = cv::Scalar(0, 0, 255);
  } else {
    colorCenter = cv::Scalar(128);
    colorCircle = cv::Scalar(192);
  }
  cv::circle(outputImage, center, 3, colorCenter, -1);
  cv::circle(outputImage, center, radius, colorCircle, 5);
  std::string outputFilePath = inputFilename + "_circle.png";
  if (!Util::PngReader::writeBGRToFile(outputFilePath.c_str(), outputImage.cols, outputImage.rows, outputImage.data)) {
    return {Origin::Output, ErrType::RuntimeError, "Could not write BGR output file to path: '" + outputFilePath + "'"};
  }

  return Status::OK();
}

Status AutoCrop::dumpOriginalFile(const std::string& inputFilename) const {
  std::string outputFilePath = inputFilename + "_original.png";
  if (!Util::PngReader::writeBGRToFile(outputFilePath.c_str(), inputCvImage.cols, inputCvImage.rows,
                                       inputCvImage.data)) {
    return {Origin::Output, ErrType::RuntimeError, "Could not write BGR output file to path: '" + outputFilePath + "'"};
  }
  return Status::OK();
}

template <typename S, typename T>
int AutoCrop::findConnectedComponent(const T& notVisitedValue, const T& componentLabel, const int& moveThreshold,
                                     const int& differenceThreshold, const cv::Point& pt, const cv::Size& size,
                                     const std::vector<S>& colors, std::vector<T>& outputComponents) {
  std::stack<cv::Point> pointStack;
  pointStack.push(pt);
  const S seedColor = colors[pt.y * size.width + pt.x];
  outputComponents[pt.y * size.width + pt.x] = componentLabel;
  int count = 1;
  // As moving to the center, the point must be really similar to previous
  // in order to take it into account
  while (!pointStack.empty()) {
    const cv::Point topPoint = pointStack.top();
    pointStack.pop();
    const S topColor = colors[topPoint.y * size.width + topPoint.x];
    for (int t = 0; t < 4; t++) {
      const cv::Point nextPoint = cv::Point(topPoint.x + rows[t], topPoint.y + cols[t]);
      if (nextPoint.x >= 0 && nextPoint.x < size.width && nextPoint.y >= 0 && nextPoint.y < size.height) {
        const S nextColor = colors[nextPoint.y * size.width + nextPoint.x];
        if ((outputComponents[nextPoint.y * size.width + nextPoint.x] == notVisitedValue) &&
            (Util::GeometryProcessing::norm2(nextColor, topColor) <= moveThreshold) &&
            (Util::GeometryProcessing::norm2(nextColor, seedColor) <= differenceThreshold)) {
          outputComponents[nextPoint.y * size.width + nextPoint.x] = componentLabel;
          pointStack.push(nextPoint);
          count++;
        }
      }
    }
  }
  return count;
}

}  // namespace AutoCrop
}  // namespace VideoStitch
