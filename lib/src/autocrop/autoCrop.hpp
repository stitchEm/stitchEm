// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "autoCropConfig.hpp"

#include <opencv2/core/core.hpp>

namespace VideoStitch {
namespace AutoCrop {

/**
 * @brief Auto-crop for circular fisheye camera
 */
class VS_EXPORT AutoCrop {
 public:
  class DistanceFromCircleCost {
   public:
    DistanceFromCircleCost(const double xx, const double yy, const double ww) : xx_(xx), yy_(yy), ww_(ww) {}
    template <typename T>
    bool operator()(const T* const x, const T* const y,
                    const T* const m,  // r = m^2
                    T* residual) const;

   private:
    // The measured x,y coordinate that should be on the circle.
    double xx_, yy_;

    // The weight of this measure
    double ww_;
  };

  explicit AutoCrop(const AutoCropConfig& config);
  ~AutoCrop();

  /**
   * @brief Find the circle in the inputImage, return the status
   */
  Status findCropCircle(const cv::Mat& inputImage, cv::Point3i& circle);

  /**
   * @brief Draw a circle on top of the inputImage and save the result to a file
   */
  Status dumpCircleFile(const cv::Point3i circle, const std::string& inputFilename) const;

  /**
   * @brief Dump the original image to a file
   */
  Status dumpOriginalFile(const std::string& inputFilename) const;

 private:
  /**
   * @brief Get the initial approximated circle for optimization
   */
  cv::Point3d getInitialCircle(const std::vector<cv::Point>& points) const;

  /**
   * @brief Perform pre-processing of the input image
   */
  Status setupImage(const cv::Mat& inputImage);

  /**
   * @brief Find the inscribed circle that pass set of points which minizizes a cost function (using ceres)
   * @Note: circle should be initialized before putting into this function
   */
  Status findInscribedCircleCeres(const std::vector<cv::Point>& convexHullPoints,
                                  const std::vector<float>& convexHullPointWeights, cv::Point3d& circle,
                                  const int num_iterations = 1000) const;

  /**
   * @brief Find the convex hull of the input "points"
   */
  Status findConvexHullBorder(const cv::Mat& image, const std::vector<cv::Point>& points,
                              std::vector<cv::Point>& convexHullPoints, std::vector<float>& convexHullPointWeights,
                              const std::vector<cv::Point>* borderPoints = nullptr) const;

  /**
   * @brief Perform outliers removal
   */
  Status removeOutliers(std::vector<cv::Point>& points) const;

  /**
   * @brief Find the circle border at full resolution
   */
  void findFineScalePoints(const std::vector<cv::Point>& circlePoints, std::vector<cv::Point>& fineTuneCirclePoints,
                           const cv::Vec2f& direction = cv::Vec2f(1, 0)) const;

  /**
   * @brief Find the set of border pixels in "binaryLabels"
   */
  Status findBorderPixels(std::vector<cv::Point>& points) const;

  /**
   * @brief Find the set of valid pixels, put the result into "binaryLabels"
   */
  void findValidPixel(const int moveThreshold, const int differenceThreshold);

  /**
   * @brief Remove all small disconnected components from "binaryLabels"
   */
  void removeSmallDisconnectedComponent();

  /**
   * @brief Find the refined circle after an initial circle was computed
   */
  Status findRefinedCircle(const cv::Point3d& inputCircle, cv::Point3d& refinedCircle);

  /**
   * @brief Connected component labeling
   */
  template <typename S, typename T>
  static int findConnectedComponent(const T& notVisitedValue, const T& componentLabel, const int& moveThreshold,
                                    const int& differenceThreshold, const cv::Point& pt, const cv::Size& size,
                                    const std::vector<S>& colors, std::vector<T>& outputComponents);

 private:
  cv::Size inputSize;
  cv::Size2f ratio;
  cv::Mat inputCvImage;
  cv::Mat inputLabImage;
  cv::Mat downLabImage;
  std::vector<unsigned char> binaryLabels;
  std::vector<cv::Vec3b> inputColors;
  const AutoCropConfig& autoCropConfig;
};

}  // namespace AutoCrop
}  // namespace VideoStitch
