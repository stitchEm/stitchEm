// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "epipolarCurvesAlgorithm.hpp"

#include "parse/json.hpp"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif
#include "calibration/calibrationUtils.hpp"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "calibration/keypointExtractor.hpp"
#include "calibration/keypointMatcher.hpp"
#include "core/controllerInputFrames.hpp"
#include "core/geoTransform.hpp"
#include "util/pngutil.hpp"
#include "util/registeredAlgo.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4190)
#endif
#include <opencv2/imgproc.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <random>

#define POINT_PICKING_TARGET_N_PAIRS 10000
#define POINT_PICKING_MAX_TRIALS (1000 * POINT_PICKING_TARGET_N_PAIRS)
#define MIN_HANDLED_DEPTH 0.001f
#define MAX_HANDLED_DEPTH 100.0f

namespace VideoStitch {
namespace EpipolarCurves {

namespace {
Util::RegisteredAlgo<EpipolarCurvesAlgorithm> registered("epipolar");
}

const char* EpipolarCurvesAlgorithm::docString =
    "An algorithm that shows epipolar curves in input pictures, based on provided points or automatically matched "
    "ones.\n";

static const cv::Scalar colors[] = {{0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {0, 255, 255}, {255, 255, 0}, {255, 0, 255}};

EpipolarCurvesAlgorithm::EpipolarCurvesAlgorithm(const Ptv::Value* config) : epipolarCurvesConfig(config) {}

EpipolarCurvesAlgorithm::~EpipolarCurvesAlgorithm() {}

/* Writes OpenCV image to file */
static Status dumpImageFile(const cv::Mat& image, const std::string& filename) {
  bool (*writeImageFileFunction)(const char* filename, int64_t width, int64_t height, const void* data) = nullptr;
  switch (image.type()) {
    case CV_8UC4:
      writeImageFileFunction = Util::PngReader::writeRGBAToFile;
      break;
    case CV_8UC3:
      writeImageFileFunction = Util::PngReader::writeBGRToFile;
      break;
    case CV_8UC1:
      writeImageFileFunction = Util::PngReader::writeMonochromToFile;
      break;
    default:
      return {Origin::Output, ErrType::RuntimeError, "Invalid image format"};
  }
  if (!writeImageFileFunction(filename.c_str(), image.cols, image.rows, image.data)) {
    return {Origin::Output, ErrType::RuntimeError, "Could not write output file to path: '" + filename + "'"};
  }
  return Status::OK();
}

// checks that a point can be correctly mapped
// points scaled from (0, 0, -1) are singular, they cannot be mapped correctly
static bool isMappable(const Core::SphericalCoords3& scaledPoint3d) {
  const float norm = std::sqrt(scaledPoint3d.x * scaledPoint3d.x + scaledPoint3d.y * scaledPoint3d.y +
                               scaledPoint3d.z * scaledPoint3d.z);
  return (std::abs(scaledPoint3d.z + norm) > 1.0e-6f);
}

/* Computes the Euclidian distance between two 2D points */
template <class Point2DType1, class Point2DType2>
float distance2D(const Point2DType1& pt1, const Point2DType2& pt2) {
  return std::sqrt((pt1.x - pt2.x) * (pt1.x - pt2.x) + (pt1.y - pt2.y) * (pt1.y - pt2.y));
}

/* Computes the minimum stitching distance for a 3D unit point - returns false if not possible */
static bool computeMinimumStitchingDistanceFor3DPoint(
    const Core::SphericalCoords3& refPoint3d, const std::vector<Core::TopLeftCoords2>& inputCenters,
    const std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>>& transforms,
    const Core::PanoDefinition* pano, float& stitchingDistance) {
  const float minDepth = MIN_HANDLED_DEPTH;
  const float maxDepth = MAX_HANDLED_DEPTH;

  auto isDepthFor3DPointWithinInputBounds = [&](const videoreaderid_t id, const float depth) -> bool {
    assert(depth >= 0.f);
    Core::SphericalCoords3 scaledPoint3d = refPoint3d;
    scaledPoint3d *= depth;

    if (!isMappable(scaledPoint3d)) {
      return false;
    }

    const Core::CenterCoords2 centerProjected =
        transforms[id]->mapRigSphericalToInput(pano->getVideoInput(id), scaledPoint3d, 0);
    const Core::TopLeftCoords2 topLeftProjected(centerProjected, inputCenters[id]);

    return transforms[id]->isWithinInputBounds(pano->getVideoInput(id), topLeftProjected);
  };

  // Check if refPoint3d is visible by a number of cameras when scaled at maxDepth
  std::vector<videoreaderid_t> camerasIds;

  for (videoreaderid_t i = 0; i < pano->numVideoInputs(); ++i) {
    if (isDepthFor3DPointWithinInputBounds(i, maxDepth)) {
      camerasIds.push_back(i);
    }
  }

  // We need the point to be visible by at least two cameras, if not, pick another point
  if (camerasIds.size() < 2) {
    return false;
  }

  // Find the minimum depth that will make the point be visible by less than 2 cameras
  float depth_below = minDepth;
  float depth_above = maxDepth;

  while (depth_above - depth_below >= .001) {
    float mid_depth = (depth_above + depth_below) / 2;

    size_t numberOfCamerasWherePointIsVisible = 0;

    for (auto camId : camerasIds) {
      if (isDepthFor3DPointWithinInputBounds(camId, mid_depth)) {
        ++numberOfCamerasWherePointIsVisible;
      }
    }

    if (numberOfCamerasWherePointIsVisible < 2) {
      // stitching distance is too small, less than 2 cameras still see the point
      depth_below = mid_depth;
    } else {
      // stitching distance is is too large, more than 2 cameras see the point
      depth_above = mid_depth;
    }
  }

  stitchingDistance = (depth_above + depth_below) / 2;

  return true;
}

/* Computes the minimum stitching distance for every point of the output panorama */
static cv::Mat computeMinimumStitchingDistancePerPointInOutputPanorama(
    const std::vector<Core::TopLeftCoords2>& inputCenters,
    const std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>>& transforms,
    const Core::PanoDefinition* pano, double imageMaxOutputDepth) {
  const int panoWidth = (int)pano->getWidth();
  const int panoHeight = (int)pano->getHeight();

  /* supporting equirectangular outputs only for the moment */
  cv::Mat outputEquirectangularDepthImage(panoHeight, panoWidth, CV_32FC1, cv::Scalar::all(0.));

  const float panoCenterX = panoWidth / 2.0f;
  const float panoCenterY = panoHeight / 2.0f;
  const Core::TransformStack::GeoTransform* transform0 = transforms[0].get();

  /* Parallel execution using C++11 lambda. */
  outputEquirectangularDepthImage.forEach<float>([&](float& pixel, const int position[]) -> void {
    const Core::CenterCoords2 panoCoords(position[1] - panoCenterX, position[0] - panoCenterY);
    const Core::SphericalCoords3 refPoint3d = transform0->mapPanoramaToRigSpherical(panoCoords);

    computeMinimumStitchingDistanceFor3DPoint(refPoint3d, inputCenters, transforms, pano, pixel);
  });

  cv::Mat output8bits;
  if (imageMaxOutputDepth > 0) {
    outputEquirectangularDepthImage.convertTo(output8bits, CV_8UC1, 255. / imageMaxOutputDepth);
  } else {
    Logger::get(Logger::Error) << "Invalid max output depth " << imageMaxOutputDepth
                               << " given to convert floating point depth to 8 bit gray-scale intensities" << std::endl;
  }

  return output8bits;
}

/* Computes the minimum stitching distance by randomly picking points on the sphere */
static float computeMinimumStitchingDistanceByRandomPoints(
    const std::vector<Core::TopLeftCoords2>& inputCenters,
    const std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>>& transforms,
    const Core::PanoDefinition* pano) {
  float minStitchingDistance = std::numeric_limits<float>::max();

  /* Needs translations between cameras to work */
  if (pano->hasTranslations()) {
    float maxStitchingDistance = 0.0f;
    const int target_n_pairs = POINT_PICKING_TARGET_N_PAIRS;
    const int max_trials = POINT_PICKING_MAX_TRIALS;
    int n_pairs = 0;

    std::seed_seq seed{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::mt19937 gen(seed);

    std::uniform_real_distribution<double> distribution_theta(0, 2 * M_PI);
    std::uniform_real_distribution<double> distribution_u(-1, 1);

    // Randomly generate 3D points to determine their minimum stitching distance (which is the distance where they will
    // no longer be seen by two cameras)
    for (int trial = 0; trial <= max_trials && n_pairs <= target_n_pairs; ++trial) {
      // Generate a point on the unit sphere using sphere point picking, see
      // http://mathworld.wolfram.com/SpherePointPicking.html
      double theta = distribution_theta(gen);
      double u = distribution_u(gen);

      const Core::SphericalCoords3 refPoint3d(static_cast<float>(std::sqrt(1. - u * u) * std::cos(theta)),
                                              static_cast<float>(std::sqrt(1. - u * u) * std::sin(theta)),
                                              static_cast<float>(u));

      float stitchingDistance;

      if (computeMinimumStitchingDistanceFor3DPoint(refPoint3d, inputCenters, transforms, pano, stitchingDistance)) {
        // Get the max of and min of minimum stitching distance
        if (minStitchingDistance > stitchingDistance) {
          minStitchingDistance = stitchingDistance;
        }
        if (maxStitchingDistance < stitchingDistance) {
          maxStitchingDistance = stitchingDistance;
        }
        ++n_pairs;
      }
    }
  }

  return minStitchingDistance;
}

/* Load input images given an array of frame numbers */
static Status loadInputImages(std::map<frameid_t, std::vector<cv::Mat>>& inputImages, const Core::PanoDefinition* pano,
                              const std::vector<frameid_t>& frameNumbers) {
  inputImages.clear();

  auto container = Core::ControllerInputFrames<PixelFormat::RGBA, uint32_t>::create(pano);
  FAIL_RETURN(container.status());

  for (auto& frameNumber : frameNumbers) {
    std::map<readerid_t, PotentialValue<GPU::HostBuffer<uint32_t>>> loadedFrames;

    FAIL_RETURN(container->seek(frameNumber));
    container->load(loadedFrames);

    /*Load input images for this frame number*/
    for (const auto& loadedFrame : loadedFrames) {
      readerid_t inputId = loadedFrame.first;
      auto potLoadedFrame = loadedFrame.second;
      FAIL_RETURN(potLoadedFrame.status());

      GPU::HostBuffer<uint32_t> frame = potLoadedFrame.value();

      /* Get the size of the current image */
      const Core::InputDefinition& inputDef = pano->getInput(inputId);
      const int width = static_cast<int>(inputDef.getWidth());
      const int height = static_cast<int>(inputDef.getHeight());

      auto potHostFrame =
          GPU::HostBuffer<unsigned char>::allocate(frame.numElements() * 4, "EpipolarCurves frame loading");
      FAIL_RETURN(potHostFrame.status());
      GPU::HostBuffer<unsigned char> hostFrame = potHostFrame.value();
      std::memcpy(hostFrame.hostPtr(), frame.hostPtr(), frame.byteSize());

      cv::Mat bgrImage;
      cv::Mat rgbaImage(cv::Size(width, height), CV_8UC4, frame.hostPtr(), cv::Mat::AUTO_STEP);
      cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);

      inputImages[frameNumber].push_back(bgrImage);

      hostFrame.release();
    }
  }

  return Status::OK();
}

static Status extractKeyPoints(
    std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList>& matchedPointsMap,
    std::map<videoreaderid_t, std::vector<Core::TopLeftCoords2>>& pointsMap, const std::vector<cv::Mat>& inputImages,
    const std::vector<Core::TopLeftCoords2>& inputCenters,
    const std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>>& transforms,
    const Core::PanoDefinition* pano, const double decimationCellFactor) {
  /* Extraction and description of features on a list of images */
  Calibration::KeypointExtractor kpExtractor(2, 2, 0.001);
  Calibration::KeypointMatcher kpMatcher(0.99);

  std::vector<Calibration::KPList> keypoints;
  std::vector<Calibration::DescriptorList> descriptors;
  keypoints.resize(inputImages.size());
  descriptors.resize(inputImages.size());

  /*Loop over cameras*/
  for (videoreaderid_t camId = 0; camId < (videoreaderid_t)inputImages.size(); ++camId) {
    /*Do the real extraction part*/
    FAIL_RETURN(kpExtractor.extract(inputImages[camId], keypoints[camId], descriptors[camId], cv::Mat()));
  }

  /*Perform matching for all pairs*/
  for (videoreaderid_t camId1 = 0; camId1 < (videoreaderid_t)inputImages.size() - 1; ++camId1) {
    for (videoreaderid_t camId2 = camId1 + 1; camId2 < (videoreaderid_t)inputImages.size(); ++camId2) {
      /*Raw blind matching*/
      Core::ControlPointList matched, validated, decimated;
      std::pair<videoreaderid_t, videoreaderid_t> pair{camId1, camId2};
      FAIL_RETURN(kpMatcher.match(0, pair, keypoints[camId1], descriptors[camId1], keypoints[camId2],
                                  descriptors[camId2], matched));

      /*Validate the matches by reprojecting them*/
      for (Core::ControlPoint& cp : matched) {
        const Core::TopLeftCoords2 pointCam1(static_cast<float>(cp.x0), static_cast<float>(cp.y0));
        const Core::TopLeftCoords2 pointCam2(static_cast<float>(cp.x1), static_cast<float>(cp.y1));

        const Core::CenterCoords2 centeredPointCam1(pointCam1, inputCenters[camId1]);
        const Core::CenterCoords2 centeredPointCam2(pointCam2, inputCenters[camId2]);

        // Up-lifting points to spheres far away
        const Core::SphericalCoords3 spherePointCam1 = transforms[camId1]->mapInputToRigSpherical(
            pano->getVideoInput(camId1), centeredPointCam1, 0, MAX_HANDLED_DEPTH);
        const Core::CenterCoords2 centeredPointCam1InCam2 =
            transforms[camId2]->mapRigSphericalToInput(pano->getVideoInput(camId2), spherePointCam1, 0);
        const Core::TopLeftCoords2 pointCam1InCam2(centeredPointCam1InCam2, inputCenters[camId2]);

        /* Reject points if reprojection is too far */
        if (distance2D(pointCam2, pointCam1InCam2) > 200.f) {
          continue;
        }

        cp.rx0 = pointCam1InCam2.x;
        cp.ry0 = pointCam1InCam2.y;

        validated.push_back(cp);
      }

      /*Sort and decimate the ControlPoints to limit their density*/
      validated.sort(Core::ControlPointComparator());
      Calibration::decimateSortedControlPoints(decimated, validated, pano->getVideoInput(camId1).getWidth(),
                                               pano->getVideoInput(camId1).getHeight(), decimationCellFactor);

      Logger::get(Logger::Verbose) << "Found " << matched.size() << " rough matched points between camera #" << camId1
                                   << " and camera #" << camId2 << std::endl;
      Logger::get(Logger::Verbose) << "Validated " << validated.size() << " matched points between camera #" << camId1
                                   << " and camera #" << camId2 << std::endl;
      Logger::get(Logger::Verbose) << "Decimated to " << decimated.size() << " points" << std::endl;

      /*Merging result*/
      matchedPointsMap[pair].insert(matchedPointsMap[pair].end(), decimated.begin(), decimated.end());

      /*Add matched points to single points map*/
      for (const auto& it : decimated) {
        pointsMap[it.index0].push_back(Core::TopLeftCoords2(static_cast<float>(it.x0), static_cast<float>(it.y0)));
        pointsMap[it.index1].push_back(Core::TopLeftCoords2(static_cast<float>(it.x1), static_cast<float>(it.y1)));
      }
    }
  }
  return Status::OK();
}

/* Returns a map of epipolar curves per camera, given a camera ID and 2D point */
static void computeEpipolarCurves(std::map<videoreaderid_t, std::vector<cv::Point>>& curveMap,
                                  const std::pair<videoreaderid_t, Core::TopLeftCoords2>& point,
                                  const std::vector<Core::TopLeftCoords2>& inputCenters,
                                  const std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>>& transforms,
                                  const Core::PanoDefinition* pano) {
  const videoreaderid_t fromId = point.first;
  const Core::TopLeftCoords2 topLeftPoint = point.second;
  const Core::CenterCoords2 centerPoint(topLeftPoint, inputCenters[fromId]);

  curveMap.clear();

  for (videoreaderid_t camId = 0; camId < pano->numVideoInputs(); camId++) {
    if (fromId == camId) {
      continue;
    }

    // minimum distance for which lifting points and reprojecting them is meaningful
    // (the sphere at this distance should contain both camera centers of projection)
    const float minDistance =
        std::max(transforms[fromId]->computeInputMinimumRigSphereRadius(pano->getVideoInput(fromId), 0),
                 transforms[camId]->computeInputMinimumRigSphereRadius(pano->getVideoInput(camId), 0));

    Core::TopLeftCoords2 lastAddedPoint;

    for (float distance = MAX_HANDLED_DEPTH; distance >= minDistance;
         (distance <= 3.f) ? distance -= .01f : distance -= .5f) {
      Core::SphericalCoords3 scaledPoint3d =
          transforms[fromId]->mapInputToRigSpherical(pano->getVideoInput(fromId), centerPoint, 0, distance);
      Core::CenterCoords2 centerProjected =
          transforms[camId]->mapRigSphericalToInput(pano->getVideoInput(camId), scaledPoint3d, 0);
      Core::TopLeftCoords2 topLeftProjected(centerProjected, inputCenters[camId]);

      if (transforms[camId]->isWithinInputBounds(pano->getVideoInput(camId), topLeftProjected)) {
        // add point only if there is at least a one pixel difference with the last added one
        if (curveMap[camId].empty() || distance2D(lastAddedPoint, topLeftProjected) >= 1.f) {
          lastAddedPoint = topLeftProjected;
          curveMap[camId].push_back(cv::Point(static_cast<int>(std::round(topLeftProjected.x)),
                                              static_cast<int>(std::round(topLeftProjected.y))));
        }
      }
    }
  }
}

/* Draws the epipolar curves given a map of points in several cameras */
static void drawEpipolarCurves(std::vector<cv::Mat>& inputImages,
                               const std::map<videoreaderid_t, std::vector<Core::TopLeftCoords2>>& pointsMap,
                               const std::vector<Core::TopLeftCoords2>& inputCenters,
                               const std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>>& transforms,
                               const Core::PanoDefinition* pano) {
  /* For each camera in map */
  for (const auto& it : pointsMap) {
    videoreaderid_t refId = it.first;

    cv::Scalar color = colors[refId % (sizeof(colors) / sizeof(colors[0]))];

    /* For each point for this camera */
    for (const auto& topLeftPoint : it.second) {
      std::map<videoreaderid_t, std::vector<cv::Point>> curveMap;

      /* Compute epipolar curves in other cameras */
      computeEpipolarCurves(curveMap, {refId, topLeftPoint}, inputCenters, transforms, pano);

      /* Do the actual drawing */
      for (videoreaderid_t camId = 0; camId < pano->numVideoInputs(); camId++) {
        if (refId == camId) {
          cv::circle(
              inputImages[refId],
              cv::Point(static_cast<int>(std::round(topLeftPoint.x)), static_cast<int>(std::round(topLeftPoint.y))), 10,
              color, 2);
        } else {
          if (!curveMap[camId].empty()) {
            /* If curve has a single point, draw a cross, else the curve */
            if (curveMap[camId].size() == 1) {
              cv::line(inputImages[camId], curveMap[camId][0] + cv::Point(-10, 0),
                       curveMap[camId][0] + cv::Point(10, 0), color, 2);
              cv::line(inputImages[camId], curveMap[camId][0] + cv::Point(0, -10),
                       curveMap[camId][0] + cv::Point(0, 10), color, 2);
            } else {
              cv::polylines(inputImages[camId], curveMap[camId], false, color, 2);
            }
          }
        }
      }
    }
  }
}

/* Computes and draws the depth for each point in map */
static void computeAndDrawDepths(
    std::vector<cv::Mat>& inputImages,
    const std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList>& matchedPointsMap,
    const std::vector<Core::TopLeftCoords2>& inputCenters,
    const std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>>& transforms,
    const Core::PanoDefinition* pano) {
  // Compute depth of matched points by intersecting rays out of cameras
  for (const auto& matchedPoint : matchedPointsMap) {
    const videoreaderid_t camId1 = matchedPoint.first.first;
    const videoreaderid_t camId2 = matchedPoint.first.second;

    for (const Core::ControlPoint& cp : matchedPoint.second) {
      const Core::TopLeftCoords2 pointCam1(static_cast<float>(cp.x0), static_cast<float>(cp.y0));
      const Core::TopLeftCoords2 pointCam2(static_cast<float>(cp.x1), static_cast<float>(cp.y1));

      const Core::CenterCoords2 centeredPointCam1(pointCam1, inputCenters[camId1]);
      const Core::CenterCoords2 centeredPointCam2(pointCam2, inputCenters[camId2]);

      // Up-lifting points to spheres of radius 1.0 and 2.0
      const Core::SphericalCoords3 firstScaledPointCam1 = transforms[camId1]->mapInputToScaledCameraSphereInRigBase(
          pano->getVideoInput(camId1), centeredPointCam1, 0, 1.0);
      const Core::SphericalCoords3 secondScaledPointCam1 = transforms[camId1]->mapInputToScaledCameraSphereInRigBase(
          pano->getVideoInput(camId1), centeredPointCam1, 0, 2.0);
      const Core::SphericalCoords3 firstScaledPointCam2 = transforms[camId2]->mapInputToScaledCameraSphereInRigBase(
          pano->getVideoInput(camId2), centeredPointCam2, 0, 1.0);
      const Core::SphericalCoords3 secondScaledPointCam2 = transforms[camId2]->mapInputToScaledCameraSphereInRigBase(
          pano->getVideoInput(camId2), centeredPointCam2, 0, 2.0);

      // Get distance of closest points between rays (or skew lines)
      // https://en.wikipedia.org/wiki/Skew_lines#Nearest_Points
      // Reusing the notations of the Wikipedia article

      // Get vectors of rays out of cameras
      // Using cv::Vec3d for points to ease arithmetic operations on them
      const cv::Vec3d p1(firstScaledPointCam1.x, firstScaledPointCam1.y, firstScaledPointCam1.z);
      const cv::Vec3d p2(firstScaledPointCam2.x, firstScaledPointCam2.y, firstScaledPointCam2.z);
      // Ray unit vectors out of the cameras (note that they have unit norm)
      const cv::Vec3d d1(secondScaledPointCam1.x - firstScaledPointCam1.x,
                         secondScaledPointCam1.y - firstScaledPointCam1.y,
                         secondScaledPointCam1.z - firstScaledPointCam1.z);
      const cv::Vec3d d2(secondScaledPointCam2.x - firstScaledPointCam2.x,
                         secondScaledPointCam2.y - firstScaledPointCam2.y,
                         secondScaledPointCam2.z - firstScaledPointCam2.z);

      const double u = d1.dot(d2);
      if (std::abs(u - 1.) > std::numeric_limits<double>::epsilon()) {
        cv::Vec3d n = d1.cross(d2);
        cv::Vec3d n1 = d1.cross(n);
        cv::Vec3d n2 = d2.cross(n);

        const double t1 = (p2 - p1).dot(n2) / d1.dot(n2);
        const double t2 = (p1 - p2).dot(n1) / d2.dot(n1);

        // c1 and c2 form the shortest line segment joining skew lines from cam1 and cam2
        cv::Vec3d c1 = p1 + t1 * d1;
        cv::Vec3d c2 = p2 + t2 * d2;

        /* Depth of closest points on skew lines in world space */
        const double depthCam1 = std::sqrt(c1.dot(c1));
        const double depthCam2 = std::sqrt(c2.dot(c2));

        Logger::get(Logger::Debug) << "distance p1 p2 " << std::sqrt((c1 - c2).dot(c1 - c2)) << std::endl;

        // Reprojecting centeredPointCam1 onto Cam2, to know if we are within the input bounds
        const double scaleForPointCam1 =
            t1 + 1 /* because t1 is the distance starting at firstScaledPointCam1, already at camera sphere scale 1 */;
        const Core::SphericalCoords3 pointCam1ToD1 = transforms[camId1]->mapInputToScaledCameraSphereInRigBase(
            pano->getVideoInput(camId1), centeredPointCam1, 0, static_cast<float>(scaleForPointCam1));
        const Core::CenterCoords2 pointCam1ToCam2 =
            transforms[camId2]->mapRigSphericalToInput(pano->getVideoInput(camId2), pointCam1ToD1, 0);
        const Core::TopLeftCoords2 reprojected(pointCam1ToCam2, inputCenters[camId2]);

        Logger::get(Logger::Debug) << "cam1 to cam2 " << reprojected.x << ", " << reprojected.y << ", cam2 " << cp.x1
                                   << ", " << cp.y1 << ", distance " << distance2D(reprojected, pointCam2)
                                   << ", rx0, ry0 " << cp.rx0 << ", " << cp.ry0 << ", distance "
                                   << distance2D(cv::Point2f(static_cast<float>(cp.rx0), static_cast<float>(cp.ry0)),
                                                 pointCam2)
                                   << std::endl;

        Logger::get(Logger::Debug) << "depth 1 " << depthCam1 << std::endl;
        Logger::get(Logger::Debug) << "depth 2 " << depthCam2 << std::endl;

        cv::Scalar colorCamId1 = colors[camId1 % (sizeof(colors) / sizeof(colors[0]))];
        cv::Scalar colorCamId2 = colors[camId2 % (sizeof(colors) / sizeof(colors[0]))];

        /* Draw the depth value in centimeters next to the points */
        if (transforms[camId2]->isWithinInputBounds(pano->getVideoInput(camId2), reprojected)) {
          cv::putText(inputImages[camId1], std::to_string(int(std::round(depthCam1 * 100.f))),
                      cv::Point(static_cast<int>(std::round(pointCam1.x + 20.f)),
                                static_cast<int>(std::round(pointCam1.y - 10.f))),
                      cv::FONT_HERSHEY_SIMPLEX, 1, colorCamId1, 2, cv::LINE_AA);
          cv::putText(inputImages[camId2], std::to_string(int(std::round(depthCam2 * 100.f))),
                      cv::Point(static_cast<int>(std::round(pointCam2.x + 20.f)),
                                static_cast<int>(std::round(pointCam2.y - 10.f))),
                      cv::FONT_HERSHEY_SIMPLEX, 1, colorCamId2, 2, cv::LINE_AA);
        } else {
          cv::putText(inputImages[camId1], "unknown",
                      cv::Point(static_cast<int>(std::round(pointCam1.x + 20.f)),
                                static_cast<int>(std::round(pointCam1.y - 10.f))),
                      cv::FONT_ITALIC, 1, colorCamId1, 2, cv::LINE_AA);
          cv::putText(inputImages[camId2], "unknown",
                      cv::Point(static_cast<int>(std::round(pointCam2.x + 20.f)),
                                static_cast<int>(std::round(pointCam2.y - 10.f))),
                      cv::FONT_ITALIC, 1, colorCamId2, 2, cv::LINE_AA);
        }
      }
    }
  }
}

/* Computes and draws a spherical grid in the input pictures */
static void computeAndDrawSphericalGrid(
    std::vector<cv::Mat>& inputImages, const float sphereScale, const std::vector<Core::TopLeftCoords2>& inputCenters,
    const std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>>& transforms,
    const Core::PanoDefinition* pano) {
  float longitude, latitude;
  videoreaderid_t camId;
  std::vector<cv::Point> curve;

  auto drawAndFlushCurve = [&]() {
    const int npts = static_cast<int>(curve.size());
    if (npts) {
      const cv::Point* pts = curve.data();
      cv::polylines(inputImages[camId], &pts, &npts, 1, false, colors[camId]);
      curve.clear();
    }
  };

  /* Lambda function to add a point to a curve if it falls within input picture, and draw and flush the curve if it
   * falls outside */
  auto addPointToCurve = [&]() {
    Core::SphericalCoords3 scaledPoint3d(sphereScale * std::cos(latitude) * std::sin(longitude),
                                         sphereScale * std::sin(latitude),
                                         sphereScale * std::cos(latitude) * std::cos(longitude));

    if (!isMappable(scaledPoint3d)) {
      return;
    }

    Core::CenterCoords2 centerProjected =
        transforms[camId]->mapRigSphericalToInput(pano->getVideoInput(camId), scaledPoint3d, 0);
    Core::TopLeftCoords2 topLeftProjected(centerProjected, inputCenters[camId]);

    if (transforms[camId]->isWithinInputBounds(pano->getVideoInput(camId), topLeftProjected)) {
      curve.push_back(cv::Point(static_cast<int>(std::round(topLeftProjected.x)),
                                static_cast<int>(std::round(topLeftProjected.y))));
    } else {
      drawAndFlushCurve();
    }
  };

  const float F_PI = static_cast<float>(M_PI);

  for (camId = 0; camId < pano->numVideoInputs(); camId++) {
    /* Project 20 latitude lines in the camId input picture */
    for (latitude = -F_PI / 2; latitude <= F_PI / 2; latitude += F_PI / 20) {
      for (longitude = -F_PI; longitude < F_PI; longitude += 2 * F_PI / 2000) {
        addPointToCurve();
      }
      drawAndFlushCurve();
    }
    /* Then project 20 longitude lines */
    for (longitude = -F_PI; longitude < F_PI; longitude += 2 * F_PI / 20) {
      for (latitude = -F_PI / 2; latitude <= F_PI / 2; latitude += F_PI / 2000) {
        addPointToCurve();
      }
      drawAndFlushCurve();
    }
  }
}

Potential<Ptv::Value> EpipolarCurvesAlgorithm::apply(Core::PanoDefinition* pano, ProgressReporter*,
                                                     Util::OpaquePtr**) const {
  if (!epipolarCurvesConfig.isValid()) {
    return {Origin::EpipolarCurvesAlgorithm, ErrType::InvalidConfiguration, "Invalid configuration for algorithm"};
  }

  /* Prepare transforms and input centers coordinates */
  std::vector<std::unique_ptr<Core::TransformStack::GeoTransform>> transforms;
  std::vector<Core::TopLeftCoords2> inputCenters;
  for (const auto& videoInputDef : pano->getVideoInputs()) {
    transforms.push_back(std::unique_ptr<Core::TransformStack::GeoTransform>(
        Core::TransformStack::GeoTransform::create(*pano, videoInputDef)));
    inputCenters.push_back(Core::TopLeftCoords2(static_cast<float>(videoInputDef.get().getWidth() / 2),
                                                static_cast<float>(videoInputDef.get().getHeight() / 2)));
  }

  /* Get minimum stitching distance by randomly picking points on the sphere */
  const float minStitchingDistance = computeMinimumStitchingDistanceByRandomPoints(inputCenters, transforms, pano);
  Logger::get(Logger::Info) << "Minimum stitching distance computed from Pano geometry " << minStitchingDistance
                            << std::endl;

  /* Get minimum stitching distance for every panorama output point and generate a picture */
  const cv::Mat minStitchingDistanceInOutputPanoram = computeMinimumStitchingDistancePerPointInOutputPanorama(
      inputCenters, transforms, pano, epipolarCurvesConfig.getImageMaxOutputDepth());
  FAIL_RETURN(dumpImageFile(minStitchingDistanceInOutputPanoram, "output_min_stitching_distance.png"));

  /* Draw depth for each point in map by triangulation, if we have translations between cameras */

  /* Load input images */
  std::vector<frameid_t> frameNumbers = epipolarCurvesConfig.getFrames();
  std::map<frameid_t, std::vector<cv::Mat>> inputImagesMap;

  FAIL_RETURN(loadInputImages(inputImagesMap, pano, frameNumbers));

  /* For each input frame number, extract keypoints, draw epipolar curves and depths */
  for (auto& frameNumber : epipolarCurvesConfig.getFrames()) {
    assert(pano->numVideoInputs() == (videoreaderid_t)inputImagesMap[frameNumber].size());

    std::map<videoreaderid_t, std::vector<Core::TopLeftCoords2>> pointsMap = epipolarCurvesConfig.getSinglePointsMap();
    std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> matchedPointsMap =
        epipolarCurvesConfig.getMatchedPointsMap();

    if (epipolarCurvesConfig.getIsAutoPointMatching()) {
      FAIL_RETURN(extractKeyPoints(matchedPointsMap, pointsMap, inputImagesMap[frameNumber], inputCenters, transforms,
                                   pano, epipolarCurvesConfig.getDecimationCellFactor()));
    }

    /* Draw epipolar curves for each point in map */
    drawEpipolarCurves(inputImagesMap[frameNumber], pointsMap, inputCenters, transforms, pano);

    /* Draw depth for each point in map, if we have translations between cameras */
    if (pano->hasTranslations()) {
      computeAndDrawDepths(inputImagesMap[frameNumber], matchedPointsMap, inputCenters, transforms, pano);
    }

    /* Draw spherical grid */
    if (epipolarCurvesConfig.getSphericalGridRadius() > 0) {
      computeAndDrawSphericalGrid(inputImagesMap[frameNumber],
                                  static_cast<float>(epipolarCurvesConfig.getSphericalGridRadius()), inputCenters,
                                  transforms, pano);
    }

    /* Save input images */
    for (videoreaderid_t camid = 0; camid < (videoreaderid_t)inputImagesMap[frameNumber].size(); camid++) {
      FAIL_RETURN(dumpImageFile(
          inputImagesMap[frameNumber][camid],
          pano->getVideoInput(camid).getReaderConfig().asString() + "_frame_" + std::to_string(frameNumber) + ".png"))
    }
  }

  /*Create the result*/
  Potential<Ptv::Value> ret(Ptv::Value::emptyObject());

  ret->push("minStitchingDistance", new Parse::JsonValue(minStitchingDistance));

  return ret;
}

}  // namespace EpipolarCurves
}  // namespace VideoStitch
