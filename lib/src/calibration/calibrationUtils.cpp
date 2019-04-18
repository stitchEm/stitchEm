// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "calibrationUtils.hpp"

#include <util/pngutil.hpp>
#include <opencv2/imgproc.hpp>

namespace VideoStitch {
namespace Calibration {

void drawMatches(const RigCvImages& rigInputImages, int idinput, videoreaderid_t idcam1, videoreaderid_t idcam2,
                 const KPList& kplist1, const KPList& kplist2, const Core::ControlPointList& input,
                 const int step /* used for the ordering of output files */, const std::string& description) {
  if (idinput < 0) {
    // call this function for all rig pictures
    for (int i = 0; i < (int)rigInputImages[idcam1].size(); i++) {
      drawMatches(rigInputImages, i, idcam1, idcam2, kplist1, kplist2, input, step, description);
    }
  } else {
    cv::Mat outimage;

    // set up match vector
    std::vector<cv::DMatch> matches1to2;
    std::vector<cv::KeyPoint> keypoints0, keypoints1;
    unsigned int counter = 0;
    for (auto& it : input) {
      matches1to2.push_back(cv::DMatch(counter, counter, (float)it.score));
      keypoints0.push_back(cv::KeyPoint(float(it.x0), float(it.y0), 0.f));
      keypoints1.push_back(cv::KeyPoint(float(it.x1), float(it.y1), 0.f));
      ++counter;
    }

    // draw the matches
    cv::drawMatches(cv::Mat(*rigInputImages[idcam1][idinput].get()), keypoints0,
                    cv::Mat(*rigInputImages[idcam2][idinput].get()), keypoints1, matches1to2, outimage);
    // overlay all keypoints
    cv::drawMatches(cv::Mat(*rigInputImages[idcam1][idinput].get()), kplist1,
                    cv::Mat(*rigInputImages[idcam2][idinput].get()), kplist2, std::vector<cv::DMatch>(), outimage,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

    std::ostringstream imagefilename;
    imagefilename << "rig" << idinput << "matches" << idcam1 << "to" << idcam2 << "step" << step << description
                  << ".png";
    Util::PngReader writer;
    writer.writeBGRToFile(imagefilename.str().c_str(), outimage.cols, outimage.rows, outimage.data);
    Logger::get(Logger::Info) << "Writing " << imagefilename.str() << std::endl;
  }
}

void drawReprojectionErrors(const RigCvImages& rigInputImages, int idinput, videoreaderid_t idcam1,
                            videoreaderid_t idcam2, const KPList& kplist1, const KPList& kplist2,
                            const Core::ControlPointList& input, const int step, const std::string& description) {
  if (idinput < 0) {
    // call this function for all rig pictures
    for (size_t i = 0; i < rigInputImages[idcam1].size(); i++) {
      drawReprojectionErrors(rigInputImages, int(i), idcam1, idcam2, kplist1, kplist2, input, step, description);
    }
  } else {
    cv::Mat outimage;

    // draw all keypoints
    cv::drawMatches(cv::Mat(*rigInputImages[idcam1][idinput].get()), kplist1,
                    cv::Mat(*rigInputImages[idcam2][idinput].get()), kplist2, std::vector<cv::DMatch>(), outimage,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>());
    // overlay the reprojections
    int cam1width = rigInputImages[idcam1][idinput]->cols;
    for (auto& it : input) {
      cv::Point p0(cvRound(it.x0), cvRound(it.y0 + .5));
      cv::Point p1(cvRound(cam1width + it.x1 + .5), cvRound(it.y1 + .5));
      cv::Point rp0(cvRound(cam1width + it.rx0 + .5), cvRound(it.ry0 + .5));
      cv::Point rp1(cvRound(it.rx1 + .5), cvRound(it.ry1 + .5));
      // use a different color for synthetic control points
      const cv::Scalar color0 = (it.artificial) ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 0, 255);
      const cv::Scalar color1 = (it.artificial) ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 255, 0);

      cv::circle(outimage, p0, 3, color0);
      cv::circle(outimage, p1, 3, color1);
      cv::line(outimage, p0, rp1, color0);
      cv::line(outimage, p1, rp0, color1);
    }

    std::ostringstream imagefilename;
    imagefilename << "rig" << idinput << "matches" << idcam1 << "to" << idcam2 << "step" << step << description
                  << ".png";
    Util::PngReader writer;
    writer.writeBGRToFile(imagefilename.str().c_str(), outimage.cols, outimage.rows, outimage.data);
    Logger::get(Logger::Info) << "Writing " << imagefilename.str() << std::endl;
  }
}

void reportProjectionStats(Core::ControlPointList& list, videoreaderid_t idcam1, videoreaderid_t idcam2,
                           const std::string& description) {
  double mean0 = 0, meansq0 = 0, mean1 = 0, meansq1 = 0;
  size_t nkeypoints = list.size();
  std::vector<double> distances0, distances1;

  for (auto& cp : list) {
    double distance0, distance1;
    double distancesq0, distancesq1;

    distancesq0 = (cp.x0 - cp.rx1) * (cp.x0 - cp.rx1) + (cp.y0 - cp.ry1) * (cp.y0 - cp.ry1);
    distancesq1 = (cp.x1 - cp.rx0) * (cp.x1 - cp.rx0) + (cp.y1 - cp.ry0) * (cp.y1 - cp.ry0);

    distance0 = std::sqrt(distancesq0);
    distance1 = std::sqrt(distancesq1);

    mean0 += distance0;
    meansq0 += distancesq0;

    mean1 += distance1;
    meansq1 += distancesq1;

    distances0.push_back(distance0);
    distances1.push_back(distance1);
  }

  if (nkeypoints) {
    mean0 /= double(nkeypoints);
    mean1 /= double(nkeypoints);
    meansq0 /= double(nkeypoints);
    meansq1 /= double(nkeypoints);

    // get median values
    std::nth_element(distances0.begin(), distances0.begin() + nkeypoints / 2, distances0.end());
    std::nth_element(distances1.begin(), distances1.begin() + nkeypoints / 2, distances1.end());

    Logger::get(Logger::Verbose) << description << ", camera " << idcam1 << " to " << idcam2 << ", " << nkeypoints
                                 << " points: " << mean1 << " (+/-"
                                 << ((nkeypoints > 1) ? std::sqrt((meansq1 - mean1 * mean1)) : 0.) << "), median "
                                 << *(distances0.begin() + nkeypoints / 2) << std::endl;
    Logger::get(Logger::Verbose) << description << ", camera " << idcam2 << " to " << idcam1 << ", " << nkeypoints
                                 << " points: " << mean0 << " (+/-"
                                 << ((nkeypoints > 1) ? std::sqrt((meansq0 - mean0 * mean0)) : 0.) << "), median "
                                 << *(distances1.begin() + nkeypoints / 2) << std::endl;
  }
}

void reportControlPointsStats(const Core::ControlPointList& list) {
  // find the number of pairs per frame, the number of points per input and the number of matched pairs per input pair
  std::map<videoreaderid_t, int> matched_per_frame;
  std::map<videoreaderid_t, int> matched_per_input;
  std::map<videoreaderid_t, std::map<videoreaderid_t, int>> matched_to_per_input;
  std::map<std::pair<videoreaderid_t, videoreaderid_t>, int> matched_per_pairs;
  for (auto& cp : list) {
    matched_per_frame[cp.frameNumber]++;
    matched_per_input[cp.index0]++;
    matched_per_input[cp.index1]++;
    matched_to_per_input[cp.index0][cp.index1]++;
    matched_to_per_input[cp.index1][cp.index0]++;
    matched_per_pairs[{cp.index0, cp.index1}]++;
  }

  Logger::get(Logger::Verbose) << "Calibration control points statistics:" << std::endl;

  // report the number of pairs per frame
  Logger::get(Logger::Verbose) << "Number of matched pairs per frame" << std::endl;
  for (auto& it : matched_per_frame) {
    Logger::get(Logger::Verbose) << "  frame_number " << it.first << ": " << it.second << std::endl;
  }
  // report the number of control points and connections to other inputs per input
  Logger::get(Logger::Verbose) << "Number of control points to other inputs" << std::endl;
  for (auto& it : matched_per_input) {
    Logger::get(Logger::Verbose) << "  input_index " << it.first << ": " << it.second << std::endl;
    for (auto& itto : matched_to_per_input[it.first]) {
      Logger::get(Logger::Verbose) << "    matched_to_input_index " << itto.first << ": " << itto.second << std::endl;
    }
  }

  // report the number of matched pairs per input pair
  Logger::get(Logger::Verbose) << "Number of matched pairs per input pair" << std::endl;
  for (auto& it : matched_per_pairs) {
    Logger::get(Logger::Verbose) << "  input pair (" << it.first.first << ',' << it.first.second << "): " << it.second
                                 << std::endl;
  }
}

void decimateSortedControlPoints(Core::ControlPointList& decimatedList, const Core::ControlPointList& sortedList,
                                 const int64_t inputWidth, const int64_t inputHeight, const double cellFactor) {
  double w = (double)inputWidth;
  double h = (double)inputHeight;
  double cellsize = cellFactor * std::sqrt(w * w + h * h);
  int gwidth = (int)std::ceil(w / cellsize);
  int gheight = (int)std::ceil(h / cellsize);
  std::vector<bool> occupancy(gwidth * gheight, false);

  decimatedList.clear();

  int indexCurrentPoint = 0;
  for (const auto& it : sortedList) {
    Logger::get(Logger::Debug) << "  currentPoint(" << indexCurrentPoint++ << "): ";
    Logger::get(Logger::Debug) << "     " << it.x0 << "," << it.y0 << "  " << it.x1 << "," << it.y1
                               << "   score: " << it.score << "   error: " << it.error << "   frame: " << it.frameNumber
                               << "   indexes: " << it.index0 << "," << it.index1 << std::endl;
    double x = it.x0 / cellsize;
    double y = it.y0 / cellsize;
    int ix = (int)x;
    int iy = (int)y;
    Logger::get(Logger::Debug) << "        x,y: " << x << "," << y << "   ix,iy: " << ix << "," << iy << std::endl;
    if (occupancy[iy * gwidth + ix] == false) {
      Logger::get(Logger::Debug) << "         Point added" << std::endl;
      occupancy[iy * gwidth + ix] = true;
      decimatedList.push_back(it);
    } else {
      Logger::get(Logger::Debug) << "         Point NOT added" << std::endl;
    }
  }

  Logger::get(Logger::Debug) << "Decimated from " << sortedList.size() << " to " << decimatedList.size() << std::endl;
}

double getMeanReprojectionDistance(const Core::ControlPointList& list) {
  double sum = 0.;

  for (const auto& it : list) {
    sum += std::sqrt((it.x0 - it.rx1) * (it.x0 - it.rx1) + (it.y0 - it.ry1) * (it.y0 - it.ry1)) +
           std::sqrt((it.x1 - it.rx0) * (it.x1 - it.rx0) + (it.y1 - it.ry0) * (it.y1 - it.ry0));
  }

  if (list.size()) {
    sum /= (2 * list.size());
  }

  return sum;
}

Status fillPano(Core::PanoDefinition& pano, const std::vector<std::shared_ptr<Camera>>& cameras) {
  if (pano.numVideoInputs() != (int)cameras.size()) {
    return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration,
            "Cannot fill in PanoDefinition, inconsistent number of video inputs and cameras"};
  }

  auto videoInputs = pano.getVideoInputs();

  for (size_t cameraid = 0; cameraid < cameras.size(); ++cameraid) {
    Core::InputDefinition& idef = videoInputs[cameraid];

    if (idef.getWidth() != (int64_t)cameras[cameraid]->getWidth() ||
        idef.getHeight() != (int64_t)cameras[cameraid]->getHeight()) {
      return {Origin::CalibrationAlgorithm, ErrType::InvalidConfiguration,
              "Cannot fill in PanoDefinition, at least one input and camera have different sizes"};
    }

    idef.setUseMeterDistortion(true);
    idef.setFormat(cameras[cameraid]->getFormat());

    size_t width, height;

    // handle the cropped area to fill in the geometry, for PTGui compatibility reasons
    if (idef.hasCroppedArea()) {
      width = idef.getCroppedWidth() + 2 * idef.getCropLeft();
      height = idef.getCroppedHeight() + 2 * idef.getCropTop();
    } else {
      width = idef.getWidth();
      height = idef.getHeight();
    }

    Core::GeometryDefinition g = idef.getGeometries().getConstantValue();
    cameras[cameraid]->fillGeometry(g, (int)width, (int)height);
    Core::GeometryDefinitionCurve* gc = new Core::GeometryDefinitionCurve(g);
    idef.replaceGeometries(gc);
  }

  return Status::OK();
}

}  // namespace Calibration
}  // namespace VideoStitch
