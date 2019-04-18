// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "cvImage.hpp"
#include "keypointExtractor.hpp"
#include "camera.hpp"

#include "libvideostitch/panoDef.hpp"

namespace VideoStitch {
namespace Calibration {

void drawMatches(const RigCvImages& rigInputImages, int idinput, videoreaderid_t idcam1, videoreaderid_t idcam2,
                 const KPList& kplist1, const KPList& kplist2, const Core::ControlPointList& input,
                 const int step /* used for the ordering of output files */, const std::string& description);

void drawReprojectionErrors(const RigCvImages& rigInputImages, int idinput, videoreaderid_t idcam1,
                            videoreaderid_t idcam2, const KPList& kplist1, const KPList& kplist2,
                            const Core::ControlPointList& input, const int step, const std::string& description);

void reportProjectionStats(Core::ControlPointList& list, videoreaderid_t idcam1, videoreaderid_t idcam2,
                           const std::string& description);

void reportControlPointsStats(const Core::ControlPointList& list);

double computeReprojectionErrors(const Core::ControlPointList& list);

void decimateSortedControlPoints(Core::ControlPointList& decimatedList, const Core::ControlPointList& sortedList,
                                 const int64_t inputWidth, const int64_t inputHeight, const double cellFactor);

/**
 @brief Get the mean reprojection distance of the provided control points
 @note projectFromCurrentSettings() or projectFromEstimatedRotation() must have been called to compute the reprojections
 of the control points
 */
double getMeanReprojectionDistance(const Core::ControlPointList& list);

/*
 @brief Fill PanoDefintion inputs from camera geometries
 */
Status fillPano(Core::PanoDefinition& pano, const std::vector<std::shared_ptr<Camera> >& cameras);

}  // namespace Calibration
}  // namespace VideoStitch
