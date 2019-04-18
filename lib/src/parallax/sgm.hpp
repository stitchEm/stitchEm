// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <core/rect.hpp>

#include <opencv2/core.hpp>

namespace VideoStitch {
namespace Core {
namespace SGM {

enum class SGMmode { SGM_5DIRS, SGM_8DIRS };

enum { DISP_SHIFT = 4, DISP_SCALE = (1 << DISP_SHIFT) };

typedef uint16_t CostType;
typedef int16_t DispType;

const CostType COST_VOLUME_INIT_VALUE = 1024;

enum { NR = 16, NR2 = NR / 2 };

/**
 * @brief Aggregates disparity cost volume according to SGM in outputRect and write it to disp1 buffer
 *        Originally comes from StereoSGBM::computeDisparitySGBM() in OpenCV 3.2.0
 *        Compared to original compputeDisparitySGBM, second disparity disp2 buffer was removed,
 *        and no left-right consistency check is applied
 *
 *        The disparity in disp1 is written with sub-pixel accuracy (4 fractional bits, see DISP_SCALE)
 *
 * @param costVolume width * height * (numDisparities) cost volume of type CostType.
 *                   Unfilled values in volume must be initialized with COST_VOLUME_INIT_VALUE (not zero, not
 * std::numeric_limits<CostType>::max())
 * @param rect ROI where to aggregate costs
 * @param disp1 OpenCV mat for returned disparity (must be allocated, width * height of type DispType)
 * @param buffer Temporary buffer, allocated within function, can be reused if costVolume dims do not change
 * @param minDisparity minimum disparity value
 * @param numDisparities number of disparities (3rd dimension of the costVolume)
 * @param P1 P1 SGM penalty
 * @param P2 P2 SGM penalty
 * @param uniquenessRatio ratio to reject ambiguities (such disparities are returned as (minDisparity-1) * DISP_SCALE
 * @param mode either SGM_5DIRS for 5 directions (paths coming from left, top-left, top, top-right, right pixels)
 *        or SGM_8DIRS for 8 directions (paths coming from left, top-lect, top, top-right, right, bottom-right, bottom,
 * bottom-left)
 *
 * @note  If numDisparities is a multiple of 16, and costVolume.date() is aligned on 16,
 *        SIMD implementation is used (about 3* faster)
 *        for 5 directions, memory consumption for temporary buffer is about width *          numDisparities *
 * sizeof(CostType) for 8 directions, memory consumption for temporary buffer is about width * height * numDisparities *
 * sizeof(CostType)
 */
void aggregateDisparityVolumeSGM(const cv::Mat& costVolume, const VideoStitch::Core::Rect& rect, cv::Mat& disp1,
                                 cv::Mat& buffer, int minDisparity, int numDisparities, int P1, int P2,
                                 int uniquenessRatio, const SGMmode mode);

/**
 * @brief Aggregates disparity cost volume according to SGM in outputRect and write it to disp1 buffer
 *        Originally comes from StereoSGBM::computeDisparitySGBM() in OpenCV 3.2.0
 *        Compared to original compputeDisparitySGBM, second disparity disp2 buffer was removed,
 *        and no left-right consistency check is applied
 *
 *        The disparity in disp1 is written with sub-pixel accuracy (4 fractional bits, see DISP_SCALE)
 *
 *        P2 penalty is adapted from the saliency image (which can be grayscale or color) through the formula
 *        P2 = max(P2Min, -P2Alpha * saliencyPathGradient + P2Gamma) like in rSGM
 *
 * @param saliency Saliency picture to guide P2 adaptation.
 * @param costVolume width * height * (numDisparities) cost volume of type CostType.
 *                   Unfilled values in volume must be initialized with COST_VOLUME_INIT_VALUE (not zero, not
 * std::numeric_limits<CostType>::max())
 * @param rect ROI where to aggregate costs
 * @param disparity OpenCV mat for returned disparity (must be allocated, width * height of type DispType, single
 * channel)
 * @param buffer Temporary buffer, allocated within function, can be reused if costVolume dims do not change
 * @param minDisparity minimum disparity value
 * @param numDisparities number of disparities (3rd dimension of the costVolume)
 * @param P1 P1 SGM penalty
 * @param P2Alpha P2 penalty alpha term
 * @param P2Gamma P2 penalty gamma term
 * @param P2Min minimum P2 penalty
 * @param uniquenessRatio ratio to reject ambiguities (such disparities are returned as (minDisparity-1) * DISP_SCALE
 * @param subPixelRefinement boolean to activate sub-pixel disparity refinement
 *        if true, the output disparities must be divided by float(DISP_SCALE) to get the floating point disparity
 *        if false, the output disparities have no scaling factor
 * @param mode either SGM_5DIRS for 5 directions (paths coming from left, top-left, top, top-right, right pixels)
 *        or SGM_8DIRS for 8 directions (paths coming from left, top-lect, top, top-right, right, bottom-right, bottom,
 * bottom-left)
 *
 * @note  If numDisparities is a multiple of 16, and costVolume.date() is aligned on 16,
 *        SIMD implementation is used (about 3* faster)
 *        for 5 directions, memory consumption for temporary buffer is about width *          numDisparities *
 * sizeof(CostType) for 8 directions, memory consumption for temporary buffer is about width * height * numDisparities *
 * sizeof(CostType)
 */
template <class saliencyType>
void aggregateDisparityVolumeWithAdaptiveP2SGM(const cv::Mat& saliency, const cv::Mat& costVolume,
                                               const VideoStitch::Core::Rect& rect, cv::Mat& disparity, cv::Mat& buffer,
                                               int minDisparity, int numDisparities, int P1, float P2Alpha, int P2Gamma,
                                               int P2Min, int uniquenessRatio, bool subPixelRefinement,
                                               const SGMmode mode);

}  // namespace SGM
}  // namespace Core
}  // namespace VideoStitch
