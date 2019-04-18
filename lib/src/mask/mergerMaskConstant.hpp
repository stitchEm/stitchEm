// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

namespace VideoStitch {
namespace MergerMask {

#define MAX_COST 200.0f
#define MIN_PENALTY_COST 0.0f
#define PENALTY_COST 5.0f  // Penalize going around the current border, no clear info could be found
#define INVALID_COST 100000
#define SEAM_DIRECTION 4
#define INVALID_VALUE Image::RGBA::pack(1, 2, 3, 0)

#ifndef VS_OPENCL
#define CONSTANT_EXP __constant__
#else
#define CONSTANT_EXP
#endif

static const CONSTANT_EXP int seam_dir_advance[SEAM_DIRECTION] = {1, 1, 0, 0};
static const CONSTANT_EXP int perpendicular_dirs[SEAM_DIRECTION] = {1, 0, 0, 1};
static const CONSTANT_EXP int seam_dir_rows[2 * SEAM_DIRECTION] = {-1, 0, 0, 1, -1, 1, 1, -1};
static const CONSTANT_EXP int seam_dir_cols[2 * SEAM_DIRECTION] = {0, -1, 1, 0, -1, -1, 1, 1};

}  // namespace MergerMask
}  // namespace VideoStitch
