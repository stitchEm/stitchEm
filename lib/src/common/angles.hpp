// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef ANGLES_HPP_
#define ANGLES_HPP_

#include <math.h>

namespace VideoStitch {

/**
 * Converts an angle in degrees to radians.
 * @param deg Angle in degrees
 * @return Angle in radians.
 */
inline double degToRad(double v) { return M_PI * (v / 180.0); }

/**
 * Converts an angle in radians to degrees.
 * @param v Angle in radians
 * @return Angle in degrees.
 */
inline double radToDeg(double v) { return v * (180.0 / M_PI); }

}  // namespace VideoStitch

#endif  // QUATERNION_HPP_
