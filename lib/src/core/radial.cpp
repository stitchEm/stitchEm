// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "radial.hpp"

#include "common/angles.hpp"

#include "libvideostitch/geometryDef.hpp"

#include <cmath>
#include <cassert>

#define LARGE_ROOT 1000.0

namespace VideoStitch {
namespace Core {
/**
 * See the following for some details:
 * http://www.panotools.org/dersch/barrel/barrel.html
 */

namespace {
/**
 * Workaround stupid pow spec to compute cubic root.
 * http://www.cplusplus.com/reference/clibrary/cmath/pow/
 */
double cubeRoot(double v) {
  if (v > 0) {
    return pow(v, 1 / 3.0);
  } else if (v < 0) {
    return -pow(-v, 1 / 3.0);
  } else {
    return 0;
  }
}

/**
 * d is the constant, ..., a the coefficient of the 3rd degree monomial.
 */
double smallestPositiveRootDegree3(double d, double c, double b, double a) {
  const double p = ((-1.0 / 3.0) * (b / a) * (b / a) + c / a) / 3.0;
  const double q = ((2.0 / 27.0) * (b / a) * (b / a) * (b / a) - (1.0 / 3.0) * (b / a) * (c / a) + d / a) / 2.0;

  if (q * q + p * p * p >= 0.0) {
    const double s = sqrt(q * q + p * p * p);
    const double p1 = -q + s;
    const double p2 = -q - s;
    const double root = cubeRoot(p1) + cubeRoot(p2) - b / (3.0 * a);
    assert(a * root * root * root + b * root * root + c * root + d < 0.001);
    return root > 0 ? root : LARGE_ROOT;
  } else {
    const double phi = acos(-q / sqrt(-p * p * p));
    double smallest = LARGE_ROOT;
    double root = 2.0 * sqrt(-p) * cos(phi / 3.0) - b / (3.0 * a);
    if (0.0 < root && root < smallest) {
      smallest = root;
    }
    root = -2.0 * sqrt(-p) * cos(phi / 3.0 + M_PI / 3.0) - b / (3.0 * a);
    if (0.0 < root && root < smallest) {
      smallest = root;
    }
    root = -2.0 * sqrt(-p) * cos(phi / 3.0 - M_PI / 3.0) - b / (3.0 * a);
    if (0.0 < root && root < smallest) {
      smallest = root;
    }
    return smallest;
  }
}

double smallestPositiveRootDegree2(double c, double b, double a) {
  // delta = b^2 - 4 a c > 0 means we have solutions
  const double delta = b * b - 4.0 * a * c;
  if (delta < 0) {
    return LARGE_ROOT;
  } else {
    const double root1 = (-b + sqrt(delta)) / (2.0 * a);
    const double root2 = -(b + sqrt(delta)) / (2.0 * a);
    if (root1 < root2) {
      if (root1 > 0) {
        return root1;
      } else if (root2 > 0) {
        return root2;
      } else {
        return LARGE_ROOT;
      }
    } else {
      if (root2 > 0) {
        return root2;
      } else if (root1 > 0) {
        return root1;
      } else {
        return LARGE_ROOT;
      }
    }
  }
}
}  // namespace

double computeRadial4(double radial0, double radial1, double radial2, double radial3) {
  /*
  We want to make sure that the pixel distance to the center lies into [0; smallest root].
  The distortion is then monotonic
  */
  if (radial3 != 0.0) {
    return smallestPositiveRootDegree3(radial0, radial1 * 2, radial2 * 3, radial3 * 4);
  } else if (radial2 != 0.0) {
    return smallestPositiveRootDegree2(radial0, radial1 * 2, radial2 * 3);
  } else if (radial1 != 0.0) {
    const double root = -radial0 / (radial1 * 2);
    return root > 0 ? root : LARGE_ROOT;
  } else {
    return LARGE_ROOT;
  }
}

void computeRadialParams(const InputDefinition& im, const GeometryDefinition& geometry, float& radial0, float& radial1,
                         float& radial2, float& radial3, float& radial4) {
  double denominator;
  /* meter distortion is unscaled */
  if (im.getUseMeterDistortion()) {
    denominator = 1.0f;
  } else {
    denominator =
        im.hasCroppedArea()
            ? ((double)(im.getCroppedWidth() < im.getCroppedHeight() ? im.getCroppedWidth() : im.getCroppedHeight()) /
               2.0f)
            : ((double)(im.getWidth() < im.getHeight() ? im.getWidth() : im.getHeight()) / 2.0f);
  }

  radial3 = (float)(geometry.getDistortA() / pow(denominator, 3.0));
  radial2 = (float)(geometry.getDistortB() / (denominator * denominator));
  radial1 = (float)(geometry.getDistortC() / denominator);
  radial0 = 1.0f - (float)(geometry.getDistortC() + geometry.getDistortB() + geometry.getDistortA());

  /* calculate the correction radius. */
  radial4 = (float)(computeRadial4(radial0, geometry.getDistortC(), geometry.getDistortB(), geometry.getDistortA()) *
                    denominator);
}

}  // namespace Core
}  // namespace VideoStitch
