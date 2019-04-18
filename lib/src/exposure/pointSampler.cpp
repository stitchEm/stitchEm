// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "pointSampler.hpp"

#include <random>

#include "libvideostitch/logging.hpp"
#include "common/container.hpp"
#include "core/geoTransform.hpp"

namespace VideoStitch {
namespace Util {

/**
 * A disjoint set of elements.
 */
class DisjointSet {
 public:
  explicit DisjointSet(int size) : numComponents(size) {
    // We start with a set of singletons.
    for (int i = 0; i < size; ++i) {
      components.push_back(i);
    }
  }

  /**
   * Tells whether adding this edge connects two components.
   */
  bool connectsComponents(int a, int b) const { return components[a] != components[b]; }

  /**
   * Adds an edge between a and b.
   */
  void addEdge(int a, int b) {
    if (connectsComponents(a, b)) {
      int componentB = components[b];
      // Merge components a and b.
      for (size_t i = 0; i < components.size(); ++i) {
        if (components[i] == componentB) {
          components[i] = components[a];
        }
      }
      --numComponents;
    }
  }

  /**
   * Returns the number of connected components.
   */
  int getNumConnectedComponents() const { return numComponents; }

 private:
  // Maps each element to its connected component id.
  std::vector<int> components;
  int numComponents;
};

std::ostream& operator<<(std::ostream& os, const Point& point) {
  os << point.videoInputId() << ":(" << point.coords().x << ", " << point.coords().y << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const PointPair& pointPair) {
  os << "{ " << *pointPair.p_k << "   " << *pointPair.p_l << " }" << std::endl;
  return os;
}

bool PointSampler::isFullyMasked(const unsigned char* maskPixelData, const int inputWidth, const int inputHeight,
                                 const int p_x, const int p_y, const int neighbourhoodSize) {
  if (maskPixelData == NULL) {
    return false;
  }
  for (int y = std::max(p_y - neighbourhoodSize, 0); y <= std::min(p_y + neighbourhoodSize, inputHeight - 1); ++y) {
    for (int x = std::max(p_x - neighbourhoodSize, 0); x <= std::min(p_x + neighbourhoodSize, inputWidth - 1); ++x) {
      // Find at least one non-masked pixel.
      if (maskPixelData[y * inputWidth + x] == 0) {
        return false;
      }
    }
  }
  // Only masked.
  return true;
}

const std::vector<PointPair*>& PointSampler::getPointPairs() const { return pointPairs; }

// When sampling, we must make sure to have a single connected compunent.
// Else, it can become possible to optimize each groups of inputs individually and end up having them badly fit.
PointSampler::PointSampler(const Core::PanoDefinition& pano, int maxSampledPoints, int minPointsPerInput,
                           int neighbourhoodSize)
    : generator(0), minPointsInOneOutput(0), numFloatingInputs(pano.numVideoInputs() - 1) {
  // Initialize random number generators.
  std::uniform_int_distribution<int> randomVideoInput(0, (int)pano.numVideoInputs() - 1);
  auto randomX = std::vector<std::shared_ptr<std::uniform_real_distribution<float>>>(pano.numVideoInputs());
  auto randomY = std::vector<std::shared_ptr<std::uniform_real_distribution<float>>>(pano.numVideoInputs());
  std::vector<Core::TopLeftCoords2> centers(pano.numVideoInputs());
  transforms = std::vector<std::shared_ptr<Core::TransformStack::GeoTransform>>(pano.numVideoInputs());

  for (videoreaderid_t i = 0; i < pano.numVideoInputs(); ++i) {
    auto& inputI = pano.getVideoInput(i);

    randomX[i] = std::make_shared<std::uniform_real_distribution<float>>(0.0f, (float)inputI.getWidth() - 1.0f);
    randomY[i] = std::make_shared<std::uniform_real_distribution<float>>(0.0f, (float)inputI.getHeight() - 1.0f);
    transforms[i] =
        std::shared_ptr<Core::TransformStack::GeoTransform>(Core::TransformStack::GeoTransform::create(pano, inputI));

    centers[i] = Core::TopLeftCoords2(float(inputI.getCropLeft() + inputI.getCropRight()) / 2.0f,
                                      float(inputI.getCropTop() + inputI.getCropBottom()) / 2.0f);
  }

  // Draw points.
  std::vector<int> pointsPerInput(pano.numVideoInputs());
  minPointsInOneOutput = 0;
  DisjointSet disjointSet((int)pano.numVideoInputs());

  for (int iteration = 0; iteration < maxSampledPoints &&
                          (minPointsInOneOutput < minPointsPerInput || disjointSet.getNumConnectedComponents() > 1);
       ++iteration) {
    // Draw an input
    const int k = randomVideoInput(generator);
    // Draw a point for the input.
    auto& videoInputK = pano.getVideoInput(k);

    const Core::TopLeftCoords2 p_k((*randomX[k])(generator), (*randomY[k])(generator));
    if (!transforms[k]->isWithinInputBounds(videoInputK, p_k) ||
        isFullyMasked(videoInputK.getMaskPixelDataIfValid(), (int)videoInputK.getWidth(), (int)videoInputK.getHeight(),
                      (int)p_k.x, (int)p_k.y, neighbourhoodSize)) {
      continue;
    }

    const Core::SphericalCoords3 point3d =
        transforms[k]->mapInputToRigSpherical(pano.getVideoInput(k), Core::CenterCoords2(p_k, centers[k]), 0);

    if (point3d.x == INVALID_INVERSE_DISTORTION && point3d.y == INVALID_INVERSE_DISTORTION &&
        point3d.z == INVALID_INVERSE_DISTORTION) {
      continue;
    }

    for (int l = 0; l < (int)pano.numVideoInputs(); ++l) {
      if (l == k) {
        continue;
      }

      const Core::TopLeftCoords2 p_l(transforms[l]->mapRigSphericalToInput(pano.getVideoInput(l), point3d, 0),
                                     centers[l]);

      if (transforms[l]->isWithinInputBounds(pano.getVideoInput(l), p_l) &&
          !isFullyMasked(pano.getVideoInput(l).getMaskPixelDataIfValid(), (int)pano.getVideoInput(l).getWidth(),
                         (int)pano.getVideoInput(l).getHeight(), (int)p_l.x, (int)p_l.y, neighbourhoodSize)) {
        pointPairs.push_back(new PointPair(new Point(k, p_k), new Point(l, p_l), point3d));
        ++pointsPerInput[l];
        ++pointsPerInput[k];
        disjointSet.addEdge(k, l);
      }
    }

    minPointsInOneOutput = std::numeric_limits<int>::max();
    for (videoreaderid_t i = 0; i < pano.numVideoInputs(); ++i) {
      if (minPointsInOneOutput > pointsPerInput[i]) {
        minPointsInOneOutput = pointsPerInput[i];
      }
    }
  }

  for (videoreaderid_t i = 0; i < pano.numVideoInputs(); ++i) {
    Logger::get(Logger::Verbose) << "Video input " << i << " has " << pointsPerInput[i] << " points." << std::endl;
  }
  numConnectedComponents = int(disjointSet.getNumConnectedComponents());
}

PointSampler::~PointSampler() { deleteAll(pointPairs); }

const std::map<videoreaderid_t, std::map<int, std::vector<PointPair*>>>& RadialPointSampler::getPointPairsByRadius()
    const {
  return pointVectors;
}

double radiusForPoint(const Core::PanoDefinition& pano, Point* p) {
  const Core::InputDefinition& input = pano.getVideoInput(p->videoInputId());

  float width = (float)input.getWidth();
  float height = (float)input.getHeight();

  if (input.hasCroppedArea()) {
    width = (float)input.getCroppedWidth();
    height = (float)input.getCroppedHeight();
  }

  const Core::CenterCoords2 centerCoordsP =
      Core::CenterCoords2(p->coords(), Core::TopLeftCoords2(width / 2.0f, height / 2.0f));

  // inverseDemi...
  const double radiusSq = 4 *
                          ((double)(centerCoordsP.x * centerCoordsP.x) + (double)(centerCoordsP.y * centerCoordsP.y)) /
                          (double)(width * width + height * height);
  const double radius = sqrt(radiusSq);

  assert(radius < 1);

  return radius;
}

RadialPointSampler::RadialPointSampler(const Core::PanoDefinition& pano, int maxSampledPoints, int minPointsPerInput,
                                       int neighbourhoodSize, int numberOfRadialBins)
    : PointSampler(pano, maxSampledPoints, minPointsPerInput, neighbourhoodSize), pointVectors() {
  for (PointPair* pointPair : getPointPairs()) {
    auto addPoint = [&](Point* p) {
      int radiusIndex = (int)(radiusForPoint(pano, p) * numberOfRadialBins);

      std::map<int, std::vector<PointPair*>>& pointsByRadius = pointVectors[p->videoInputId()];
      std::vector<PointPair*>& pointVector = pointsByRadius[radiusIndex];
      pointVector.push_back(pointPair);
    };

    addPoint(pointPair->p_k);
    addPoint(pointPair->p_l);
  }
}

}  // namespace Util
}  // namespace VideoStitch
