// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Util tests

#include "gpu/testing.hpp"

#include "libvideostitch/ptv.hpp"
#include <common/thread.hpp>
#include <util/lmfit/lmmin.hpp>
#include <util/pngutil.hpp>
#include <util/ransac.hpp>

#include <sstream>

#define IMAGE_WIDTH 123
#define IMAGE_HEIGHT 87

namespace VideoStitch {

namespace Testing {
void testPngMaskRoundtrip() {
  void* data = malloc(IMAGE_WIDTH * IMAGE_HEIGHT);
  for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
    ((unsigned char*)data)[i] = (i % 21) > 10 ? 1 : 0;
  }
  Util::PngReader png;
  std::string pngData;
  ENSURE(png.writeMaskToMemory(pngData, IMAGE_WIDTH, IMAGE_HEIGHT, data));
  void* data2;
  int64_t width, height;
  ENSURE(png.readMaskFromMemory((const unsigned char*)pngData.data(), pngData.size(), width, height, &data2));
  ENSURE_EQ((int)width, IMAGE_WIDTH);
  ENSURE_EQ((int)height, IMAGE_HEIGHT);
  ENSURE_ARRAY_EQ((const unsigned char*)data, (const unsigned char*)data2, IMAGE_WIDTH * IMAGE_HEIGHT);
  free(data);
  free(data2);
}

/**
 * A problem to fit a plane to a bunch of points.
 * params are { b, c, d }. x + 2.0 * y + 3.0 * z - 5.0 * d = 0
 */
class SimpleProblem : public Util::SolverProblem {
 public:
  struct Point {
    Point(double x, double y, double z) : x(x), y(y), z(z) {}
    double x, y, z;
  };

  static double rand01() { return rand() / (double)RAND_MAX; }

  SimpleProblem(int numPoints, double noise) {
    for (int i = 0; i < numPoints; ++i) {
      const double x = rand01();
      const double y = rand01();
      addPoint(x + noise * rand01(), y + noise * rand01(), computeZ(2.0, 3.0, -5.0, x, y) + noise * rand01());
    }
  }

  void addPoint(double x, double y, double z) { points.push_back(Point(x, y, z)); }

  double computeZ(double b, double c, double d, double x, double y) const { return -(x + b * y + d) / c; }

  double eqn(double b, double c, double d, const Point& point) const { return point.x + b * point.y + c * point.z + d; }

  virtual void eval(const double* params, int /*m_dat*/, double* fvec, const char* fFilter, int /*iterationNumber*/,
                    bool* /*requestBreak*/) const {
    for (size_t i = 0; i < points.size(); ++i) {
      if (!fFilter || fFilter[i] > 0) {
        fvec[i] = eqn(params[0], params[1], params[2], points[i]);
      } else {
        fvec[i] = 0.0;
      }
    }
  }

  virtual int numParams() const { return 3; }

  virtual int getNumInputSamples() const { return (int)points.size(); }

  virtual int getNumValuesPerSample() const { return 1; }

  virtual int getNumAdditionalValues() const { return 0; }

 private:
  std::vector<Point> points;
};

/**
 * Simple ransac solver
 */
class SimpleRansacSolver : public Util::RansacSolver<Util::LmminSolver<SimpleProblem>> {
 public:
  SimpleRansacSolver(const SimpleProblem& problem, int numPoints)
      : Util::RansacSolver<Util::LmminSolver<SimpleProblem>>(problem, 4, 50, (2 * numPoints) / 3) {}

  virtual bool isConsensualSample(double* values) const { return fabs(*values) < 0.02; }
};

void testRansac() {
  const int numPoints = 300;
  SimpleProblem problem(numPoints, 0.01);

  // Run the problem as an lmmin problem.
  {
    Util::LmminSolver<SimpleProblem> solver(problem, NULL, false);
    std::vector<double> params(3);
    ENSURE(solver.run(params));
    ENSURE_APPROX_EQ(2.0, params[0], 0.05);
    ENSURE_APPROX_EQ(3.0, params[1], 0.05);
    ENSURE_APPROX_EQ(-5.0, params[2], 0.05);
  }

  // And add a few crazy outliers, lmmin won't work anymore; run it as a ransac problem.
  {
    problem.addPoint(1.0, 1.0, 1.0);
    problem.addPoint(0.0, 0.0, 0.0);
    problem.addPoint(1.0, 0.7, 0.5);
    SimpleRansacSolver solver(problem, numPoints);
    std::vector<double> params(3);
    ENSURE(solver.run(params));
    ENSURE_APPROX_EQ(2.0, params[0], 0.05);
    ENSURE_APPROX_EQ(3.0, params[1], 0.05);
    ENSURE_APPROX_EQ(-5.0, params[2], 0.05);
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testPngMaskRoundtrip();
  VideoStitch::Testing::testRansac();
  return 0;
}
