// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <memory>

#include <common/angles.hpp>
#include "libvideostitch/curves.hpp"

namespace VideoStitch {

namespace Testing {
void testCurveExtend() {
  {
    Core::Spline* sourceSplines = Core::Spline::point(50, 7.0);
    sourceSplines->cubicTo(60, 6.0)->cubicTo(70, 5.0);
    std::unique_ptr<Core::Curve> source(new Core::Curve(sourceSplines));

    // S:                   x----x----x
    // this: x-----x----x
    //       x-----x----x---x----x----x
    {
      Core::Spline* curveSplines = Core::Spline::point(20, 1.0);
      curveSplines->cubicTo(30, 2.0)->cubicTo(40, 3.0);
      std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
      curve->extend(source.get());
      Core::Spline* expectedSplines = Core::Spline::point(20, 1.0);
      expectedSplines->cubicTo(30, 2.0)->cubicTo(40, 3.0)->cubicTo(50, 7.0)->cubicTo(60, 6.0)->cubicTo(70, 5.0);
      std::unique_ptr<Core::Curve> expected(new Core::Curve(expectedSplines));
      ENSURE(*expected == *curve);
    }

    // S:    x-----x----x
    // this:                x----x----x
    //       x-----x----x---x----x----x
    {
      Core::Spline* curveSplines = Core::Spline::point(80, 1.0);
      curveSplines->cubicTo(90, 2.0)->cubicTo(100, 3.0);
      std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
      curve->extend(source.get());
      Core::Spline* expectedSplines = Core::Spline::point(50, 7.0);
      expectedSplines->cubicTo(60, 6.0)->cubicTo(70, 5.0)->cubicTo(80, 1.0)->cubicTo(90, 2.0)->cubicTo(100, 3.0);
      std::unique_ptr<Core::Curve> expected(new Core::Curve(expectedSplines));
      ENSURE(*expected == *curve);
    }

    // This is a noop:
    // S:                   x----x----x
    // this: x-----x----x-----x------x----x
    {
      Core::Spline* curveSplines = Core::Spline::point(10, 1.0);
      curveSplines->cubicTo(20, 2.0)->cubicTo(30, 3.0)->cubicTo(60, 3.0)->cubicTo(80, 3.0);
      std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
      curve->extend(source.get());
      std::unique_ptr<Core::Curve> expected(curve->clone());
      ENSURE(*expected == *curve);
    }

    // S:             x----x----x
    // this: x----x-----x
    //       x----x-----x--x----x
    {
      Core::Spline* curveSplines = Core::Spline::point(20, 1.0);
      curveSplines->cubicTo(30, 2.0)->cubicTo(55, 3.0);
      std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
      curve->extend(source.get());
      Core::Spline* expectedSplines = Core::Spline::point(20, 1.0);
      expectedSplines->cubicTo(30, 2.0)->cubicTo(55, 3.0)->cubicTo(60, 6.0)->cubicTo(70, 5.0);
      std::unique_ptr<Core::Curve> expected(new Core::Curve(expectedSplines));
      ENSURE(*expected == *curve);
    }

    // S:    x----x-----x
    // this:          x----x----x
    //       x----x---x----x----x
    {
      Core::Spline* curveSplines = Core::Spline::point(65, 1.0);
      curveSplines->cubicTo(80, 2.0)->cubicTo(90, 3.0);
      std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
      curve->extend(source.get());
      Core::Spline* expectedSplines = Core::Spline::point(50, 7.0);
      expectedSplines->cubicTo(60, 6.0)->cubicTo(65, 1.0)->cubicTo(80, 2.0)->cubicTo(90, 3.0);
      std::unique_ptr<Core::Curve> expected(new Core::Curve(expectedSplines));
      ENSURE(*expected == *curve);
    }

    // S:    x----x-----x
    // this:    x----x
    //       x--x----x--x
    {
      Core::Spline* curveSplines = Core::Spline::point(55, 1.0);
      curveSplines->cubicTo(65, 2.0);
      std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
      curve->extend(source.get());
      Core::Spline* expectedSplines = Core::Spline::point(50, 7.0);
      expectedSplines->cubicTo(55, 1.0)->cubicTo(65, 2.0)->cubicTo(70, 5.0);
      std::unique_ptr<Core::Curve> expected(new Core::Curve(expectedSplines));
      ENSURE(*expected == *curve);
    }

    // S:    x----x-----x
    // this:    x
    //       x--x-x-----x
    {
      std::unique_ptr<Core::Curve> curve(new Core::Curve(Core::Spline::point(55, 1.0)));
      curve->extend(source.get());
      Core::Spline* expectedSplines = Core::Spline::point(50, 7.0);
      expectedSplines->cubicTo(55, 1.0)->cubicTo(60, 6.0)->cubicTo(70, 5.0);
      std::unique_ptr<Core::Curve> expected(new Core::Curve(expectedSplines));
      ENSURE(*expected == *curve);
    }

    // S:    x----x-----x
    // this: ------------  (const)
    //       x----x-----x
    {
      std::unique_ptr<Core::Curve> curve(new Core::Curve(12.0));
      curve->extend(source.get());
      ENSURE(*source == *curve, "extending a constant curve should yield the source");
    }

    // S:    ------------  (const)
    // this: x----x-----x
    //       x----x-----x
    {
      std::unique_ptr<Core::Curve> constantSource(new Core::Curve(12.0));
      std::unique_ptr<Core::Curve> curve(source->clone());
      curve->extend(constantSource.get());
      ENSURE(*source == *curve, "extending with a constant curve should be a noop");
    }
  }

  // Single-point-curve versions of the above.
  {
    std::unique_ptr<Core::Curve> source(new Core::Curve(Core::Spline::point(50, 7.0)));
    // S:    x
    // this:                x----x----x
    //       x--------------x----x----x
    {
      Core::Spline* curveSplines = Core::Spline::point(80, 1.0);
      curveSplines->cubicTo(90, 2.0)->cubicTo(100, 3.0);
      std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
      curve->extend(source.get());
      Core::Spline* expectedSplines = Core::Spline::point(50, 7.0);
      expectedSplines->cubicTo(80, 1.0)->cubicTo(90, 2.0)->cubicTo(100, 3.0);
      std::unique_ptr<Core::Curve> expected(new Core::Curve(expectedSplines));
      ENSURE(*expected == *curve);
    }

    // S:    x
    // this:                x
    //       x--------------x
    {
      Core::Spline* curveSplines = Core::Spline::point(80, 1.0);
      std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
      curve->extend(source.get());
      Core::Spline* expectedSplines = Core::Spline::point(50, 7.0);
      expectedSplines->cubicTo(80, 1.0);
      std::unique_ptr<Core::Curve> expected(new Core::Curve(expectedSplines));
      ENSURE(*expected == *curve);
    }

    // This is a noop:
    // S:    x
    // this: x
    {
      std::unique_ptr<Core::Curve> curve(new Core::Curve(Core::Spline::point(50, 2.0)));
      curve->extend(source.get());
      std::unique_ptr<Core::Curve> expected(curve->clone());
      ENSURE(*expected == *curve);
    }

    // S:             x
    // this: x----x-----x
    //       x----x-----x
    {
      Core::Spline* curveSplines = Core::Spline::point(35, 1.0);
      curveSplines->cubicTo(45, 2.0)->cubicTo(55, 3.0);
      std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
      curve->extend(source.get());
      std::unique_ptr<Core::Curve> expected(curve->clone());
      ENSURE(*expected == *curve);
    }
  }
}

void testLinear() {
  Core::Spline* curveSplines = Core::Spline::point(0, 0.0);
  curveSplines->lineTo(100, 100.0)->lineTo(200, 500.0);
  std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
  for (int i = -100; i <= 0; ++i) {
    ENSURE_APPROX_EQ(0.0, curve->at(i), 0.0001);
  }
  for (int i = 0; i <= 100; ++i) {
    ENSURE_APPROX_EQ((double)i, curve->at(i), 0.0001);
  }
  for (int i = 100; i <= 200; ++i) {
    ENSURE_APPROX_EQ(100.0 + (i - 100.0) * 4.0, curve->at(i), 0.0001);
  }
}

void testCubic() {
  Core::Spline* curveSplines = Core::Spline::point(0, 0.0);
  curveSplines->cubicTo(100, 100.0)->cubicTo(200, 500.0);

  std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));

  ENSURE_APPROX_EQ(0.0, curve->at(-100), 0.0001);
  ENSURE_APPROX_EQ(0.0, curve->at(-1), 0.0001);

  ENSURE_APPROX_EQ(0.0, curve->at(0), 0.0001);
  ENSURE_APPROX_EQ(100.0, curve->at(100), 0.0001);
  ENSURE_APPROX_EQ(500.0, curve->at(200), 0.0001);

  ENSURE_APPROX_EQ(500.0, curve->at(201), 0.0001);
  ENSURE_APPROX_EQ(500.0, curve->at(300), 0.0001);

  ENSURE_EQ(curve->at(0) - curve->at(-1), 0., "Slope should be 0 before curve starts");

  // approx value chosen after current implementation
  ENSURE_APPROX_EQ(curve->at(1) - curve->at(0), 0., 0.02, "Ease-in: slope should be close to zero");

  // approx value chosen after current implementation
  ENSURE_APPROX_EQ(curve->at(200) - curve->at(199), 0., 0.15, "Ease-out: slope should be close to zero");

  ENSURE_EQ(curve->at(200) - curve->at(201), 0., "Slope should be 0 after curve ends");
}

void testCubicGeometryDefinition() {
  Core::SplineTemplate<Core::GeometryDefinition>* curveSplines =
      Core::SplineTemplate<Core::GeometryDefinition>::point(0, Core::GeometryDefinition());

  // Basic checks to ensure that all members are interpolated
  Core::GeometryDefinition geom;
  geom.setCenterX(1.);
  geom.setCenterY(2.);
  geom.setDistortA(3.);
  geom.setDistortB(4.);
  geom.setDistortC(5.);
  geom.setDistortP1(6.);
  geom.setDistortP2(7.);
  geom.setDistortS1(8.);
  geom.setDistortS2(9.);
  geom.setDistortS3(10.);
  geom.setDistortS4(11.);
  geom.setDistortTau1(12.);
  geom.setDistortTau2(13.);
  geom.setHorizontalFocal(14.);
  geom.setVerticalFocal(15.);
  geom.setYaw(16.);
  geom.setPitch(17.);
  geom.setRoll(18.);
  geom.setTranslationX(19.);
  geom.setTranslationY(20.);
  geom.setTranslationZ(21.);

  curveSplines->cubicTo(100, geom);

  std::unique_ptr<Core::GeometryDefinitionCurve> curve(new Core::GeometryDefinitionCurve(curveSplines));

  Core::GeometryDefinition interpolated = curve->at(50);

  ENSURE_APPROX_EQ(0.5, interpolated.getCenterX(), 0.0001);
  ENSURE_APPROX_EQ(1.0, interpolated.getCenterY(), 0.0001);
  ENSURE_APPROX_EQ(1.5, interpolated.getDistortA(), 0.0001);
  ENSURE_APPROX_EQ(2.0, interpolated.getDistortB(), 0.0001);
  ENSURE_APPROX_EQ(2.5, interpolated.getDistortC(), 0.0001);
  ENSURE_APPROX_EQ(3.0, interpolated.getDistortP1(), 0.0001);
  ENSURE_APPROX_EQ(3.5, interpolated.getDistortP2(), 0.0001);
  ENSURE_APPROX_EQ(4.0, interpolated.getDistortS1(), 0.0001);
  ENSURE_APPROX_EQ(4.5, interpolated.getDistortS2(), 0.0001);
  ENSURE_APPROX_EQ(5.0, interpolated.getDistortS3(), 0.0001);
  ENSURE_APPROX_EQ(5.5, interpolated.getDistortS4(), 0.0001);
  ENSURE_APPROX_EQ(6.0, interpolated.getDistortTau1(), 0.0001);
  ENSURE_APPROX_EQ(6.5, interpolated.getDistortTau2(), 0.0001);
  ENSURE_APPROX_EQ(507.0, interpolated.getHorizontalFocal(),
                   0.0001);  // 507.0 because GeometryDefinition at 0 has a default focal of 1000.0
  ENSURE_APPROX_EQ(507.5, interpolated.getVerticalFocal(),
                   0.0001);  // 507.5 because GeometryDefinition at 0 has a default focal of 1000.0
  ENSURE_APPROX_EQ(8.0, interpolated.getYaw(), 0.0001);
  ENSURE_APPROX_EQ(8.5, interpolated.getPitch(), 0.0001);
  ENSURE_APPROX_EQ(9.0, interpolated.getRoll(), 0.0001);
  ENSURE_APPROX_EQ(9.5, interpolated.getTranslationX(), 0.0001);
  ENSURE_APPROX_EQ(10.0, interpolated.getTranslationY(), 0.0001);
  ENSURE_APPROX_EQ(10.5, interpolated.getTranslationZ(), 0.0001);
}

void testVSA1090() {
  Core::Spline* curveSplines = Core::Spline::point(50, 10.0);
  curveSplines->lineTo(100, 100.0);
  std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
  curve->splitAt(25);

  curveSplines = Core::Spline::point(25, 10.0);
  curveSplines->lineTo(50, 10.0)->lineTo(100, 100.0);
  std::unique_ptr<Core::Curve> expected(new Core::Curve(curveSplines));
  ENSURE(*expected == *curve);
}

void testVSA1181() {
  Core::Spline* curveSplines = Core::Spline::point(25, 10.0);
  curveSplines->lineTo(50, 10.0)->lineTo(100, 10.0);
  std::unique_ptr<Core::Curve> curve(new Core::Curve(curveSplines));
  curve->mergeAt(100);
  curve->splitAt(60);
  curve->mergeAt(101);
  curve->splitAt(102);

  curveSplines = Core::Spline::point(25, 10.0);
  curveSplines->lineTo(50, 10.0)->lineTo(60, 10.0)->lineTo(102, 10.0);
  std::unique_ptr<Core::Curve> expected(new Core::Curve(curveSplines));
  ENSURE(*expected == *curve);
}

bool equalToRadian(const double expected, const double actual, double epsilon) {
  return fabs(expected - actual) < epsilon || fabs(expected - actual + 2.0 * M_PI) < epsilon ||
         fabs(expected - actual - 2.0 * M_PI) < epsilon;
}

/**
 * Tests that euler <-> quaternion roundtrip is identity.
 * @param yaw in radians
 * @param pitch in radians
 * @param roll in radians
 */
void testQuaternionToEulerRoundtrip(const double yaw, const double pitch, const double roll) {
  const Quaternion<double> q = Quaternion<double>::fromEulerZXY(yaw, pitch, roll);
  double yaw2 = 0.0;
  double pitch2 = 0.0;
  double roll2 = 0.0;
  q.toEuler(yaw2, pitch2, roll2);
  if (!(equalToRadian(yaw, yaw2, 0.001) && equalToRadian(pitch, pitch2, 0.001) && equalToRadian(roll, roll2, 0.001))) {
    std::cerr << "TEST FAILED: Expected (" << yaw << ", " << pitch << ", " << roll << ") +- 2 PI', got (" << yaw2
              << ", " << pitch2 << ", " << roll2 << ")" << std::endl;
    std::raise(SIGABRT);
  }
}

/**
 * Simple test with rotation on only one axis (yaw), constant rate, symetric.
 * It is equivalent to a Slerp between q01 and q02
 */
void simpleCatmullRomTestYaw(double angle0, double angle1, double angle2, double angle3) {
  const int numSteps = 100;

  Quaternion<double> q00 = Quaternion<double>::fromEulerZXY(angle0, 0.0, 0.0);
  Quaternion<double> q01 = Quaternion<double>::fromEulerZXY(angle1, 0.0, 0.0);
  Quaternion<double> q02 = Quaternion<double>::fromEulerZXY(angle2, 0.0, 0.0);
  Quaternion<double> q03 = Quaternion<double>::fromEulerZXY(angle3, 0.0, 0.0);

  for (int i = 0; i <= numSteps; ++i) {
    double expectedAngle = angle1 + (double)i * (angle2 - angle1) / (double)numSteps;

    while (expectedAngle > M_PI) {
      expectedAngle -= 2 * M_PI;
    }
    while (expectedAngle < -M_PI) {
      expectedAngle += 2 * M_PI;
    }

    const double t = double(i) / numSteps;
    Quaternion<double> q2 = Quaternion<double>::catmullRom(q00, q01, q02, q03, t);
    double yaw, pitch, roll;
    q2.toEuler(yaw, pitch, roll);

    Quaternion<double> qSlerp = Quaternion<double>::slerp(q01, q02, t);
    double yawS, pitchS, rollS;
    qSlerp.toEuler(yawS, pitchS, rollS);

    ENSURE_APPROX_EQ(expectedAngle, yaw, 1e-6);
    ENSURE_APPROX_EQ(0.0, pitch, 1e-6);
    ENSURE_APPROX_EQ(0.0, roll, 1e-6);

    ENSURE_APPROX_EQ(expectedAngle, yawS, 1e-6);
    ENSURE_APPROX_EQ(0.0, pitchS, 1e-6);
    ENSURE_APPROX_EQ(0.0, rollS, 1e-6);
  }
}

/**
 * Simple test with rotation on only one axis (pitch), constant rate, symetric.
 * It is equivalent to a Slerp between q01 and q02
 */
void simpleCatmullRomTestPitch(double angle0, double angle1, double angle2, double angle3) {
  const int numSteps = 100;

  Quaternion<double> q00 = Quaternion<double>::fromEulerZXY(0.0, angle0, 0.0);
  Quaternion<double> q01 = Quaternion<double>::fromEulerZXY(0.0, angle1, 0.0);
  Quaternion<double> q02 = Quaternion<double>::fromEulerZXY(0.0, angle2, 0.0);
  Quaternion<double> q03 = Quaternion<double>::fromEulerZXY(0.0, angle3, 0.0);

  for (int i = 0; i <= numSteps; ++i) {
    double expectedAngle = angle1 + (double)i * (angle2 - angle1) / (double)numSteps;

    while (expectedAngle > M_PI / 2) {
      expectedAngle -= 2 * M_PI;
    }
    while (expectedAngle < -M_PI / 2) {
      expectedAngle += 2 * M_PI;
    }

    const double t = double(i) / numSteps;
    Quaternion<double> q2 = Quaternion<double>::catmullRom(q00, q01, q02, q03, t);
    double yaw, pitch, roll;
    q2.toEuler(yaw, pitch, roll);

    Quaternion<double> qSlerp = Quaternion<double>::slerp(q01, q02, t);
    double yawS, pitchS, rollS;
    qSlerp.toEuler(yawS, pitchS, rollS);

    ENSURE_APPROX_EQ(0.0, yaw, 1e-6);
    ENSURE_APPROX_EQ(expectedAngle, pitch, 1e-6);
    ENSURE_APPROX_EQ(0.0, roll, 1e-6);

    ENSURE_APPROX_EQ(0.0, yawS, 1e-6);
    ENSURE_APPROX_EQ(expectedAngle, pitchS, 1e-6);
    ENSURE_APPROX_EQ(0.0, rollS, 1e-6);
  }
}

/**
 * Simple test with rotation on only one axis (roll), constant rate, symetric.
 * It is equivalent to a Slerp between q01 and q02
 */
void simpleCatmullRomTestRoll(double angle0, double angle1, double angle2, double angle3) {
  const int numSteps = 100;

  Quaternion<double> q00 = Quaternion<double>::fromEulerZXY(0.0, 0.0, angle0);
  Quaternion<double> q01 = Quaternion<double>::fromEulerZXY(0.0, 0.0, angle1);
  Quaternion<double> q02 = Quaternion<double>::fromEulerZXY(0.0, 0.0, angle2);
  Quaternion<double> q03 = Quaternion<double>::fromEulerZXY(0.0, 0.0, angle3);

  for (int i = 0; i <= numSteps; ++i) {
    double expectedAngle = angle1 + (double)i * (angle2 - angle1) / (double)numSteps;

    while (expectedAngle > M_PI) {
      expectedAngle -= 2 * M_PI;
    }
    while (expectedAngle < -M_PI) {
      expectedAngle += 2 * M_PI;
    }

    const double t = double(i) / numSteps;
    Quaternion<double> q2 = Quaternion<double>::catmullRom(q00, q01, q02, q03, t);
    double yaw, pitch, roll;
    q2.toEuler(yaw, pitch, roll);

    Quaternion<double> qSlerp = Quaternion<double>::slerp(q01, q02, t);
    double yawS, pitchS, rollS;
    qSlerp.toEuler(yawS, pitchS, rollS);

    ENSURE_APPROX_EQ(0.0, yaw, 1e-6);
    ENSURE_APPROX_EQ(0.0, pitch, 1e-6);
    ENSURE_APPROX_EQ(expectedAngle, roll, 1e-6);

    ENSURE_APPROX_EQ(0.0, yawS, 1e-6);
    ENSURE_APPROX_EQ(0.0, pitchS, 1e-6);
    ENSURE_APPROX_EQ(expectedAngle, rollS, 1e-6);
  }
}

/*
 * Some wxmaxima code;
c(angle) := cos(angle/2);
s(angle) := sin(angle/2);
q0(y,p,r) := - c(r) * c(p) * c(y) - s(r) * s(p) * s(y);
q1(y,p,r) := c(r) * s(p) * c(y) + s(r) * c(p) * s(y);
q2(y,p,r) := c(r) * c(p) * s(y) - s(r) * s(p) * c(y);
q3(y,p,r) := - c(r) * s(p) * s(y) + s(r) * c(p) * c(y);
t1(y,p,r) := q1(y,p,r) * q3(y,p,r) - q0(y,p,r) * q2(y,p,r);
t2(y,p,r) := q3(y,p,r) * q3(y,p,r) - q2(y,p,r) * q2(y,p,r) - q1(y,p,r) * q1(y,p,r) + q0(y,p,r) * q0(y,p,r);
yout(y, p,r) := atan2(2 * t1(y,p,r), t2(y,p,r));
yout2(y, p,r) := atan(2 * t1(y,p,r) / t2(y,p,r));
trigsimp(2 * t1(y,p,r) / t2(y,p,r));
yout(0.21, 0.12, 0.42);

t3(y,p,r) := q1(y,p,r) * q2(y,p,r) - q0(y,p,r) * q3(y,p,r);
t4(y,p,r) := q2(y,p,r) * q2(y,p,r) - q3(y,p,r) * q3(y,p,r) - q1(y,p,r) * q1(y,p,r) + q0(y,p,r) * q0(y,p,r);
trigsimp(2 * t3(y,p,r) / t4(y,p,r));
rout(y, p,r) := atan2(2 * t3(y,p,r), t4(y,p,r));
q0(0, 1.5708, 0.0628319);
q1(0, 1.5708, 0.0628319);
q2(0, 1.5708, 0.0628319);
q3(0, 1.5708, 0.0628319);
t3(0, 1.5708, 0.0628319);
t4(0, 1.5708, 0.0628319);
2.0 * t3(0, 1.5708, 0.0628319) / t4(0, 1.5708, 0.0628319);
rout(0, 1.5708, 0.0628319) + 3.14159;
*/
void testQuaternion() {
  const int steps = 7;
  for (int i = 0; i < steps; ++i) {
    const double yaw = 2.0 * M_PI * (double)i / (double)steps;
    for (int j = 1; j < steps - 1; ++j) {  // Avoid being too close to gimbal lock.
      const double pitch = M_PI * (-0.5 + (double)j / (double)steps);
      for (int k = 0; k < steps; ++k) {
        const double roll = 2.0 * M_PI * (double)k / (double)steps;
        testQuaternionToEulerRoundtrip(yaw, pitch, roll);
      }
    }
  }

  // Very simple catmullRom test.
  double angle0 = 0 * M_PI / 6;
  double angle1 = 1 * M_PI / 6;
  double angle2 = 2 * M_PI / 6;
  double angle3 = 3 * M_PI / 6;
  simpleCatmullRomTestYaw(angle0, angle1, angle2, angle3);
  simpleCatmullRomTestRoll(angle0, angle1, angle2, angle3);

  angle0 = 0 * 2 * M_PI / 3;
  angle1 = 1 * 2 * M_PI / 3;
  angle2 = 2 * 2 * M_PI / 3;
  angle3 = 3 * 2 * M_PI / 3;
  simpleCatmullRomTestYaw(angle0, angle1, angle2, angle3);
  simpleCatmullRomTestRoll(angle0, angle1, angle2, angle3);

  angle0 = M_PI / 2;
  angle1 = M_PI / 4;
  angle2 = 0;
  angle3 = -M_PI / 4;
  simpleCatmullRomTestPitch(angle0, angle1, angle2, angle3);

  angle0 = 5 * M_PI / 11;
  angle1 = 2 * M_PI / 11;
  angle2 = -1 * M_PI / 11;
  angle3 = -4 * M_PI / 11;
  simpleCatmullRomTestPitch(angle0, angle1, angle2, angle3);
}

void testQuaternionCurves() {
  {
    //    Quaternion<double> q0(1,0,0,0);
    //    Quaternion<double> q1(0.974731,0.0899088,0.0735522,-0.190806);
    const Quaternion<double> q0 =
        Quaternion<double>::fromEulerZXY(degToRad(-1.18172), degToRad(-47.7996), degToRad(0.0));

    const Quaternion<double> q1 =
        Quaternion<double>::fromEulerZXY(degToRad(-1.18172), degToRad(-47.7996), degToRad(21.4936));
    //    Quaternion<double> q2(0.783536,0.343338,0.215238,-0.471023);
    const Quaternion<double> q2 =
        Quaternion<double>::fromEulerZXY(degToRad(-1.18172), degToRad(-47.7996), degToRad(61.5007));
    //    Quaternion<double> q3(-0.41248,-0.0943216,0.0365096,0.905335);
    const Quaternion<double> q3 =
        Quaternion<double>::fromEulerZXY(degToRad(-1.18172), degToRad(-47.7996), degToRad(131.079));
    /*
       {
         std::cout << std::endl << "LINEAR" << std::endl;
         Core::SphericalSpline* spline = Core::SphericalSpline::point(0, q0);
         spline->lineTo(31, q1)->lineTo(62, q2)->lineTo(92, q3);
         std::unique_ptr<Core::QuaternionCurve> curve(new Core::QuaternionCurve(spline));
         for (int i = 0; i < 93; ++i) {
           Quaternion<double> t = curve->at(i);
    //        std::cout << t.q0 << " " << t.q1 << " " << t.q2 << " " << t.q3 << std::endl;
           if (i == 0 || i == 31 || i == 62 || i == 92)  std::cout << "  KEYFRAME ";
           double yaw, pitch, roll;
           t.toEuler(yaw, pitch, roll);
           std::cout << radToDeg(yaw) << " " << radToDeg(pitch) << " " << radToDeg(roll) << std::endl;
         }
       }
    */
    {
      std::cout << std::endl << "CUBIC" << std::endl;
      Core::SphericalSpline* spline = Core::SphericalSpline::point(0, q0);
      spline->cubicTo(31, q1)->cubicTo(62, q2)->cubicTo(92, q3);
      std::unique_ptr<Core::QuaternionCurve> curve(new Core::QuaternionCurve(spline));
      for (int i = 0; i < 93; ++i) {
        //        std::cout << t.q0 << " " << t.q1 << " " << t.q2 << " " << t.q3 << std::endl;
        if (i == 0 || i == 31 || i == 62 || i == 92) std::cout << "  KEYFRAME ";
        Quaternion<double> t = curve->at(i);
        double yaw, pitch, roll;
        t.toEuler(yaw, pitch, roll);
        // std::cout << radToDeg(yaw) << " " << radToDeg(pitch) << " " << radToDeg(roll) << std::endl;
      }
      /*
           Quaternion<double> t = curve->at(92);
           double yaw, pitch, roll;
           t.toEuler(yaw, pitch, roll);
           std::cout << radToDeg(yaw) << " " << radToDeg(pitch) << " " << radToDeg(roll) << std::endl;
      */
    }

    {
      Quaternion<double> q0(0.00264554, -0.0161071, -0.0133043, 0.0339792);
      Quaternion<double> q1(0.953552, -0.264985, 0.139959, -0.0305589);
      Core::SphericalSpline* spline = Core::SphericalSpline::point(5856, q0);
      std::unique_ptr<Core::QuaternionCurve> curve(new Core::QuaternionCurve(spline));
      spline->lineTo(5948, q1);
      Quaternion<double> t = curve->at(5890);
      double y, p, r;
      std::cout << "FIRST MIDDLE LAST" << std::endl;
      q0.toEuler(y, p, r);
      std::cout << radToDeg(y) << " " << radToDeg(p) << " " << radToDeg(r) << std::endl;
      t.toEuler(y, p, r);
      std::cout << radToDeg(y) << " " << radToDeg(p) << " " << radToDeg(r) << std::endl;
      q1.toEuler(y, p, r);
      std::cout << radToDeg(y) << " " << radToDeg(p) << " " << radToDeg(r) << std::endl;
    }
  }

  {
    double yaw = 0.642, pitch = -0.312, roll = 1.765;
    Quaternion<double> q = Quaternion<double>::fromEulerZXY(yaw, pitch, roll);
    Matrix33<double> m1 = q.toRotationMatrix();
    Matrix33<double> m2 = Matrix33<double>::fromEulerZXY(yaw, pitch, roll);
    ENSURE_APPROX_EQ(m1(0, 0), m2(0, 0), 0.001);
    ENSURE_APPROX_EQ(m1(0, 1), m2(0, 1), 0.001);
    ENSURE_APPROX_EQ(m1(0, 2), m2(0, 2), 0.001);
    ENSURE_APPROX_EQ(m1(1, 0), m2(1, 0), 0.001);
    ENSURE_APPROX_EQ(m1(1, 1), m2(1, 1), 0.001);
    ENSURE_APPROX_EQ(m1(1, 2), m2(1, 2), 0.001);
    ENSURE_APPROX_EQ(m1(2, 0), m2(2, 0), 0.001);
    ENSURE_APPROX_EQ(m1(2, 1), m2(2, 1), 0.001);
    ENSURE_APPROX_EQ(m1(2, 2), m2(2, 2), 0.001);
  }
}

void testConstant() {
  Core::Curve curve(12.0);
  ENSURE(!curve.splines(), "constant curve should not have splines");
  ENSURE_APPROX_EQ(12.0, curve.at(0), 0.00001);
  ENSURE_APPROX_EQ(12.0, curve.at(20), 0.00001);
  ENSURE(!curve.splines(), "got splines, expected none");

  curve.splitAt(10);
  ENSURE(curve.splines(), "non-constant curve should have splines");
  ENSURE_APPROX_EQ(12.0, curve.at(0), 0.00001);
  ENSURE_APPROX_EQ(12.0, curve.at(20), 0.00001);
  ENSURE(curve.splines(), "got no splines, expected one");
  ENSURE(curve.splines()->getType() == Core::Spline::PointType);

  curve.splines()->end.v = 3.0;
  curve.mergeAt(10);
  ENSURE(!curve.splines(), "constant curve should not have splines");
  ENSURE_APPROX_EQ(3.0, curve.at(0), 0.00001);
  ENSURE_APPROX_EQ(3.0, curve.at(20), 0.00001);
  ENSURE(!curve.splines(), "got splines, expected none");
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  VideoStitch::Testing::testQuaternion();
  VideoStitch::Testing::testCurveExtend();
  VideoStitch::Testing::testLinear();
  VideoStitch::Testing::testCubic();
  VideoStitch::Testing::testCubicGeometryDefinition();
  VideoStitch::Testing::testVSA1090();
  VideoStitch::Testing::testVSA1181();
  VideoStitch::Testing::testQuaternionCurves();
  VideoStitch::Testing::testConstant();
  return 0;
}
