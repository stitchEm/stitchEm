// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <common/angles.hpp>
#include <motion/rotationalMotion.hpp>
#include <parse/json.hpp>
#include "libvideostitch/quaternion.hpp"
#include "libvideostitch/panoDef.hpp"

#include <random>
#include <memory>

namespace VideoStitch {
namespace Testing {

Ptv::Value *createMinimalPTV() {
  // ************* ALL THE VALUES BETWEEN COMMENTS ARE REQUIRED (begin) *************
  // build minimal global PTV
  Ptv::Value *ptv = Ptv::Value::emptyObject();
  ptv->push("width", new Parse::JsonValue(1024));
  ptv->push("height", new Parse::JsonValue(512));
  ptv->push("hfov", new Parse::JsonValue(360));
  ptv->push("proj", new Parse::JsonValue("equirectangular"));

  // add an input (required by Controller)
  Ptv::Value *jsonInputs = new Parse::JsonValue((void *)NULL);
  Ptv::Value *input = Ptv::Value::emptyObject();
  input->push("width", new Parse::JsonValue(1024));
  input->push("height", new Parse::JsonValue(512));
  input->push("hfov", new Parse::JsonValue(360));
  input->push("yaw", new Parse::JsonValue(0.0));
  input->push("pitch", new Parse::JsonValue(0.0));
  input->push("roll", new Parse::JsonValue(0.0));
  input->push("proj", new Parse::JsonValue("equirectangular"));
  input->push("viewpoint_model", new Parse::JsonValue("hugin"));
  input->push("response", new Parse::JsonValue("emor"));

  // add a procedural input
  Ptv::Value *inputConfig = Ptv::Value::emptyObject();
  inputConfig->push("filename", new Parse::JsonValue("toto"));
  inputConfig->push("type", new Parse::JsonValue("procedural"));
  inputConfig->push("name", new Parse::JsonValue("frameNumber"));
  input->push("reader_config", inputConfig);
  jsonInputs->asList().push_back(input);
  ptv->push("inputs", jsonInputs);
  // ************* ALL THE VALUES BETWEEN COMMENTS ARE REQUIRED (end) *************

  return ptv;
}

/**
 * @brief Check that 2 quaternions are equal (up to rounding errors)
 */
template <typename T>
void checkEqualQuaternions(const Quaternion<T> &qFirst, const Quaternion<T> &qSecond, double tolerance = 1e-6) {
  ENSURE_APPROX_EQ(qFirst.getQ0(), qSecond.getQ0(), tolerance);
  ENSURE_APPROX_EQ(qFirst.getQ1(), qSecond.getQ1(), tolerance);
  ENSURE_APPROX_EQ(qFirst.getQ2(), qSecond.getQ2(), tolerance);
  ENSURE_APPROX_EQ(qFirst.getQ3(), qSecond.getQ3(), tolerance);
}

/**
 * Check that we can go back and forth from Quaternions to 3x3 rotation matrices
 */
template <typename T>
void checkConsistencyQuaternionMatrix(Vector3<T> v) {
  Quaternion<double> q_rot = Quaternion<double>::fromAxisAngle(v);

  double yaw, pitch, roll;
  q_rot.toEuler(yaw, pitch, roll);

  Quaternion<double> q_fromYPR = Quaternion<double>::fromEulerZXY(yaw, pitch, roll);
  checkEqualQuaternions(q_rot, q_fromYPR);

  Matrix33<double> M = Matrix33<double>::fromEulerZXY(yaw, pitch, roll);
  ENSURE_APPROX_EQ(M.det(), 1., 1e-6);
  Quaternion<double> qM = Quaternion<double>::fromRotationMatrix(M);
  checkEqualQuaternions(q_rot, qM);

  // rotate unit vector along X axis
  Vector3<double> vx(1, 0, 0);
  Vector3<double> vx_rot = M * vx;
  Quaternion<double> px(0, 1, 0, 0);
  Quaternion<double> px_rot = q_rot * px * q_rot.conjugate();
  ENSURE_APPROX_EQ(px_rot.getQ0(), 0., 1e-6);
  ENSURE_APPROX_EQ(px_rot.getQ1(), vx_rot(0), 1e-6);
  ENSURE_APPROX_EQ(px_rot.getQ2(), vx_rot(1), 1e-6);
  ENSURE_APPROX_EQ(px_rot.getQ3(), vx_rot(2), 1e-6);

  // rotate unit vector along Y axis
  Vector3<double> vy(0, 1, 0);
  Vector3<double> vy_rot = M * vy;
  Quaternion<double> py(0, 0, 1, 0);
  Quaternion<double> py_rot = q_rot * py * q_rot.conjugate();
  ENSURE_APPROX_EQ(py_rot.getQ0(), 0., 1e-6);
  ENSURE_APPROX_EQ(py_rot.getQ1(), vy_rot(0), 1e-6);
  ENSURE_APPROX_EQ(py_rot.getQ2(), vy_rot(1), 1e-6);
  ENSURE_APPROX_EQ(py_rot.getQ3(), vy_rot(2), 1e-6);

  // rotate unit vector along Z axis
  Vector3<double> vz(0, 0, 1);
  Vector3<double> vz_rot = M * vz;
  Quaternion<double> pz(0, 0, 0, 1);
  Quaternion<double> pz_rot = q_rot * pz * q_rot.conjugate();
  ENSURE_APPROX_EQ(pz_rot.getQ0(), 0., 1e-6);
  ENSURE_APPROX_EQ(pz_rot.getQ1(), vz_rot(0), 1e-6);
  ENSURE_APPROX_EQ(pz_rot.getQ2(), vz_rot(1), 1e-6);
  ENSURE_APPROX_EQ(pz_rot.getQ3(), vz_rot(2), 1e-6);
}

class RotationalMotionEstimationTest : public Motion::RotationalMotionModelEstimation {
 public:
  explicit RotationalMotionEstimationTest(Core::PanoDefinition &pano) : Motion::RotationalMotionModelEstimation(pano) {}

  void testConversions() {
    double x = 0, y = 0, z = 0;
    Vector3<double> v(0., 0., 0.);
    Quaternion<double> q_fromYPR;
    Quaternion<double> q_fromMatrix;
    double yaw, pitch, roll;
    double angle = degToRad(25);

    // Rotation around X axis
    x = angle;
    y = 0;
    z = 0;
    v = Vector3<double>(x, y, z);
    Quaternion<double> qX = Quaternion<double>::fromAxisAngle(v);
    ENSURE_APPROX_EQ(qX.getQ0(), cos(angle / 2.), 1e-6);
    ENSURE_APPROX_EQ(qX.getQ1(), sin(angle / 2.), 1e-6);
    ENSURE_APPROX_EQ(qX.getQ2(), 0., 1e-6);
    ENSURE_APPROX_EQ(qX.getQ3(), 0., 1e-6);

    qX.toEuler(yaw, pitch, roll);
    q_fromYPR = Quaternion<double>::fromEulerZXY(yaw, pitch, roll);
    checkEqualQuaternions(qX, q_fromYPR);

    Matrix33<double> Mx = Matrix33<double>::fromEulerZXY(yaw, pitch, roll);
    ENSURE_APPROX_EQ(Mx(0, 0), 1., 1e-6);
    ENSURE_APPROX_EQ(Mx(0, 1), 0., 1e-6);
    ENSURE_APPROX_EQ(Mx(0, 2), 0., 1e-6);
    ENSURE_APPROX_EQ(Mx(1, 0), 0., 1e-6);
    ENSURE_APPROX_EQ(Mx(1, 1), cos(angle), 1e-6);
    ENSURE_APPROX_EQ(Mx(1, 2), -sin(angle), 1e-6);
    ENSURE_APPROX_EQ(Mx(2, 0), 0., 1e-6);
    ENSURE_APPROX_EQ(Mx(2, 1), sin(angle), 1e-6);
    ENSURE_APPROX_EQ(Mx(2, 2), cos(angle), 1e-6);

    q_fromMatrix = Quaternion<double>::fromRotationMatrix(Mx);
    checkEqualQuaternions(qX, q_fromMatrix);

    // Rotation around Y axis
    x = 0;
    y = angle;
    z = 0;
    v = Vector3<double>(x, y, z);
    Quaternion<double> qY = Quaternion<double>::fromAxisAngle(v);
    ENSURE_APPROX_EQ(qY.getQ0(), cos(angle / 2.), 1e-6);
    ENSURE_APPROX_EQ(qY.getQ1(), 0., 1e-6);
    ENSURE_APPROX_EQ(qY.getQ2(), sin(angle / 2.), 1e-6);
    ENSURE_APPROX_EQ(qY.getQ3(), 0., 1e-6);

    qY.toEuler(yaw, pitch, roll);
    q_fromYPR = Quaternion<double>::fromEulerZXY(yaw, pitch, roll);
    checkEqualQuaternions(qY, q_fromYPR);

    Matrix33<double> My = Matrix33<double>::fromEulerZXY(yaw, pitch, roll);
    ENSURE_APPROX_EQ(My(0, 0), cos(angle), 1e-6);
    ENSURE_APPROX_EQ(My(0, 1), 0., 1e-6);
    ENSURE_APPROX_EQ(My(0, 2), sin(angle), 1e-6);
    ENSURE_APPROX_EQ(My(1, 2), 0., 1e-6);
    ENSURE_APPROX_EQ(My(1, 1), 1., 1e-6);
    ENSURE_APPROX_EQ(My(1, 2), 0., 1e-6);
    ENSURE_APPROX_EQ(My(2, 0), -sin(angle), 1e-6);
    ENSURE_APPROX_EQ(My(2, 1), 0., 1e-6);
    ENSURE_APPROX_EQ(My(2, 2), cos(angle), 1e-6);

    q_fromMatrix = Quaternion<double>::fromRotationMatrix(My);
    checkEqualQuaternions(qY, q_fromMatrix);

    // Rotation around Z axis
    x = 0;
    y = 0;
    z = angle;
    v = Vector3<double>(x, y, z);
    Quaternion<double> qZ = Quaternion<double>::fromAxisAngle(v);
    ENSURE_APPROX_EQ(qZ.getQ0(), cos(angle / 2.), 1e-6);
    ENSURE_APPROX_EQ(qZ.getQ1(), 0., 1e-6);
    ENSURE_APPROX_EQ(qZ.getQ2(), 0., 1e-6);
    ENSURE_APPROX_EQ(qZ.getQ3(), sin(angle / 2.), 1e-6);

    qZ.toEuler(yaw, pitch, roll);
    q_fromYPR = Quaternion<double>::fromEulerZXY(yaw, pitch, roll);
    checkEqualQuaternions(qZ, q_fromYPR);

    Matrix33<double> Mz = Matrix33<double>::fromEulerZXY(yaw, pitch, roll);
    ENSURE_APPROX_EQ(Mz(0, 0), cos(angle), 1e-6);
    ENSURE_APPROX_EQ(Mz(0, 1), -sin(angle), 1e-6);
    ENSURE_APPROX_EQ(Mz(0, 2), 0., 1e-6);
    ENSURE_APPROX_EQ(Mz(1, 0), sin(angle), 1e-6);
    ENSURE_APPROX_EQ(Mz(1, 1), cos(angle), 1e-6);
    ENSURE_APPROX_EQ(Mz(1, 2), 0., 1e-6);
    ENSURE_APPROX_EQ(Mz(2, 0), 0., 1e-6);
    ENSURE_APPROX_EQ(Mz(2, 1), 0., 1e-6);
    ENSURE_APPROX_EQ(Mz(2, 2), 1., 1e-6);

    q_fromMatrix = Quaternion<double>::fromRotationMatrix(Mz);
    checkEqualQuaternions(qZ, q_fromMatrix);

    /////////
    // Rotation of 120 degrees around axis (1/sqrt(3) , 1/sqrt(3), 1/sqrt(3))
    angle = degToRad(120);
    x = y = z = angle / sqrt(3.);
    v = Vector3<double>(x, y, z);
    Quaternion<double> q_rot = Quaternion<double>::fromAxisAngle(v);
    ENSURE_APPROX_EQ(q_rot.getQ0(), 0.5, 1e-6);
    ENSURE_APPROX_EQ(q_rot.getQ1(), 0.5, 1e-6);
    ENSURE_APPROX_EQ(q_rot.getQ2(), 0.5, 1e-6);
    ENSURE_APPROX_EQ(q_rot.getQ3(), 0.5, 1e-6);

    q_rot.toEuler(yaw, pitch, roll);
    q_fromYPR = Quaternion<double>::fromEulerZXY(yaw, pitch, roll);
    checkEqualQuaternions(q_fromYPR, q_rot);

    Quaternion<double> px(0, 1, 0, 0);
    Quaternion<double> px_rot = q_rot * px * q_rot.conjugate();  ///< (1,0,0) is mapped to (0,1,0)
    ENSURE_APPROX_EQ(px_rot.getQ0(), 0., 1e-6);
    ENSURE_APPROX_EQ(px_rot.getQ1(), 0., 1e-6);
    ENSURE_APPROX_EQ(px_rot.getQ2(), 1., 1e-6);
    ENSURE_APPROX_EQ(px_rot.getQ3(), 0., 1e-6);

    Quaternion<double> py(0, 0, 1, 0);
    Quaternion<double> py_rot = q_rot * py * q_rot.conjugate();  ///< (0,1,0) is mapped to (0,0,1)
    ENSURE_APPROX_EQ(py_rot.getQ0(), 0., 1e-6);
    ENSURE_APPROX_EQ(py_rot.getQ1(), 0., 1e-6);
    ENSURE_APPROX_EQ(py_rot.getQ2(), 0., 1e-6);
    ENSURE_APPROX_EQ(py_rot.getQ3(), 1., 1e-6);

    Quaternion<double> pz(0, 0, 0, 1);
    Quaternion<double> pz_rot = q_rot * pz * q_rot.conjugate();  ///< (0,0,1) is mapped to (1,0,0)
    ENSURE_APPROX_EQ(pz_rot.getQ0(), 0., 1e-6);
    ENSURE_APPROX_EQ(pz_rot.getQ1(), 1., 1e-6);
    ENSURE_APPROX_EQ(pz_rot.getQ2(), 0., 1e-6);
    ENSURE_APPROX_EQ(pz_rot.getQ3(), 0., 1e-6);

    checkConsistencyQuaternionMatrix(v);
    /////////

    // Check with another rotation
    v = Vector3<double>(0.9432381304920563, 0.12571967532785994, 0.05507618306922233);
    checkConsistencyQuaternionMatrix(v);

    // Check with another rotation
    v = Vector3<double>(0.0, -1.1197138652874574, 0.4571212713122885);
    checkConsistencyQuaternionMatrix(v);
  }

  void testRotation() {
    Motion::SphericalSpace::MotionVectorField field;

    double findYaw = 15.0, findPitch = 30.0, findRoll = -15.0;
    Quaternion<double> findMe =
        Quaternion<double>::fromEulerZXY(degToRad(findYaw), degToRad(findPitch), degToRad(findRoll));

    // create 10 points on the unit sphere randomly
    // generate 3 gaussian variables x, y, z
    // the distribution of vectors is now uniform
    std::default_random_engine generator;
    std::normal_distribution<double> xdist;
    std::normal_distribution<double> ydist;
    std::normal_distribution<double> zdist;
    for (int i = 0; i < 10; ++i) {
      double x = xdist(generator);
      double y = ydist(generator);
      double z = zdist(generator);
      double norm = sqrt(x * x + y * y + z * z);
      Quaternion<double> src(0, x / norm, y / norm, z / norm);
      // rotate by the ground-truth quaternion
      Quaternion<double> dst = findMe * src * findMe.conjugate();
      field.push_back(Motion::SphericalSpace::MotionVector(src, dst));
    }

    // add a few crazy outliers
    for (int j = 0; j < 2; ++j) {
      double x = xdist(generator);
      double y = ydist(generator);
      double z = zdist(generator);
      double norm = sqrt(x * x + y * y + z * z);
      Quaternion<double> src(0, x / norm, y / norm, z / norm);
      x = xdist(generator);
      y = ydist(generator);
      z = zdist(generator);
      norm = sqrt(x * x + y * y + z * z);
      Quaternion<double> dst(0, x / norm, y / norm, z / norm);
      field.push_back(Motion::SphericalSpace::MotionVector(src, dst));
    }

    // solve the optimization problem
    Motion::SphericalSpace::MotionVectorFieldTimeSeries ts;
    ts[0] = field;
    MotionModel model;
    motionModel(ts, model);

    double yaw, pitch, roll;
    model[0].toEuler(yaw, pitch, roll);

    ENSURE_APPROX_EQ(radToDeg(yaw), findYaw, 0.0001);
    ENSURE_APPROX_EQ(radToDeg(pitch), findPitch, 0.0001);
    ENSURE_APPROX_EQ(radToDeg(roll), findRoll, 0.0001);
  }

  void testRotationRansac() {
    Motion::SphericalSpace::MotionVectorField field;

    double findYaw = 15.0, findPitch = 30.0, findRoll = -15.0;
    Quaternion<double> findMe =
        Quaternion<double>::fromEulerZXY(degToRad(findYaw), degToRad(findPitch), degToRad(findRoll));

    // create 10 points on the unit sphere randomly
    // generate 3 gaussian variables x, y, z
    // the distribution of vectors is now uniform
    std::default_random_engine generator;
    std::normal_distribution<double> xdist;
    std::normal_distribution<double> ydist;
    std::normal_distribution<double> zdist;
    for (int i = 0; i < 10; ++i) {
      double x = xdist(generator);
      double y = ydist(generator);
      double z = zdist(generator);
      double norm = sqrt(x * x + y * y + z * z);
      Quaternion<double> src(0, x / norm, y / norm, z / norm);
      // rotate by the ground-truth quaternion
      Quaternion<double> dst = findMe * src * findMe.conjugate();
      field.push_back(Motion::SphericalSpace::MotionVector(src, dst));
    }

    // add a few crazy outliers
    for (int j = 0; j < 2; ++j) {
      double x = xdist(generator);
      double y = ydist(generator);
      double z = zdist(generator);
      double norm = sqrt(x * x + y * y + z * z);
      Quaternion<double> src(0, x / norm, y / norm, z / norm);
      x = xdist(generator);
      y = ydist(generator);
      z = zdist(generator);
      norm = sqrt(x * x + y * y + z * z);
      Quaternion<double> dst(0, x / norm, y / norm, z / norm);
      field.push_back(Motion::SphericalSpace::MotionVector(src, dst));
    }

    std::default_random_engine gen(42);
    Motion::RotationRansac rotationRansac(field, 0.026185, 3, field.size() / 2, gen);
    Quaternion<double> qRot;
    bool ransacResult = rotationRansac.ransac(qRot);

    ENSURE(ransacResult);
    checkEqualQuaternions(findMe, qRot);

    double yaw, pitch, roll;
    qRot.toEuler(yaw, pitch, roll);
    ENSURE_APPROX_EQ(radToDeg(yaw), findYaw, 1e-6);
    ENSURE_APPROX_EQ(radToDeg(pitch), findPitch, 1e-6);
    ENSURE_APPROX_EQ(radToDeg(roll), findRoll, 1e-6);
  }
};

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  std::unique_ptr<VideoStitch::Ptv::Value> ptv(VideoStitch::Testing::createMinimalPTV());
  // PanoDefinition creation should fill with default values
  std::unique_ptr<VideoStitch::Core::PanoDefinition> pano(VideoStitch::Core::PanoDefinition::create(*ptv.get()));
  if (!pano.get()) {
    std::cout << "PanoDefinition creation failed. Needed value to build it may have changed." << std::endl;
    return 1;
  }
  VideoStitch::Testing::RotationalMotionEstimationTest test(*pano);
  test.testConversions();
  test.testRotation();
  test.testRotationRansac();
  return 0;
}
