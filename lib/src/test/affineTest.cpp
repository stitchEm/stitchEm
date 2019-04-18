// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <motion/affineMotion.hpp>
#include <parse/json.hpp>
#include "libvideostitch/panoDef.hpp"

#include <random>
#include <memory>

namespace VideoStitch {
namespace Testing {

class AffineMotionEstimationTest : public Motion::AffineMotionModelEstimation {
 public:
  void testMotion(const int64_t panoWidth, const int64_t panoHeight) {
    const std::unique_ptr<VideoStitch::Ptv::Value> ptv(createMinimalPTV(panoWidth, panoHeight));
    // PanoDefinition creation should fill with default values
    const std::unique_ptr<VideoStitch::Core::PanoDefinition> pano(VideoStitch::Core::PanoDefinition::create(*ptv));
    ENSURE((bool)pano, "PanoDefinition creation failed. Needed value to build it may have changed.");
    testMotion(*pano);
  }

 private:
  static Ptv::Value *createMinimalPTV(const int64_t panoWidth, const int64_t panoHeight) {
    // ************* ALL THE VALUES BETWEEN COMMENTS ARE REQUIRED (begin) *************
    // build minimal global PTV
    Ptv::Value *ptv = Ptv::Value::emptyObject();
    ptv->push("width", new Parse::JsonValue(panoWidth));
    ptv->push("height", new Parse::JsonValue(panoHeight));
    ptv->push("hfov", new Parse::JsonValue(360));
    ptv->push("proj", new Parse::JsonValue("equirectangular"));

    // add an input (required by Controller)
    Ptv::Value *jsonInputs = new Parse::JsonValue((void *)NULL);
    Ptv::Value *input = Ptv::Value::emptyObject();
    input->push("width", new Parse::JsonValue(panoWidth));
    input->push("height", new Parse::JsonValue(panoHeight));
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

  void testMotion(const Core::PanoDefinition &pano) {
    Motion::ImageSpace::MotionVectorField field;

    std::vector<float> ground_truth;
    const float theta = (float)M_PI_4;
    ground_truth.push_back(cosf(theta));
    ground_truth.push_back(-sinf(theta));
    ground_truth.push_back(10.1f);
    ground_truth.push_back(sinf(theta));
    ground_truth.push_back(cosf(theta));
    ground_truth.push_back(-3.0f);

    // create 10 points on the [0, pano.getWidth()] x [0, pano.getHeight()] plan
    // generate 3 gaussian variables x, y, z
    // the distribution of vectors is now uniform
    std::default_random_engine generator;
    std::uniform_int_distribution<int> xdist;
    std::uniform_int_distribution<int> ydist;
    for (int i = 0; i < 150; ++i) {
      float2 src;
      src.x = (float)(xdist(generator) % pano.getWidth());
      src.y = (float)(ydist(generator) % pano.getHeight());
      // move by the ground-truth affine transform
      float2 dst;
      dst.x = (float)((src.x - (float)pano.getWidth() / 2.0f) * ground_truth[0] +
                      (src.y - (float)pano.getHeight() / 2.0f) * ground_truth[1] + ground_truth[2]) +
              (float)pano.getWidth() / 2.0f;
      dst.y = (float)((src.x - (float)pano.getWidth() / 2.0f) * ground_truth[3] +
                      (src.y - (float)pano.getHeight() / 2.0f) * ground_truth[4] + ground_truth[5]) +
              (float)pano.getHeight() / 2.0f;
      field.push_back(Motion::ImageSpace::MotionVector(src, dst));
    }

    // add a few crazy outliers
    for (int j = 0; j < 15; ++j) {
      float2 src;
      src.x = (float)(xdist(generator) % pano.getWidth());
      src.y = (float)(ydist(generator) % pano.getHeight());
      float2 dst;
      dst.x = (float)(xdist(generator) % pano.getWidth());
      dst.y = (float)(ydist(generator) % pano.getHeight());
      field.push_back(Motion::ImageSpace::MotionVector(src, dst));
    }

    // solve the optimization problem
    Motion::ImageSpace::MotionVectorFieldTimeSeries ts;
    ts[0] = field;
    MotionModel model;
    motionModel(ts, model, pano.getInput(0));

    std::cout << "Model:" << std::endl;
    std::cout << model[0].second(0, 0) << " " << model[0].second(0, 1) << " " << model[0].second(0, 2) << std::endl;
    std::cout << model[0].second(1, 0) << " " << model[0].second(1, 1) << " " << model[0].second(1, 2) << std::endl;
    std::cout << "Ground truth:" << std::endl;
    std::cout << ground_truth[0] << " " << ground_truth[1] << " " << ground_truth[2] << std::endl;
    std::cout << ground_truth[3] << " " << ground_truth[4] << " " << ground_truth[5] << std::endl;
    std::cout << std::endl;

    ENSURE_APPROX_EQ(ground_truth[0], (float)model[0].second(0, 0), 0.0001f);
    ENSURE_APPROX_EQ(ground_truth[1], (float)model[0].second(0, 1), 0.0001f);
    ENSURE_APPROX_EQ(ground_truth[2], (float)model[0].second(0, 2), 0.0001f);
    ENSURE_APPROX_EQ(ground_truth[3], (float)model[0].second(1, 0), 0.0001f);
    ENSURE_APPROX_EQ(ground_truth[4], (float)model[0].second(1, 1), 0.0001f);
    ENSURE_APPROX_EQ(ground_truth[5], (float)model[0].second(1, 2), 0.0001f);
  }
};

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::AffineMotionEstimationTest test;
  test.testMotion(1024, 512);
  test.testMotion(1025, 512);
  return 0;
}
