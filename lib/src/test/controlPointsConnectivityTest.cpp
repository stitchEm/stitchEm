// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/logging.hpp"
#include <calibration/calibration.hpp>

#include <algorithm>
#include <memory>
#include <sstream>

using namespace VideoStitch;

namespace VideoStitch {
namespace Testing {

std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> parseIntoKeyPointsMap(
    const std::string jsonFileName) {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  Logger::get(Logger::Info) << "parsing" << std::endl;
  if (!parser->parse(jsonFileName)) {
    Logger::get(Logger::Error) << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }
  Logger::get(Logger::Info) << "creating" << std::endl;
  Potential<Core::ControlPointListDefinition> cplDef = Core::ControlPointListDefinition::create(parser->getRoot());
  ENSURE(cplDef.ok());
  auto keypoints = cplDef->getCalibrationControlPointList();

  std::map<std::pair<videoreaderid_t, videoreaderid_t>, Core::ControlPointList> keypoints_map;

  /*Get control points from config object and add them to configmatches_map map*/
  for (auto& cp : keypoints) {
    keypoints_map[{cp.index0, cp.index1}].push_back(cp);
  }

  return keypoints_map;
}

void stringstreamReset(std::stringstream& msg) {
  msg.str("");
  msg.clear();
}

void testKeypointsConnectivity() {
  auto keypoints_map = parseIntoKeyPointsMap("data/controlKeypoints.json");
  ENSURE(!keypoints_map.empty());

  std::map<videoreaderid_t, std::set<videoreaderid_t>> connectiviy;
  std::set<videoreaderid_t> singleConnectedInputs;
  std::set<videoreaderid_t> nonConnectedInputs;
  std::stringstream report;

  // has 6 cameras, analyzing connectivity with 7 should fail
  ENSURE(!Calibration::Calibration::analyzeKeypointsConnectivity(keypoints_map, 7, &report));
  Logger::get(Logger::Info) << "report: " << report.str() << std::endl;
  stringstreamReset(report);

  ENSURE(Calibration::Calibration::analyzeKeypointsConnectivity(keypoints_map, 6, &report, &connectiviy,
                                                                &singleConnectedInputs, &nonConnectedInputs));
  Logger::get(Logger::Info) << "report: " << report.str() << std::endl;
  stringstreamReset(report);

  ENSURE_EQ(singleConnectedInputs.size(), (size_t)0);
  ENSURE_EQ(nonConnectedInputs.size(), (size_t)0);
  ENSURE_EQ(connectiviy[0].size(), (size_t)4);
  ENSURE(connectiviy[0].find(2) != connectiviy[0].end());
  ENSURE(connectiviy[0].find(3) != connectiviy[0].end());
  ENSURE(connectiviy[0].find(4) != connectiviy[0].end());
  ENSURE(connectiviy[0].find(5) != connectiviy[0].end());

  ENSURE_EQ(connectiviy[1].size(), (size_t)2);
  ENSURE(connectiviy[1].find(2) != connectiviy[0].end());
  ENSURE(connectiviy[1].find(4) != connectiviy[0].end());

  // remove connectiviy 1 <-> 2
  keypoints_map[{(videoreaderid_t)1, (videoreaderid_t)2}].clear();

  ENSURE(Calibration::Calibration::analyzeKeypointsConnectivity(keypoints_map, 6, &report, &connectiviy,
                                                                &singleConnectedInputs, &nonConnectedInputs));
  Logger::get(Logger::Info) << "report: " << report.str() << std::endl;
  stringstreamReset(report);

  // 1 is a single connected input
  ENSURE_EQ(singleConnectedInputs.size(), (size_t)1);
  ENSURE(singleConnectedInputs.find((videoreaderid_t)1) != singleConnectedInputs.end());

  // remove connectivity 3 <-> 0, 3 <-> 4 and 3 <-> 5
  keypoints_map.erase({(videoreaderid_t)3, (videoreaderid_t)0});
  keypoints_map.erase({(videoreaderid_t)0, (videoreaderid_t)3});
  keypoints_map.erase({(videoreaderid_t)3, (videoreaderid_t)4});
  keypoints_map.erase({(videoreaderid_t)4, (videoreaderid_t)3});
  keypoints_map.erase({(videoreaderid_t)3, (videoreaderid_t)5});
  keypoints_map.erase({(videoreaderid_t)5, (videoreaderid_t)3});

  ENSURE(!Calibration::Calibration::analyzeKeypointsConnectivity(keypoints_map, 6, &report, &connectiviy,
                                                                 &singleConnectedInputs, &nonConnectedInputs));
  Logger::get(Logger::Info) << "report: " << report.str() << std::endl;
  stringstreamReset(report);

  // 3 is not connected
  ENSURE_EQ(nonConnectedInputs.size(), (size_t)1);
  ENSURE(nonConnectedInputs.find((videoreaderid_t)3) != nonConnectedInputs.end());
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  // This tests Calibration::Calibration::analyzeKeypointsConnectivity()
  VideoStitch::Testing::testKeypointsConnectivity();
  return 0;
}
