// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "libvideostitch/config.hpp"

#include <memory>

namespace VideoStitch {

namespace Core {
struct PanoDeviceDefinition;
class PanoDefinition;
class InputController;
}  // namespace Core

namespace Ptv {
class Parser;
class Value;
}  // namespace Ptv

namespace Cmd {

bool loadGPUBackend(const int deviceId, int& returnCode);

bool parseInputPath(int argc, char** argv, int index, char** ptvPath);
bool parseOutputPath(int argc, char** argv, int index, char** ptvPath);
bool parseFirstFrame(int argc, char** argv, int index, int* firstFrame);
bool parseLastFrame(int argc, char** argv, int index, int* lastFrame);

bool selectGPUDevice(int argc, char** argv, int& deviceId, int& returnCode);
bool checkGPUDevice(Core::PanoDeviceDefinition& dev);
const char* changeWorkingPathToPtvFolder(char* ptvPath);

std::unique_ptr<Core::PanoDefinition> parsePanoDef(const Ptv::Value& ptvRoot, const char* ptvFile);
bool parsePtvFile(Ptv::Parser& parser, const char* ptvFile);

std::unique_ptr<Core::PanoDefinition> parsePanoDef(Ptv::Parser& parser, const char* ptvFile);

bool normalizeFrameBoundaries(const Core::InputController& controller, const frameid_t firstFrame,
                              frameid_t& lastFrame);

}  // namespace Cmd
}  // namespace VideoStitch
