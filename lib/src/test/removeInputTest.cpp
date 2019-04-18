// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include "libvideostitch/panoDef.hpp"
#include "libvideostitch/parse.hpp"
#include "libvideostitch/ptv.hpp"
#include "libvideostitch/logging.hpp"
#include "libvideostitch/audioPipeDef.hpp"

#include <algorithm>
#include <memory>
#include <sstream>

namespace VideoStitch {
namespace Testing {

void checkAudioInputsMatches(Core::PanoDefinition* panoDef, Core::AudioPipeDefinition* audioPipeDef) {
  std::cout << "check audio pipeline is consistent" << std::endl;
  for (audioreaderid_t i = 0; i < (audioreaderid_t)panoDef->numAudioInputs(); ++i) {
    const std::string& videoName = panoDef->getAudioInput(i).getDisplayName();
    const std::string& audioName = audioPipeDef->getInput(i)->getName();
    ENSURE_EQ(videoName, audioName, "check that the inputs matches");
  }
}

Core::AudioPipeDefinition* getTestAudioPipeDef(Core::PanoDefinition* panoDef) {
  // Create an audio pipeline with the same ammount of pano inputs.
  std::vector<Core::InputParam> inputParams;
  for (audioreaderid_t i = 0; i < panoDef->numAudioInputs(); ++i) {
    inputParams.push_back(
        Core::InputParam(panoDef->getAudioInput(i).getDisplayName(), audioreaderid_t(i), Audio::STEREO));
  }
  std::unique_ptr<Core::AudioPipeDefinition> audioPipe(Core::AudioPipeDefinition::create(inputParams));
  ENSURE_EQ(audioPipe->numAudioInputs(), panoDef->numAudioInputs(),
            "check audio inputs are the same than the pano inputs");
  return audioPipe.release();
}

Core::PanoDefinition* getTestPanoDef() {
  static std::unique_ptr<Core::PanoDefinition> panoDef{[]() -> std::unique_ptr<Core::PanoDefinition> {
    Potential<Ptv::Parser> parser = Ptv::Parser::create();
    std::cout << "parsing" << std::endl;
    if (!parser->parse("data/pano_definition_with_duplicated_inputs.json")) {
      std::cerr << parser->getErrorMessage() << std::endl;
      ENSURE(false, "could not parse");
      return nullptr;
    }
    std::unique_ptr<Core::PanoDefinition> panoDef(Core::PanoDefinition::create(parser->getRoot()));
    ENSURE((bool)panoDef);
    return panoDef;
  }()};
  return panoDef->clone();
}

void checkRemoveFirstInput() {
  std::cout << "creating" << std::endl;
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef());
  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef(getTestAudioPipeDef(panoDef.get()));
  std::stringstream validationMessages;

  ENSURE_EQ(panoDef->numInputs(), (readerid_t)6);
  ENSURE_EQ(panoDef->numVideoInputs(), (videoreaderid_t)5);
  ENSURE_EQ(panoDef->numAudioInputs(), (audioreaderid_t)5);
  ENSURE_EQ(panoDef->convertAudioInputIndexToInputIndex(0), (readerid_t)0);
  ENSURE_EQ(panoDef->convertAudioInputIndexToInputIndex(1), (readerid_t)1);
  ENSURE_EQ(panoDef->convertAudioInputIndexToInputIndex(3), (readerid_t)4);
  ENSURE_EQ(panoDef->convertInputIndexToAudioInputIndex(0), (audioreaderid_t)0);
  ENSURE_EQ(panoDef->convertInputIndexToAudioInputIndex(1), (audioreaderid_t)1);
  ENSURE_EQ(panoDef->convertInputIndexToAudioInputIndex(3), (audioreaderid_t)2);
  ENSURE_EQ(panoDef->convertInputIndexToAudioInputIndex(4), (audioreaderid_t)3);
  ENSURE_EQ(panoDef->convertInputIndexToVideoInputIndex(3), (videoreaderid_t)3);
  ENSURE_EQ(panoDef->convertInputIndexToVideoInputIndex(5), (videoreaderid_t)4);
  ENSURE_EQ(&panoDef->getAudioInput(2), &panoDef->getInput(3));
  ENSURE_EQ(&panoDef->getVideoInput(3), &panoDef->getInput(3));
  ENSURE_EQ(&panoDef->getVideoInput(4), &panoDef->getInput(5));
  ENSURE_EQ(panoDef->getAudioInput(2).getIsAudioEnabled(), true);
  ENSURE_EQ(panoDef->getInput(4).getIsVideoEnabled(), false);
  ENSURE_EQ(panoDef->getInput(4).getIsAudioEnabled(), true);
  ENSURE_EQ(panoDef->getAudioInput(3).getIsVideoEnabled(), false);
  ENSURE_EQ(panoDef->getAudioInput(3).getIsAudioEnabled(), true);
  ENSURE_EQ(panoDef->getInput(5).getIsVideoEnabled(), true);
  ENSURE_EQ(panoDef->getInput(5).getIsAudioEnabled(), true);
  ENSURE_EQ(panoDef->getInput(3).getIsAudioEnabled(), true);
  ENSURE_EQ(panoDef->getInput(0).getIsVideoEnabled(), true);

  std::cout << "removing first input" << std::endl;

  ENSURE(panoDef->removeInput(0));
  const bool ret = panoDef->validate(validationMessages);
  if (!ret) {
    std::cout << validationMessages.str() << std::endl;
  }

  ENSURE(ret);

  ENSURE_EQ(panoDef->numInputs(), (readerid_t)5);
  ENSURE_EQ(panoDef->numInputs() - 1, panoDef->numVideoInputs());
  ENSURE_EQ(panoDef->numAudioInputs(), (audioreaderid_t)4);
  ENSURE_EQ(panoDef->convertAudioInputIndexToInputIndex(0), (readerid_t)0);
  ENSURE_EQ(panoDef->convertAudioInputIndexToInputIndex(1), (readerid_t)2);
  ENSURE_EQ(panoDef->convertAudioInputIndexToInputIndex(2), (readerid_t)3);
  ENSURE_EQ(panoDef->convertInputIndexToAudioInputIndex(0), (audioreaderid_t)0);
  ENSURE_EQ(panoDef->convertInputIndexToAudioInputIndex(2), (audioreaderid_t)1);
  ENSURE_EQ(&panoDef->getAudioInput(1), &panoDef->getInput(2));
  ENSURE_EQ(panoDef->getAudioInput(2).getIsVideoEnabled(), false);
  ENSURE_EQ(panoDef->getAudioInput(2).getIsAudioEnabled(), true);
  ENSURE_EQ(panoDef->getAudioInput(1).getIsAudioEnabled(), true);
  ENSURE_EQ(panoDef->getInput(2).getIsAudioEnabled(), true);

  audioPipeDef->removeInput(0);
  checkAudioInputsMatches(panoDef.get(), audioPipeDef.get());
}

void checkRemoveMiddleInput() {
  std::cout << "creating" << std::endl;
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef());
  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef(getTestAudioPipeDef(panoDef.get()));
  std::stringstream validationMessages;
  bool ret = false;
  readerid_t numInputs = panoDef->numInputs();

  std::cout << "removing middle input" << std::endl;
  ENSURE(panoDef->removeInput((numInputs - 1) / 2));
  ret = panoDef->validate(validationMessages);
  if (!ret) {
    std::cout << validationMessages.str() << std::endl;
  }
  ENSURE(ret);

  ENSURE_EQ(panoDef->numInputs(), (readerid_t)5);
  ENSURE_EQ(panoDef->numInputs() - 1, panoDef->numVideoInputs());
  ENSURE_EQ(panoDef->numAudioInputs(), (audioreaderid_t)5);
  ENSURE_EQ(panoDef->convertAudioInputIndexToInputIndex(0), (readerid_t)0);
  ENSURE_EQ(panoDef->convertAudioInputIndexToInputIndex(1), (readerid_t)1);
  ENSURE_EQ(panoDef->convertAudioInputIndexToInputIndex(3), (readerid_t)3);
  ENSURE_EQ(panoDef->convertInputIndexToAudioInputIndex(0), (audioreaderid_t)0);
  ENSURE_EQ(panoDef->convertInputIndexToAudioInputIndex(2), (audioreaderid_t)2);
  ENSURE_EQ(panoDef->convertInputIndexToAudioInputIndex(3), (audioreaderid_t)3);
  ENSURE_EQ(panoDef->convertInputIndexToVideoInputIndex(4), (videoreaderid_t)3);
  ENSURE_EQ(&panoDef->getAudioInput(1), &panoDef->getInput(1));
  ENSURE_EQ(panoDef->getAudioInput(3).getIsVideoEnabled(), false);
  ENSURE_EQ(panoDef->getAudioInput(3).getIsAudioEnabled(), true);
  ENSURE_EQ(panoDef->getAudioInput(1).getIsAudioEnabled(), true);
  ENSURE_EQ(panoDef->getInput(2).getIsAudioEnabled(), true);

  checkAudioInputsMatches(panoDef.get(), audioPipeDef.get());
}

void checkRemoveLastInputs() {
  std::cout << "creating" << std::endl;
  std::unique_ptr<Core::PanoDefinition> panoDef(getTestPanoDef());
  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef(getTestAudioPipeDef(panoDef.get()));
  std::stringstream validationMessages;
  bool ret = false;

  while (panoDef->numInputs() > 1) {
    std::cout << "removing last input" << std::endl;
    const readerid_t toRemove = panoDef->numInputs() - 1;
    ENSURE(panoDef->removeInput(toRemove));
    ret = panoDef->validate(validationMessages);
    if (!ret) {
      std::cout << validationMessages.str() << std::endl;
    }
    ENSURE(ret);
    audioPipeDef->removeInput(toRemove);
  }
  checkAudioInputsMatches(panoDef.get(), audioPipeDef.get());
}
}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initTest();

  // This checks the consistency of the PanoDefinition after an input is removed.
  VideoStitch::Testing::checkRemoveFirstInput();
  VideoStitch::Testing::checkRemoveMiddleInput();
  VideoStitch::Testing::checkRemoveLastInputs();
  return 0;
}
