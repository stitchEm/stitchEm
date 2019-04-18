// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <parse/json.hpp>
#include "libvideostitch/audioPipeDef.hpp"
#include "libvideostitch/parse.hpp"
#include "audio/audioPipeFactory.hpp"
#include "audio/audioPipeline.hpp"
#include "audio/sampleDelay.hpp"
#include "audio/gain.hpp"

#include <fstream>
#include <memory>

namespace VideoStitch {
namespace Testing {
using namespace Core;
using namespace Audio;

static const std::string DEFAULT_PANO_FILENAME = "data/4i_default_pano_definition.json";

PanoDefinition *getDefaultPano() {
  auto parser = Ptv::Parser::create();

  if (!parser->parse(DEFAULT_PANO_FILENAME)) {
    std::cerr << parser->getErrorMessage() << std::endl;
    ENSURE(false);
  }

  return PanoDefinition::create(parser->getRoot());
}

// For testing purpose we serialize the audio pipe and recreate it afterwards to check serialization is passing fine
AudioPipeDefinition *reCreateAudiopipeAfterSerialization(AudioPipeDefinition *audioPipeDefBeforeSerialization) {
  // Serialize the audio pipe definition
  std::unique_ptr<Ptv::Value> pipeSerialized(audioPipeDefBeforeSerialization->serialize());
  std::ofstream os;
  std::string testData = getDataFolder();
  std::string pipeFileName = testData + "/titi.json";
  os.open(pipeFileName, std::ios::out);
  pipeSerialized->printJson(os);
  os.close();

  // Re open it and check everything is as expected
  Potential<Ptv::Parser> parser(Ptv::Parser::create());
  ENSURE(parser->parse(pipeFileName));
  std::unique_ptr<Ptv::Value> ptv(parser->getRoot().clone());
  std::unique_ptr<Ptv::Value> audioPipeConfig(ptv->clone());
  return AudioPipeDefinition::create(*audioPipeConfig);
}

void testCreateAudioPipeDef() {
  std::cout << "TEST creation of audio pipe definition" << std::endl;
  AudioPipeDefinition *audioPipeDef(AudioPipeDefinition::createDefault());
  ENSURE_EQ(audioPipeDef->numAudioInputs(), 0, "check default number of inputs");

  // Add a new input with 2 sources to the audio pipe
  InputParam firstInputParam("my_first_input", 0, STEREO);
  std::unique_ptr<AudioInputDefinition> newInput(AudioInputDefinition::create(firstInputParam));
  ENSURE_EQ((size_t)2, newInput->numSources(), "check new number of sources");
  newInput->setIsMaster(true);
  ENSURE(!newInput->setLayout("wrongLayout").ok(), "Test setting a wrong layout");

  newInput->setLayout(getStringFromChannelLayout(STEREO));
  newInput->getSource(0)->setReaderId(0);
  newInput->getSource(0)->setChannel(0);

  newInput->getSource(1)->setReaderId(1);
  newInput->getSource(1)->setChannel(1);

  audioPipeDef->addInput(newInput.release());

  // Add a second input with 2 sources to the audio pipe
  InputParam secondInputParam("my_second_input", 1, STEREO);
  std::unique_ptr<AudioInputDefinition> secondInput(AudioInputDefinition::create(secondInputParam));
  secondInput->getSource(0)->setReaderId(0);
  secondInput->getSource(0)->setChannel(-1);
  audioPipeDef->addInput(secondInput.release());

  ENSURE_EQ(2, audioPipeDef->numAudioInputs(), "check new inputs have been created");
  ENSURE_EQ(audioPipeDef->getInput(0)->getName(), firstInputParam.name, "check default reader id");
  ENSURE_EQ(audioPipeDef->getInput(0)->getSource(0)->getReaderId(), (readerid_t)0, "check default reader id");
  ENSURE_EQ(audioPipeDef->getInput(0)->getSource(0)->getChannel(), (size_t)0, "check default channel");
  ENSURE_EQ(audioPipeDef->getInput(0)->getSource(1)->getReaderId(), (readerid_t)1, "check default reader id");
  ENSURE_EQ(audioPipeDef->getInput(0)->getSource(1)->getChannel(), (size_t)1, "check default channel");
  ENSURE_EQ(true, audioPipeDef->getInput(0)->getIsMaster(), "check is master");
  ENSURE_EQ(std::string("stereo"), audioPipeDef->getInput(0)->getLayout(), "check layout");

  // Add a a delay processor
  std::cout << "TEST add delay processor" << std::endl;
  double delay_s = 3.0;
  audioPipeDef->addDelayProcessor(secondInputParam.name, delay_s);

  audioPipeDef->addDelayProcessor(firstInputParam.name, 1.1);
  ENSURE_EQ(delay_s, audioPipeDef->getDelay(secondInputParam.name).value(), "Check creation of delay processor");
  ENSURE_EQ(1.1, audioPipeDef->getDelay(firstInputParam.name).value(), "Check add delay processor for an other input");

  // Check invalid input name at the creation of the delay processor
  ENSURE(!audioPipeDef->addDelayProcessor("invalidName", delay_s).ok());

  // Add a gain processor
  std::cout << "TEST add delay processor" << std::endl;
  bool reversePol = true, mute = false;
  audioPipeDef->addGainProcessor(firstInputParam.name, 2., reversePol, mute);
  ENSURE_EQ(2., audioPipeDef->getGain(firstInputParam.name).value(), "Check gain control value");
  ENSURE_EQ(mute, audioPipeDef->getMute(firstInputParam.name).value(), "Check mute control value");
  ENSURE_EQ(reversePol, audioPipeDef->getReversePolarity(firstInputParam.name).value(), "Check mute control value");

  audioPipeDef->addGainProcessor(firstInputParam.name, -2., reversePol, mute);
  audioPipeDef->setGain(firstInputParam.name, -20.);
  audioPipeDef->setReversePolarity(firstInputParam.name, !reversePol);
  audioPipeDef->setMute(firstInputParam.name, !mute);
  ENSURE_EQ(-20., audioPipeDef->getGain(firstInputParam.name).value(), "Check gain control value");
  ENSURE_EQ(!mute, audioPipeDef->getMute(firstInputParam.name).value(), "Check mute control value");
  ENSURE_EQ(!reversePol, audioPipeDef->getReversePolarity(firstInputParam.name).value(), "Check mute control value");

  PanoDefinition *panoDef = getDefaultPano();
  Potential<AudioPipeline> audioPipe = AudioPipeFactory::create(*audioPipeDef, *panoDef);
  ENSURE(audioPipe.ok(), "Check creation of the audio pipe");

  // Check removing a processor
  std::cout << "TEST remove delay processor" << std::endl;
  ENSURE(audioPipeDef->removeProcessor(kDelayProcessorName).ok(), "Check remove delay processor");
  ENSURE(!audioPipeDef->getProcessor(kDelayProcessorName).ok(), "Check remove delay processor");

  std::cout << "TEST remove gain processor" << std::endl;
  ENSURE(audioPipeDef->removeProcessor(kGainProcessorName).ok(), "Check remove gain processor");
  ENSURE(!audioPipeDef->getProcessor(kGainProcessorName).ok(), "Check remove gain processor");

  delete audioPipeDef;
  delete panoDef;
  return;
}

void testReadAudioPipeDef() {
  std::string inputFile = "data/audio_pipe.json";
  Potential<Ptv::Parser> parser(Ptv::Parser::create());
  ENSURE(parser->parse(inputFile));
  std::unique_ptr<Ptv::Value> ptv(parser->getRoot().clone());
  std::unique_ptr<Ptv::Value> audioPipeConfig(ptv->has("audio_pipe")->clone());

  // Test the create from a ptv config
  std::unique_ptr<AudioPipeDefinition> audioPipeDefOrigin(AudioPipeDefinition::create(*audioPipeConfig));

  // Test the clone method
  std::unique_ptr<AudioPipeDefinition> audioPipeDef(audioPipeDefOrigin->clone());

  ENSURE_EQ(audioPipeDef->getSamplingRate(), 44100, "default sampling rate");
  ENSURE_EQ(audioPipeDef->getBlockSize(), 512, "default internal block size");
  ENSURE_EQ(audioPipeDef->numAudioInputs(), 1, "number of audio inputs");
  std::vector<size_t> expectedNumSources = {2};
  for (audioreaderid_t i = 0; i < audioPipeDef->numAudioInputs(); i++) {
    AudioInputDefinition *inputDef = audioPipeDef->getInput(i);
    std::stringstream ss;
    ss << "input " << i << ": number of sources" << inputDef->numSources() << "!=" << expectedNumSources[i];
    ENSURE_EQ(inputDef->numSources(), expectedNumSources[i], ss.str().c_str());
    std::vector<int> expectedChannel = {0, 1};
    if (i == 0) {
      ENSURE_EQ(true, inputDef->getIsMaster(), "Check master parameter");
      ENSURE_EQ(std::string("stereo"), inputDef->getLayout(), "Check layout parameter");
    }
    for (size_t j = 0; j < inputDef->numSources(); j++) {
      AudioSourceDefinition *sourceDef = inputDef->getSource(j);
      ENSURE_EQ(sourceDef->getReaderId(), (readerid_t)0, "Reader id of the source");
      ENSURE_EQ(sourceDef->getChannel(), (size_t)expectedChannel[j], "Reader id of the source");
    }
  }
  ENSURE_EQ(static_cast<size_t>(3), audioPipeDef->numProcessors(), "number of audio processor");

  // First audio processor should be audio delay
  std::string expectedProcName = Core::kDelayProcessorName;
  std::string procName = audioPipeDef->getProcessor(0)->getName();
  ENSURE_EQ(expectedProcName, procName, std::string("name of the audio processor " + expectedProcName).c_str());

  // Second audio processor should be audio gain
  expectedProcName = Core::kGainProcessorName;
  procName = audioPipeDef->getProcessor(1)->getName();
  ENSURE_EQ(expectedProcName, procName, std::string("name of the audio processor " + expectedProcName).c_str());

  // Third audio processor should be audio ambRotator
  expectedProcName = Core::kAmbRotateProcessorName;
  procName = audioPipeDef->getProcessor(2)->getName();
  ENSURE_EQ(expectedProcName, procName, std::string("name of the audio processor " + expectedProcName).c_str());

  std::unique_ptr<Ptv::Value> pipeSerialized(audioPipeDef->serialize());
  std::ofstream os;
  std::string testData = getDataFolder();
  os.open(testData + "/toto.json", std::ios::out);
  pipeSerialized->printJson(os);
  os.close();
}

void testUpdateAudioProcessorParams() {
  std::cout << "Test Update audio processors params" << std::endl;
  std::string inputFile = "data/audio_pipe.json";
  Potential<Ptv::Parser> parser(Ptv::Parser::create());
  ENSURE(parser->parse(inputFile));
  std::unique_ptr<Ptv::Value> ptv(parser->getRoot().clone());
  std::unique_ptr<Ptv::Value> audioPipeConfig(ptv->has("audio_pipe")->clone());

  std::unique_ptr<AudioPipeDefinition> AudioPipeDefBeforeSerialization(AudioPipeDefinition::create(*audioPipeConfig));
  std::unique_ptr<AudioPipeDefinition> audioPipeDef(
      reCreateAudiopipeAfterSerialization(AudioPipeDefBeforeSerialization.get()));
  std::unique_ptr<PanoDefinition> pano(getDefaultPano());
  Potential<AudioPipeline> audioPipeline(AudioPipeFactory::create(*audioPipeDef, *pano));

  // Change delay of the audio pipe definition
  audioPipeDef->setDelay("bbb", 3.0);
  PotentialValue<double> delay = audioPipeDef->getDelay("bbb");
  ENSURE(delay.ok(), "check getDelay message");
  ENSURE_EQ(3.0, delay.value(), "Check delay value");

  // Change gain parameters of the audio pipe definition
  audioPipeDef->setGain("bbb", -10.0);
  audioPipeDef->setMute("bbb", false);
  audioPipeDef->setReversePolarity("bbb", true);
  PotentialValue<double> gain = audioPipeDef->getGain("bbb");
  PotentialValue<bool> mute = audioPipeDef->getMute("bbb");
  PotentialValue<bool> reversePolarity = audioPipeDef->getReversePolarity("bbb");
  ENSURE(gain.ok(), "check getGain message");
  ENSURE_EQ(-10., gain.value(), "Check gain value");
  ENSURE(mute.ok(), "check getMute message");
  ENSURE_EQ(false, mute.value(), "Check mute value");
  ENSURE(reversePolarity.ok(), "check getReversePolarity message");
  ENSURE_EQ(true, reversePolarity.value(), "Check reverse polarity value");

  // update audio processors param of the pipeline
  audioPipeline->applyProcessorParam(*audioPipeDef);
  PotentialValue<AudioObject *> audioObject = audioPipeline->getAudioProcessor("bbb", "delay");
  double delay_s = static_cast<SampleDelay *>(audioObject.value())->getDelaySeconds();
  ENSURE_EQ(3.0, delay_s, "Check delay of the pipeline");

  audioObject = audioPipeline->getAudioProcessor("bbb", "gain");
  double gaindB = static_cast<Gain *>(audioObject.value())->getGainDB();
  bool mutePipe = static_cast<Gain *>(audioObject.value())->getMute();
  bool rpPipe = static_cast<Gain *>(audioObject.value())->getReversePolarity();
  ENSURE_EQ(-10., gaindB, "Check gain of the pipeline");
  ENSURE_EQ(false, mutePipe, "Check gain of the pipeline");
  ENSURE_EQ(true, rpPipe, "Check gain of the pipeline");

  // Try to set/get the delay of a non existing input
  ENSURE(!audioPipeDef->setDelay("xxx", 3.0).ok(), "check that setting delay of non existing input fails");
  ENSURE(!audioPipeDef->getDelay("xxx").ok(), "check that getting delay of non existing input fails");
  // Try to set/get the gain of a non existing input
  ENSURE(!audioPipeDef->setGain("xxx", 3.0).ok(), "check that setting gain of non existing input fails");
  ENSURE(!audioPipeDef->getGain("xxx").ok(), "check that getting gain of non existing input fails");
  // Try to set/get the mute of a non existing input
  ENSURE(!audioPipeDef->setMute("xxx", true).ok(), "check that setting mute of non existing input fails");
  ENSURE(!audioPipeDef->getMute("xxx").ok(), "check that getting mute of non existing input fails");
  // Try to set/get the reverse polarity of a non existing input
  ENSURE(!audioPipeDef->setReversePolarity("xxx", true).ok(),
         "check that setting reverse polarity of non existing input fails");
  ENSURE(!audioPipeDef->getReversePolarity("xxx").ok(),
         "check that getting reverse polarity of non existing input fails");

  // Check ambrotator param
  PotentialValue<Audio::AudioObject *> ambRotator =
      audioPipeline->getAudioProcessor("bbb", Core::kAmbRotateProcessorName);
  Vector3<double> offsetRotation = static_cast<Audio::AmbRotator *>(ambRotator.value())->getRotationOffset();
  ENSURE_EQ(3.14, offsetRotation(0), "Check yaw offset");
  ENSURE_EQ(1.57, offsetRotation(1), "Check pitch offset");
  ENSURE_EQ(0., offsetRotation(2), "Check roll offset");
}

void testCreateAudioPipeDefFromPano() {
  std::unique_ptr<PanoDefinition> panoDef(getDefaultPano());
  std::unique_ptr<AudioPipeDefinition> audioPipeDefBeforeSerialization(
      AudioPipeDefinition::createAudioPipeFromPanoInputs(panoDef.get()));
  ENSURE(audioPipeDefBeforeSerialization->readersAreConsistent(panoDef.get()),
         "Audio pipe and pano def should be consistent");
  std::unique_ptr<AudioPipeDefinition> audioPipeDef(
      reCreateAudiopipeAfterSerialization(audioPipeDefBeforeSerialization.get()));
  // VSA-7376
  ENSURE(panoDef.get()->numAudioInputs() > 0);
  InputDefinition *lastInput = panoDef.get()->popInput(panoDef.get()->numInputs() - 1);
  ENSURE(!audioPipeDef->readersAreConsistent(panoDef.get()), "Audio pipe and pano def should not be consistent");
  panoDef.get()->insertInput(lastInput, -1);
  ENSURE(audioPipeDef->readersAreConsistent(panoDef.get()), "Audio pipe and pano def should be consistent");

  ENSURE_EQ(panoDef->numAudioInputs(), audioPipeDef->numAudioInputs(),
            "Check number of audio inputs of the pano and the audio pipe");
  ENSURE_EQ(3, audioPipeDef->numAudioInputs(), "Check number of audio inputs of the audio pipe");
  ENSURE_EQ((size_t)audioPipeDef->numAudioInputs(), audioPipeDef->numAudioMixes(),
            "Check default number of audio mixes of the audio pipe");

  // Check channel layout of the audio inputs
  ENSURE_EQ(std::string(getStringFromChannelLayout(STEREO)), audioPipeDef->getInput(0)->getLayout(),
            "Check channel layout of the first input");
  ENSURE_EQ(std::string(getStringFromChannelLayout(MONO)), audioPipeDef->getInput(1)->getLayout(),
            "Check channel layout of the second input");
  ENSURE_EQ(std::string(getStringFromChannelLayout(MONO)), audioPipeDef->getInput(2)->getLayout(),
            "Check channel layout of the third input");

  // Change pano definition
  Ptv::Value *readerConfig(panoDef->getAudioInput(0).getReaderConfig().clone());
  readerConfig->get("audio_channels")->asInt() = readerConfig->has("audio_channels")->asInt() - 1;
  panoDef->getAudioInput(0).setReaderConfig(readerConfig);
  ENSURE(!audioPipeDef->readersAreConsistent(panoDef.get()), "Audio pipe and pano def should not be consistent");
}

void checkAudioInputLayout(ChannelLayout layoutToTest) {
  std::cout << "TEST make audio " << getStringFromChannelLayout(layoutToTest) << " input " << std::endl;
  std::unique_ptr<AudioPipeDefinition> audioPipeDef(AudioPipeDefinition::createDefault());
  AudioInputDefinition *inDef(AudioInputDefinition::create(InputParam("blabla", 0, layoutToTest)));
  inDef->setLayout(getStringFromChannelLayout(layoutToTest));
  audioPipeDef->addInput(inDef);
  audioPipeDef->addGainProcessor(inDef->getName(), 0, false, false);
  std::string testData = getDataFolder();
  audioPipeDef->setDebugFolder(testData);

  std::unique_ptr<PanoDefinition> pano(getDefaultPano());
  Potential<AudioPipeline> audioPipeline(AudioPipeFactory::create(*audioPipeDef, *pano));

  AudioBlock blk(getAChannelLayoutFromNbChannels(getNbChannelsFromChannelLayout(layoutToTest)), 0);
  audioBlockReaderMap_t samplesPerRd;
  audioBlockGroupMap_t samplesPerGr;

  blk.resize(3);
  double x = 0;
  for (auto &track : blk) {
    track[0] = 0. + x;
    track[1] = 0.1 + x;
    track[2] = 0.2 + x;
    x = x + 0.01;
  }
  // Copy the block as it will be moved by the audio pipeline
  AudioBlock copyBlk = blk.clone();

  samplesPerRd[0] = std::move(blk);
  samplesPerGr[0] = std::move(samplesPerRd);
  audioPipeline->process(samplesPerGr);

  // Call desctructor of the audiopipeline to force the writing of debug files
  delete audioPipeline.release();
  std::string wavFile(audioPipeDef->getDebugFolder() + "/" + inDef->getName() + "input.wav");
  WavReader wvRd(wavFile);
  ENSURE_EQ((int)inDef->numSources(), wvRd.getChannels(), "Check number of channels produced by the audio pipe");
  AudioBlock out;
  wvRd.step(out);
  ENSURE_EQ(copyBlk.getLayout(), out.getLayout(), "Check layout");
  ENSURE_EQ(copyBlk.numSamples(), out.numSamples(), "Check numSamples");
  for (const auto &outT : out) {
    for (size_t i = 0; i < outT.size(); i++) {
      ENSURE_APPROX_EQ(copyBlk[outT.channel()][i], outT[i], 1e-4, "Check sample value");
    }
  }
  std::cout << "TEST make audio " << getStringFromChannelLayout(layoutToTest) << " input: PASSED " << std::endl;
}

void testAudioInputs() {
  std::vector<ChannelLayout> layoutToTest({MONO, STEREO, AMBISONICS_WXYZ, _5POINT1});
  for (const auto &layout : layoutToTest) {
    checkAudioInputLayout(layout);
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  VideoStitch::Testing::testReadAudioPipeDef();
  VideoStitch::Testing::testCreateAudioPipeDef();
  VideoStitch::Testing::testCreateAudioPipeDefFromPano();
  VideoStitch::Testing::testAudioInputs();
}
