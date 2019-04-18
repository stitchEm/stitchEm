// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "audio/sigGen.hpp"

#include "libvideostitch/ambDecoderDef.hpp"
#include "libvideostitch/ambisonic.hpp"
#include "libvideostitch/audioWav.hpp"
#include "libvideostitch/parse.hpp"

#include "parse/json.hpp"
#include "gpu/testing.hpp"

#include <fstream>

namespace VideoStitch {
namespace Testing {
using namespace Audio;

static channelCoefTable_t stereoTable;
static channelCoefTable_t five1Table;
const std::string fumaPresets = "data/ambisonic-fuma-decoding.preset";

void initDecodeTablesWithFumaCoef() {
  // These coefficients follow the C-Sound ambisonic decoder:
  // http://csounds.com/resources/Bformatdec.csd
  // They could be changed for other decoders
  // Stereo Coefficients
  stereoTable[SPEAKER_FRONT_LEFT][SPEAKER_AMB_W] = 0.7071;
  stereoTable[SPEAKER_FRONT_LEFT][SPEAKER_AMB_X] = 0.;
  stereoTable[SPEAKER_FRONT_LEFT][SPEAKER_AMB_Y] = 0.5;
  stereoTable[SPEAKER_FRONT_LEFT][SPEAKER_AMB_Z] = 0.;

  stereoTable[SPEAKER_FRONT_RIGHT][SPEAKER_AMB_W] = 0.7071;
  stereoTable[SPEAKER_FRONT_RIGHT][SPEAKER_AMB_X] = 0.;
  stereoTable[SPEAKER_FRONT_RIGHT][SPEAKER_AMB_Y] = -0.5;
  stereoTable[SPEAKER_FRONT_RIGHT][SPEAKER_AMB_Z] = 0.;

  // 5.1 Coefficients
  five1Table[SPEAKER_FRONT_LEFT][SPEAKER_AMB_W] = 0.1690;
  five1Table[SPEAKER_FRONT_LEFT][SPEAKER_AMB_X] = 0.0797;
  five1Table[SPEAKER_FRONT_LEFT][SPEAKER_AMB_Y] = 0.0891;
  five1Table[SPEAKER_FRONT_LEFT][SPEAKER_AMB_Z] = 0.;

  five1Table[SPEAKER_FRONT_CENTER][SPEAKER_AMB_W] = 0.1635;
  five1Table[SPEAKER_FRONT_CENTER][SPEAKER_AMB_X] = 0.0923;
  five1Table[SPEAKER_FRONT_CENTER][SPEAKER_AMB_Y] = 0.;
  five1Table[SPEAKER_FRONT_CENTER][SPEAKER_AMB_Z] = 0.;

  five1Table[SPEAKER_FRONT_RIGHT][SPEAKER_AMB_W] = 0.1690;
  five1Table[SPEAKER_FRONT_RIGHT][SPEAKER_AMB_X] = 0.0797;
  five1Table[SPEAKER_FRONT_RIGHT][SPEAKER_AMB_Y] = -0.0891;
  five1Table[SPEAKER_FRONT_RIGHT][SPEAKER_AMB_Z] = 0.;

  five1Table[SPEAKER_LOW_FREQUENCY][SPEAKER_AMB_W] = 1.;
  five1Table[SPEAKER_LOW_FREQUENCY][SPEAKER_AMB_X] = 0.;
  five1Table[SPEAKER_LOW_FREQUENCY][SPEAKER_AMB_Y] = 0.;
  five1Table[SPEAKER_LOW_FREQUENCY][SPEAKER_AMB_Z] = 0.;

  five1Table[SPEAKER_SIDE_LEFT][SPEAKER_AMB_W] = 0.4563;
  five1Table[SPEAKER_SIDE_LEFT][SPEAKER_AMB_X] = -0.1259;
  five1Table[SPEAKER_SIDE_LEFT][SPEAKER_AMB_Y] = 0.1543;
  five1Table[SPEAKER_SIDE_LEFT][SPEAKER_AMB_Z] = 0.;

  five1Table[SPEAKER_SIDE_RIGHT][SPEAKER_AMB_W] = 0.4563;
  five1Table[SPEAKER_SIDE_RIGHT][SPEAKER_AMB_X] = -0.1259;
  five1Table[SPEAKER_SIDE_RIGHT][SPEAKER_AMB_Y] = -0.1543;
  five1Table[SPEAKER_SIDE_RIGHT][SPEAKER_AMB_Z] = 0.;
}

void checkAmbDecoderCoef(channelCoefTable_t table, ChannelLayout layout) {
  ChannelMap c = SPEAKER_FRONT_LEFT;
  while (c < NO_SPEAKER) {
    ChannelMap ac = SPEAKER_AMB_W;
    while (ac < NO_SPEAKER && (ac & AMBISONICS_WXYZ)) {
      std::stringstream ss;
      ss << "Check " << getStringFromChannelLayout(layout) << " " << getStringFromChannelType(c) << " "
         << getStringFromChannelType(ac);
      if (layout == STEREO && c & layout) {
        ENSURE_EQ(stereoTable.at(c).at(ac), table.at(c).at(ac), ss.str().c_str());
      } else if (layout == _5POINT1 && c & layout) {
        ENSURE_EQ(five1Table.at(c).at(ac), table.at(c).at(ac), ss.str().c_str());
      }
      ac = static_cast<ChannelMap>(static_cast<int64_t>(ac) << 1);
    }
    c = static_cast<ChannelMap>(static_cast<int64_t>(c) << 1);
  }
}

void check51OutputChannel(AudioBlock &output, ChannelMap m) {
  std::stringstream ss;
  ss << "Check sample of " << getStringFromChannelType(m);
  ENSURE_APPROX_EQ(five1Table[m][SPEAKER_AMB_W], output[m][0], 1e-5, ss.str().c_str());
  ENSURE_APPROX_EQ(five1Table[m][SPEAKER_AMB_X], output[m][1], 1e-5, ss.str().c_str());
  ENSURE_APPROX_EQ(five1Table[m][SPEAKER_AMB_Y], output[m][2], 1e-5, ss.str().c_str());
  ENSURE_APPROX_EQ(five1Table[m][SPEAKER_AMB_Z], output[m][3], 1e-5, ss.str().c_str());
  double sum = five1Table[m][SPEAKER_AMB_W] + five1Table[m][SPEAKER_AMB_X] + five1Table[m][SPEAKER_AMB_Y] +
               five1Table[m][SPEAKER_AMB_Z];
  ENSURE_APPROX_EQ(sum, output[m][4], 1e-5, ss.str().c_str());
}

void checkStereoOutputChannel(AudioBlock &output, ChannelMap m) {
  std::stringstream ss;
  ss << "Check sample of " << getStringFromChannelType(m);
  // decoding a sample S is given by:
  // S = alpha * W + beta * X + gamma * Y + delta * Z
  ENSURE_APPROX_EQ(stereoTable[m][SPEAKER_AMB_W], output[m][0], 1e-5, ss.str().c_str());  // S(0) = W
  ENSURE_APPROX_EQ(stereoTable[m][SPEAKER_AMB_W] + stereoTable[m][SPEAKER_AMB_Y], 1e-5, output[m][1],
                   ss.str().c_str());  // S(1) = alpha * W + gamma * Y
  ENSURE_APPROX_EQ(stereoTable[m][SPEAKER_AMB_Y], output[m][2], 1e-5, ss.str().c_str());  // S(2) = gamma * Y
}

void checkProcessStereoDecoding(const AmbisonicDecoderDef &decoderDef) {
  AmbDecoder ambDecoder(STEREO, decoderDef.getCoefficients());
  AudioBlock input(AMBISONICS_WXYZ), output;
  input.assign(3, 0.);
  input[SPEAKER_AMB_W][0] = 1.;
  input[SPEAKER_AMB_W][1] = 1.;
  input[SPEAKER_AMB_W][2] = 0.;
  input[SPEAKER_AMB_Y][0] = 0.;
  input[SPEAKER_AMB_Y][1] = 1.;
  input[SPEAKER_AMB_Y][2] = 1.;
  ambDecoder.step(output, input);
  ENSURE_EQ(STEREO, output.getLayout(), "Check output layout.");
  ENSURE_EQ(input.numSamples(), output.numSamples(), "Check output number of samples.");
  // Check Front left samples
  checkStereoOutputChannel(output, SPEAKER_FRONT_LEFT);
  checkStereoOutputChannel(output, SPEAKER_FRONT_RIGHT);
}

void checkProcess51Decoding(const AmbisonicDecoderDef &decoderDef) {
  AmbDecoder ambDecoder(_5POINT1, decoderDef.getCoefficients());
  AudioBlock input(AMBISONICS_WXYZ), output;
  input.assign(5, 0.);
  // Initialize input like this
  // 1-0-0-0
  // 0-1-0-0
  // 0-0-1-0
  // 0-0-0-1
  // 1-1-1-1
  input[SPEAKER_AMB_W][0] = 1.;
  input[SPEAKER_AMB_X][1] = 1.;
  input[SPEAKER_AMB_Y][2] = 1.;
  input[SPEAKER_AMB_Z][3] = 1.;
  input[SPEAKER_AMB_W][4] = 1.;
  input[SPEAKER_AMB_X][4] = 1.;
  input[SPEAKER_AMB_Y][4] = 1.;
  input[SPEAKER_AMB_Z][4] = 1.;
  ambDecoder.step(output, input);
  ENSURE_EQ(_5POINT1, output.getLayout(), "Check output layout.");
  ENSURE_EQ(input.numSamples(), output.numSamples(), "Check output number of samples.");

  check51OutputChannel(output, SPEAKER_FRONT_LEFT);
  check51OutputChannel(output, SPEAKER_FRONT_RIGHT);
}

void testAmbDecoder() {
  Potential<VideoStitch::Ptv::Parser> parser(Ptv::Parser::create());
  ENSURE(parser->parse(fumaPresets));
  std::unique_ptr<Ptv::Value> ptv(parser->getRoot().clone());
  AmbisonicDecoderDef decoderDefOriginal(*ptv);

  // Test serialization and clone
  std::unique_ptr<AmbisonicDecoderDef> decoderCloned(decoderDefOriginal.clone());
  std::unique_ptr<Ptv::Value> decoderSerialized(decoderCloned->serialize());
  std::string testData = getDataFolder();
  std::string serializedFile = testData + "/toto.preset";
  std::ofstream serialized;
  serialized.open(serializedFile, std::ios_base::out);
  decoderSerialized->printJson(serialized);
  serialized.close();

  // Reopen the ambisonic decoder serialized
  ENSURE(parser->parse(serializedFile));
  std::unique_ptr<Ptv::Value> ptvAfterSerialization(parser->getRoot().clone());
  AmbisonicDecoderDef decoderDef(*ptvAfterSerialization);

  channelCoefTable_t stereoCoef = decoderDef.getCoefficientsByLayout(STEREO).value();
  ENSURE_EQ((size_t)2, stereoCoef.size(), "Check size of stereo coef");
  checkAmbDecoderCoef(stereoCoef, STEREO);
  checkProcessStereoDecoding(decoderDef);

  channelCoefTable_t five1Coef = decoderDef.getCoefficientsByLayout(_5POINT1).value();
  ENSURE_EQ((size_t)6, five1Coef.size(), "Check size of stereo coef");
  checkAmbDecoderCoef(five1Coef, _5POINT1);
  checkProcess51Decoding(decoderDef);
}

void compareAudioBlocks(const AudioBlock &a, const AudioBlock &b) {
  ENSURE_EQ(a.getLayout(), b.getLayout(), "Check layout");
  ENSURE_EQ(a.numSamples(), b.numSamples(), "Check nb of samples");

  for (const auto &aTrack : a) {
    for (int iSample = 0; iSample < (int)a.numSamples(); iSample++) {
      std::string msg = "Check sample " + std::to_string(iSample);
      ENSURE_APPROX_EQ(aTrack[iSample], b[aTrack.channel()][iSample], 1e-5, msg.c_str());
    }
  }
}

void testAmbisonicPipeline(bool inPlace = false, int nbBlockToProcess = 100, ChannelLayout layoutToTest = STEREO,
                           AmbisonicNorm ambNorm = AmbisonicNorm::FUMA, bool checkSamples = true,
                           bool writeWavFiles = false) {
  std::cout << "Test ambisonic encoder/decoder for " << getStringFromChannelLayout(layoutToTest) << " encoder norm "
            << getStringFromAmbisonicNorm(ambNorm) << " check samples " << checkSamples << " write wave files "
            << writeWavFiles << std::endl;

  // Genrate signal
  std::vector<double> freqs;
  for (int iFreq = 0; iFreq < getNbChannelsFromChannelLayout(layoutToTest); iFreq++) {
    freqs.push_back((iFreq + 1.) * 110.);
  }
  SigGenSine stereoSineGen(freqs, getDefaultSamplingRate(), 1.0);
  AmbEncoder ambEnc(AmbisonicOrder::FIRST_ORDER, ambNorm);

  // Create decoder
  Potential<VideoStitch::Ptv::Parser> parser(Ptv::Parser::create());
  ENSURE(parser->parse(fumaPresets));
  std::unique_ptr<Ptv::Value> ptv(parser->getRoot().clone());
  AmbisonicDecoderDef decoderDef(*ptv);
  AmbDecoder ambDec(layoutToTest, decoderDef.getCoefficients());
  AudioBlock input(layoutToTest, 0), outEncoder, outDecoder, saveInput;
  input.resize(getDefaultBlockSize());

  // wav writers
  std::string testData = getDataFolder();
  WavWriter inWriter(testData + "/inAmb.wav", layoutToTest, getDefaultSamplingRate());
  WavWriter outEncWriter(testData + "/outEnc.wav", AMBISONICS_WXYZ, getDefaultSamplingRate());
  WavWriter outDecWriter(testData + "/outDec.wav", layoutToTest, getDefaultSamplingRate());
  int iBlk = 0;

  if (!inPlace) {
    // Check not in place method
    while (iBlk < nbBlockToProcess) {
      stereoSineGen.step(input);
      ENSURE_EQ(layoutToTest, input.getLayout(), "Check input layout");
      if (SPEAKER_LOW_FREQUENCY & layoutToTest) {
        input[SPEAKER_LOW_FREQUENCY].assign(getDefaultBlockSize(), 0.);
      }
      ambEnc.step(outEncoder, input);
      ENSURE_EQ(AMBISONICS_WXYZ, outEncoder.getLayout(), "Check layout after the encoder");
      ambDec.step(outDecoder, outEncoder);
      ENSURE_EQ(layoutToTest, outDecoder.getLayout(), "Check layout after the decoder");
      if (checkSamples) {
        compareAudioBlocks(input, outDecoder);
      }

      if (writeWavFiles) {
        inWriter.step(input);
        outEncWriter.step(outEncoder);
        outDecWriter.step(outDecoder);
      }
      iBlk++;
    }
  } else {
    // Check in place method
    iBlk = 0;
    while (iBlk < nbBlockToProcess) {
      // Gen signal
      stereoSineGen.step(input);
      ENSURE_EQ(layoutToTest, input.getLayout(), "Check input layout");
      if (writeWavFiles) {
        inWriter.step(input);
      }
      saveInput = input.clone();

      // Encode in B-format
      ambEnc.step(input);
      ENSURE_EQ(AMBISONICS_WXYZ, input.getLayout(), "Check layout after the encoder");
      if (writeWavFiles) {
        outEncWriter.step(input);
      }

      // Decode in B-format
      ambDec.step(input);
      ENSURE_EQ(layoutToTest, input.getLayout(), "Check layout after the decoder");
      if (writeWavFiles) {
        outEncWriter.step(input);
      }
      if (checkSamples) {
        compareAudioBlocks(input, saveInput);
      }
      iBlk++;
    }
  }

  if (writeWavFiles) {
    inWriter.close();
    outEncWriter.close();
    outDecWriter.close();
  }
}

}  // namespace Testing
}  // namespace VideoStitch

int main() {
  VideoStitch::Testing::initDecodeTablesWithFumaCoef();
  std::cout << "RUN Test Ambisonic Decoder" << std::endl;
  VideoStitch::Testing::testAmbDecoder();
  std::cout << "RUN Test Ambisonic Decoder PASSED" << std::endl;

  std::cout << "RUN Test Ambisonic Pipeline" << std::endl;
  VideoStitch::Testing::testAmbisonicPipeline(false, 100);
  VideoStitch::Testing::testAmbisonicPipeline(true, 100);
  // TODO find good decoding coefficients for Stereo SN3D normalization
  // VideoStitch::Testing::testAmbisonicPipeline(false, 100, VideoStitch::Audio::STEREO,
  // VideoStitch::Audio::AmbisonicNorm::SN3D, true);
  VideoStitch::Testing::testAmbisonicPipeline(false, 100, VideoStitch::Audio::_5POINT1,
                                              VideoStitch::Audio::AmbisonicNorm::FUMA, false);
  // TODO find good encoding and decoding coef for the 5.1 layout for FUMA and SN3D normalization
  std::cout << "RUN Test Ambisonic Pipeline PASSED" << std::endl;

  return 0;
}
