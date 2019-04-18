// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// Audio unit test for the following test cases:
// - wavReader and wavWriter to read and write wave files
// - resampler
// - conversion Audio::Samples to AudioBlock
// - conversion Audio::Block to Audio::Samples

#include "gpu/testing.hpp"
#include "common/audioUtils.hpp"

#include "libvideostitch/ambisonic.hpp"
#include "libvideostitch/audio.hpp"
#include "libvideostitch/audioObject.hpp"
#include "libvideostitch/audioWav.hpp"
#include "audio/envelopeDetector.hpp"
#include "audio/sigGen.hpp"
#include "audio/resampler.hpp"
#include "audio/summer.hpp"

namespace VideoStitch {
namespace Testing {
using namespace Audio;

void cosineTest() {
  double freq = 440., rate = 48000., amp = 0.8, duration = 5.5 / freq;
  SigGenSine::SigGenSine1Dim sigCos(freq, rate, amp);

  // Generate signal
  size_t nSamples = static_cast<size_t>(rate * duration);
  size_t nchannels = 1;
  AudioTrack track(SPEAKER_FRONT_LEFT);
  AudioTrack track2(SPEAKER_FRONT_LEFT);
  for (size_t s = 0; s < nSamples / 2; s++) {
    track.push_back(0);
    track2.push_back(0);
  }

  // Generate 1 sec of 440 Hz cosine
  sigCos.step(track);
  // generate 1 sec of 880 Hz cosine
  sigCos.setFrequency(freq * 2.0);
  sigCos.step(track2);
  for (size_t s = 0; s < nSamples / 2; s++) {
    track.push_back(track2[s]);
  }

  // compare signal generated to the reference signal
  WavReader refFile("data/snd/cos48k_mono.wav");
  AudioBlock refbuf(MONO);
  refFile.step(refbuf);
  size_t nRefChannels = refbuf.size();
  size_t nbRefSamples = refbuf[SPEAKER_FRONT_LEFT].size();
  ENSURE_EQ(nRefChannels, nchannels, "Unexpected number of channels");
  ENSURE_EQ(nbRefSamples, nSamples, "Unexpected number of samples");
  double eps = 1. / 32267.;
  for (auto &refTrack : refbuf) {
    for (size_t j = 0; j < nbRefSamples; j++) {
      ENSURE_APPROX_EQ(refTrack[j], track[j], eps);
    }
  }
}

static const size_t nSamples = 3;

void testSamplesToAudioBlock() {
  AudioBlock block;

  float *raw[MAX_AUDIO_CHANNELS];
  raw[0] = new float[nSamples * sizeof(float)];
  raw[1] = new float[nSamples * sizeof(float)];
  float left[nSamples] = {-1, 0, 1};
  float right[nSamples] = {-0.5, 0, 0.5};
  for (size_t i = 0; i < nSamples; i++) {
    raw[0][i] = left[i];
    raw[1][i] = right[i];
  }

  Samples samples(SamplingRate::SR_48000, SamplingDepth::FLT_P, ChannelLayout::STEREO, 1234., (uint8_t **)raw,
                  nSamples);

  samples2AudioBlock(block, samples);

  double eps = 0.000001;
  ENSURE_EQ(block.getLayout(), samples.getChannelLayout());
  ENSURE_EQ(block.getTimestamp(), samples.getTimestamp());

  for (size_t s = 0; s < nSamples; s++) {
    ENSURE_APPROX_EQ(block[SPEAKER_FRONT_LEFT][s], (double)left[s], eps);
    ENSURE_APPROX_EQ(block[SPEAKER_FRONT_RIGHT][s], (double)right[s], eps);
  }
}

void testAudioBlkToSamples() {
  AudioBlock block(STEREO, 1234.);
  audioSample_t left[nSamples] = {-1, 0, 1};
  audioSample_t right[nSamples] = {-0.5, 0, 0.5};

  for (size_t s = 0; s < nSamples; s++) {
    block[SPEAKER_FRONT_LEFT].push_back(left[s]);
    block[SPEAKER_FRONT_RIGHT].push_back(right[s]);
  }

  Samples outSamples;
  audioBlock2Samples(outSamples, block);

  audioSample_t **outData = (audioSample_t **)outSamples.getSamples().data();

  audioSample_t eps = 0.000001;
  for (size_t s = 0; s < nSamples; s++) {
    ENSURE_APPROX_EQ(outData[0][s], left[s], eps);
    ENSURE_APPROX_EQ(outData[1][s], right[s], eps);
  }
}

void testDrop() {
  float *raw[MAX_AUDIO_CHANNELS];
  raw[0] = new float[nSamples * sizeof(float)];
  raw[1] = new float[nSamples * sizeof(float)];
  float left[nSamples] = {0.1f, 0.2f, 0.3f};
  float right[nSamples] = {-0.1f, -0.2f, -0.3f};
  for (size_t i = 0; i < nSamples; i++) {
    raw[0][i] = left[i];
    raw[1][i] = right[i];
  }

  Samples samples(SamplingRate::SR_48000, SamplingDepth::FLT_P, ChannelLayout::STEREO, 1234., (uint8_t **)raw,
                  nSamples);

  // test fail case
  if (samples.drop(4).ok()) {
    std::stringstream ss;
    ss << "TEST FAILED: drop more samples than available shouldn't pass without error" << std::endl;
    std::cerr << ss.str();
    std::raise(SIGABRT);
  }

  samples.drop(2);
  ENSURE_EQ(samples.getNbOfSamples(), nSamples - 2, "unexpected number of samples resulted");
  float **afterDrop = (float **)samples.getSamples().data();
  ENSURE_EQ((float)0.3, afterDrop[0][0], "sample left not expected");
  ENSURE_EQ((float)-0.3, afterDrop[1][0], "sample left not expected");

  samples.drop(1);
  ENSURE_EQ((int)samples.getNbOfSamples(), 0, "check number of samples left");
}

void testAppend() {
  float *raw[MAX_AUDIO_CHANNELS];
  raw[0] = new float[1 * sizeof(float)];
  raw[1] = new float[1 * sizeof(float)];
  float left[1] = {0.1f};
  float right[1] = {-0.1f};
  raw[0][0] = left[0];
  raw[1][0] = right[0];

  Samples samples(SamplingRate::SR_48000, SamplingDepth::FLT_P, ChannelLayout::STEREO, 1234., (uint8_t **)raw, 1);

  int nSamplesToAppend = 2;

  float *rawAppend[MAX_AUDIO_CHANNELS];
  rawAppend[0] = new float[nSamplesToAppend * sizeof(float)];
  rawAppend[1] = new float[nSamplesToAppend * sizeof(float)];
  float leftAppend[nSamples] = {0.2f, 0.3f};
  float rightAppend[nSamples] = {-0.2f, -0.3f};
  for (size_t i = 0; i < nSamples; i++) {
    rawAppend[0][i] = leftAppend[i];
    rawAppend[1][i] = rightAppend[i];
  }
  Samples samplesToAppend(SamplingRate::SR_48000, SamplingDepth::FLT_P, ChannelLayout::STEREO, 1234.,
                          (uint8_t **)rawAppend, nSamplesToAppend);

  samples.append(samplesToAppend);
  ENSURE_EQ(3, (int)samples.getNbOfSamples(), "unexpected number of samples resulted");
  float **afterAppend = (float **)samples.getSamples().data();
  ENSURE_EQ((float)0.1, afterAppend[0][0], "sample value not expected");
  ENSURE_EQ((float)-0.1, afterAppend[1][0], "sample value not expected");
  ENSURE_EQ((float)0.2, afterAppend[0][1], "sample value not expected");
  ENSURE_EQ((float)-0.2, afterAppend[1][1], "sample value not expected");
  ENSURE_EQ((float)0.3, afterAppend[0][2], "sample value not expected");
  ENSURE_EQ((float)-0.3, afterAppend[1][2], "sample value not expected");

  samplesToAppend.drop(samplesToAppend.getNbOfSamples());

  samples.append(samplesToAppend);
  ENSURE_EQ(3, (int)samples.getNbOfSamples(), "unexpected number of samples resulted");
}

void testSimpleEnvDetector() {
  ChannelLayout l = MONO;
  AudioBlock in(l);
  in.resize(10);
  for (int i = 0; i < 10; ++i) {
    in[SPEAKER_FRONT_LEFT][i] = i + 1;
  }

  VuMeter vm((int)getDefaultSamplingRate());
  vm.setSmoothing(3);

  for (int i = 0; i < 1; i++) {
    vm.step(in);
    std::vector<double> peaks = vm.getPeakValues();
    std::vector<double> rms = vm.getRmsValues();
    std::cout << "Peak = " << peaks[0] << std::endl;
    std::cout << "Rms = " << rms[0] << std::endl;
  }
}

void checkTimeConstants(const AudioBlock &env, double attack, double release) {
  double fs = getDefaultSamplingRate();
  // Check attack time constant : https://en.wikipedia.org/wiki/Time_constant
  int i = 0;
  for (audioSample_t s : *env.begin()) {
    if (s > 0.632) {
      break;
    }
    i++;
  }
  double effectiveActtackTime = static_cast<double>(i) / fs - 0.5;
  ENSURE_APPROX_EQ(attack, effectiveActtackTime, 0.02);

  // Check peak release time constant : https://en.wikipedia.org/wiki/Time_constant
  int j;
  for (j = static_cast<int>(fs); j < static_cast<int>(env.begin()->size()); j++) {
    if (env.begin()->at(j) < 0.368) {
      break;
    }
  }
  double effectiveReleaseTime = static_cast<double>(j) / fs - 1.0;
  ENSURE_APPROX_EQ(release, effectiveReleaseTime, 0.02);
}

void testEnvDetector() {
  ChannelLayout l = STEREO;
  AudioBlock in(l);
  double fs = getDefaultSamplingRate();
  SigGenSquare squareGen(fs, 1., 1., l);
  in.resize((size_t)fs * 2);
  squareGen.step(in);

  VuMeter vm(static_cast<int>(getDefaultSamplingRate()));
  vm.setDebug(true);
  vm.setPeakAttack(0.1);
  vm.setPeakRelease(0.2);
  vm.setRmsAttack(0.2);
  vm.setRmsRelease(0.3);
  vm.step(in);
  std::vector<double> peaks = vm.getPeakValues();
  std::vector<double> rms = vm.getRmsValues();
  ENSURE_EQ(2, (int)peaks.size(), "Check output peaks data size");
  ENSURE_EQ(2, (int)rms.size(), "Check output rms data size");
  checkTimeConstants(vm.getPeakEnvelope(), vm.getPeakAttack(), vm.getPeakRelease());
  checkTimeConstants(vm.getRmsEnvelope(), vm.getRmsAttack(), vm.getRmsRelease());
}

void testAudioSum() {
  // Check normal case
  std::vector<AudioBlock> blocks;
  AudioBlock output;
  blocks.emplace_back(STEREO, 0);
  blocks.emplace_back(STEREO, 0);
  std::vector<audioSample_t> values = {0, 1, 2};
  for (AudioBlock &block : blocks) {
    block.resize(3);
    for (AudioTrack &track : block) {
      int i = 0;
      for (audioSample_t &x : track) {
        x = values[i];
        i++;
      }
    }
  }
  ENSURE(sum(blocks, output).ok());
  ENSURE(output.getLayout() == STEREO);
  for (const AudioTrack &otr : output) {
    ENSURE_EQ(0., otr[0], "Check += first value.");
    ENSURE_EQ(2., otr[1], "Check += second value.");
    ENSURE_EQ(4., otr[2], "Check += third value.");
  }

  // Check + operator
  output.assign(3, 0.);
  output.setChannelLayout(STEREO);
  output = blocks[0] + blocks[1];
  for (const AudioTrack &otr : output) {
    ENSURE_EQ(0., otr[0], "Check + first value.");
    ENSURE_EQ(2., otr[1], "Check + second value.");
    ENSURE_EQ(4., otr[2], "Check + third value.");
  }

  // Check failure cases if one block has not the same length
  blocks[0].resize(4);
  ENSURE(!sum(blocks, output).ok());

  // put back to normal
  blocks[0].resize(3);
  ENSURE(sum(blocks, output).ok());

  // if one block has a different layout
  blocks[1].setChannelLayout(_2_1);
  ENSURE(!sum(blocks, output).ok());
}

void checkAmbisonicTrack(const AudioTrack &in, const AudioTrack &out, const double mul) {
  ENSURE(((in.channel() | AMBISONICS_3RD) != 0), "channel map should be ambisonic");
  ENSURE_EQ(in.size(), out.size(), "Check output size");
  for (size_t i = 0; i < in.size(); ++i) {
    std::stringstream ss;
    ss << "Check samples of audio track " << getStringFromChannelType(out.channel()) << " sample " << i;
    ENSURE_APPROX_EQ(mul * in[i], out[i], 0.000001, ss.str().c_str());
  }
}

int getAmbChannelIndexFromChannelMap(ChannelMap m) {
  return getChannelIndexFromChannelMap(m) - getChannelIndexFromChannelMap(SPEAKER_AMB_W);
}

void checkAmbisonicOutput(const AudioBlock &inputMono, const AudioBlock &output,
                          const std::map<ChannelMap, double> &res, AmbisonicOrder order) {
  ENSURE_EQ(MONO, inputMono.getLayout(), "Channel input map should be mono.");

  ChannelLayout expectedLayout;
  switch (order) {
    case (AmbisonicOrder::FIRST_ORDER):
      expectedLayout = AMBISONICS_WXYZ;
      break;
    case (AmbisonicOrder::SECOND_ORDER):
      expectedLayout = AMBISONICS_2ND;
      break;
    case (AmbisonicOrder::THIRD_ORDER):
      expectedLayout = AMBISONICS_3RD;
      break;
    default:
      std::stringstream ss;
      ss << "Order " << getStringFromAmbisonicOrder(order) << " cannot be tested";
      ENSURE(false, ss.str().c_str());
      return;
  }
  ENSURE(expectedLayout == output.getLayout(), "Check expected layout");

  for (auto &track : output) {
    checkAmbisonicTrack(inputMono[SPEAKER_FRONT_LEFT], track, res.at(track.channel()));
  }
}

void setExpectedAmbCoef(std::map<ChannelMap, double> &expectedRes, double w, double x, double y, double z,
                        double v = 0., double t = 0., double r = 0., double s = 0., double u = 0.) {
  expectedRes[SPEAKER_AMB_W] = w;
  expectedRes[SPEAKER_AMB_X] = x;
  expectedRes[SPEAKER_AMB_Y] = y;
  expectedRes[SPEAKER_AMB_Z] = z;
  expectedRes[SPEAKER_AMB_V] = v;
  expectedRes[SPEAKER_AMB_T] = t;
  expectedRes[SPEAKER_AMB_R] = r;
  expectedRes[SPEAKER_AMB_S] = s;
  expectedRes[SPEAKER_AMB_U] = u;
}

void testAmbisonicEncoderFirstOrder() {
  std::vector<double> freqs = {440.};
  SigGenSine sineGen(freqs, getDefaultSamplingRate(), 1.0);
  AudioBlock input(MONO);
  AudioBlock output;
  AmbEncoder ambEncoder(AmbisonicOrder::FIRST_ORDER, AmbisonicNorm::SN3D);
  input.resize(512);
  sineGen.step(input);
  ambEncoder.step(output, input);

  ENSURE_EQ(output.numSamples(), input.numSamples(), "Check number of samples");

  // Default position (0,0) -> (W,X,Y,Z) = (S,0,0,S)
  std::map<ChannelMap, double> expectedRes;
  setExpectedAmbCoef(expectedRes, 1., 1., 0., 0.);
  checkAmbisonicOutput(input, output, expectedRes, AmbisonicOrder::FIRST_ORDER);
  output.clear();

  // position (pi,0) -> (W,Y,Z,X) = (S,0,0,-S)
  ambEncoder.setMonoSourcePosition({M_PI, 0.});
  ambEncoder.step(output, input);
  setExpectedAmbCoef(expectedRes, 1., -1., 0., 0.);
  checkAmbisonicOutput(input, output, expectedRes, AmbisonicOrder::FIRST_ORDER);
  output.clear();

  // position (pi/2,0) -> (W,Y,Z,X) = (S,S,0,0)
  ambEncoder.setMonoSourcePosition({M_PI / 2., 0.});
  ambEncoder.step(output, input);
  setExpectedAmbCoef(expectedRes, 1., 0., 1., 0.);
  checkAmbisonicOutput(input, output, expectedRes, AmbisonicOrder::FIRST_ORDER);
  output.clear();

  // position (-pi/2,0) -> (W,Y,Z,X) = (S,-S,0,0)
  ambEncoder.setMonoSourcePosition({-M_PI / 2., 0.});
  ambEncoder.step(output, input);
  setExpectedAmbCoef(expectedRes, 1., 0., -1., 0.);
  checkAmbisonicOutput(input, output, expectedRes, AmbisonicOrder::FIRST_ORDER);
  output.clear();

  // position (x,pi/2) -> (W,Y,Z,X) = (S,0,S,0)
  ambEncoder.setMonoSourcePosition({M_PI / 4., M_PI / 2.});
  ambEncoder.step(output, input);
  setExpectedAmbCoef(expectedRes, 1., 0., 0., 1.);
  checkAmbisonicOutput(input, output, expectedRes, AmbisonicOrder::FIRST_ORDER);
  output.clear();

  // position (x,-pi/2) -> (W,Y,Z,X) = (S,0,-S,0)
  ambEncoder.setMonoSourcePosition({M_PI / 4., -M_PI / 2.});
  ambEncoder.step(output, input);
  setExpectedAmbCoef(expectedRes, 1., 0., 0., -1.);
  checkAmbisonicOutput(input, output, expectedRes, AmbisonicOrder::FIRST_ORDER);
  output.clear();
}

void testAmbisonicEncoderSecondOrder() {
  std::vector<double> freqs = {440.};
  SigGenSine sineGen(freqs, getDefaultSamplingRate(), 1.0);
  AudioBlock input(MONO);
  AudioBlock output;
  AmbEncoder ambEncoder(AmbisonicOrder::SECOND_ORDER, AmbisonicNorm::SN3D);
  input.resize(512);
  sineGen.step(input);
  ambEncoder.step(output, input);

  ENSURE_EQ(output.numSamples(), input.numSamples(), "Check number of samples");
  // Default position (0,0) -> (W,Y,Z,X,V,T,R,S,U) = (S,0,0,S,0,0,-1/2,0,sqrt(3)/2)
  std::map<ChannelMap, double> expectedCoef;
  setExpectedAmbCoef(expectedCoef, 1., 1., 0., 0., 0., 0., -1. / 2., 0., sqrt(3.) / 2.);
  checkAmbisonicOutput(input, output, expectedCoef, AmbisonicOrder::SECOND_ORDER);
  output.clear();

  // position (pi,0) -> (W,Y,Z,X,V,T,R,S,U)
  ambEncoder.setMonoSourcePosition({M_PI, 0.});
  ambEncoder.step(output, input);
  setExpectedAmbCoef(expectedCoef, 1., -1., 0., 0., 0., 0., -1. / 2., 0., sqrt(3.) / 2.);
  checkAmbisonicOutput(input, output, expectedCoef, AmbisonicOrder::SECOND_ORDER);
  output.clear();

  // position (pi/2,0) -> (W,Y,Z,X,V,T,R,S,U)
  ambEncoder.setMonoSourcePosition({M_PI / 2., 0.});
  ambEncoder.step(output, input);
  setExpectedAmbCoef(expectedCoef, 1., 0., 1., 0., 0., 0., -1. / 2., 0., -sqrt(3.) / 2.);
  checkAmbisonicOutput(input, output, expectedCoef, AmbisonicOrder::SECOND_ORDER);
  output.clear();

  // TODO: extend the test to other positions such as (0, pi/4), (0,-pi/4), (pi/4, 0), (-pi/4/0) etc...
}

void compareTracks(const AudioTrack &refTrack, const AudioTrack &outTrack, audioSample_t coef = 1.0) {
  ENSURE_EQ(refTrack.size(), outTrack.size(), "Check track size");
  for (size_t i = 0; i < outTrack.size(); ++i) {
    std::string str("Check out signal of track " + std::string(getStringFromChannelType(outTrack.channel())));
    ENSURE_APPROX_EQ(coef * refTrack[i], outTrack[i], 1e-5, str.c_str());
  }
}

void testAmbisonicRotator() {
  AmbRotator ambRotator(AmbisonicOrder::FIRST_ORDER);
  AudioBlock in(AMBISONICS_WXYZ), out;

  in.assign(3, 0.);
  audioSample_t iTrack = 1.;
  for (auto &track : in) {
    for (size_t s = 0; s < in.numSamples(); s++) {
      track[s] = (iTrack + audioSample_t(s)) * 1e-2;
    }
    iTrack++;
  }

  // Check default orientation
  ambRotator.setRotation(0., 0., 0.);
  ambRotator.step(out, in);
  ENSURE_EQ(in.getLayout(), out.getLayout(), "check layout");
  ENSURE_EQ(in.numSamples(), out.numSamples(), "check num samples");

  compareTracks(in[SPEAKER_AMB_W], out[SPEAKER_AMB_W]);
  compareTracks(in[SPEAKER_AMB_X], out[SPEAKER_AMB_X]);
  compareTracks(in[SPEAKER_AMB_Y], out[SPEAKER_AMB_Y]);
  compareTracks(in[SPEAKER_AMB_Z], out[SPEAKER_AMB_Z]);

  // Check yaw rotation of PI/2
  ambRotator.setRotation(M_PI_2, 0., 0.);
  ambRotator.step(out, in);
  compareTracks(in[SPEAKER_AMB_W], out[SPEAKER_AMB_W]);
  compareTracks(in[SPEAKER_AMB_X], out[SPEAKER_AMB_Y]);        // X ->  Y
  compareTracks(in[SPEAKER_AMB_Y], out[SPEAKER_AMB_X], -1.0);  // Y -> -X
  compareTracks(in[SPEAKER_AMB_Z], out[SPEAKER_AMB_Z]);        // Z ->  Z
  out.clear();

  // Check pitch rotation of PI/2
  // X ->  Z
  // Y ->  Y
  // Z -> -X
  ambRotator.setRotation(0., M_PI_2, 0.);
  ambRotator.step(out, in);
  compareTracks(in[SPEAKER_AMB_W], out[SPEAKER_AMB_W]);
  compareTracks(in[SPEAKER_AMB_X], out[SPEAKER_AMB_Z]);
  compareTracks(in[SPEAKER_AMB_Y], out[SPEAKER_AMB_Y]);
  compareTracks(in[SPEAKER_AMB_Z], out[SPEAKER_AMB_X], -1.0);
  out.clear();

  // Check roll rotation of PI/2
  // X ->  X
  // Y -> -Z
  // Z ->  Y
  ambRotator.setRotation(0., 0., M_PI_2);
  ambRotator.step(out, in);
  compareTracks(in[SPEAKER_AMB_W], out[SPEAKER_AMB_W]);
  compareTracks(in[SPEAKER_AMB_X], out[SPEAKER_AMB_X]);
  compareTracks(in[SPEAKER_AMB_Y], out[SPEAKER_AMB_Z]);
  compareTracks(in[SPEAKER_AMB_Z], out[SPEAKER_AMB_Y], -1.0);
  out.clear();

  // Check offset on roll
  // back to the origin
  ambRotator.setRotationOffset(0., 0., -M_PI_2);
  ambRotator.step(out, in);
  compareTracks(in[SPEAKER_AMB_W], out[SPEAKER_AMB_W]);
  compareTracks(in[SPEAKER_AMB_X], out[SPEAKER_AMB_X]);
  compareTracks(in[SPEAKER_AMB_Y], out[SPEAKER_AMB_Y]);
  compareTracks(in[SPEAKER_AMB_Z], out[SPEAKER_AMB_Z]);
  out.clear();

  // Check offset on pitch
  ambRotator.setRotationOffset(0., M_PI_2, -M_PI_2);
  ambRotator.step(out, in);
  compareTracks(in[SPEAKER_AMB_W], out[SPEAKER_AMB_W]);
  compareTracks(in[SPEAKER_AMB_X], out[SPEAKER_AMB_Z]);
  compareTracks(in[SPEAKER_AMB_Y], out[SPEAKER_AMB_Y]);
  compareTracks(in[SPEAKER_AMB_Z], out[SPEAKER_AMB_X], -1.0);
  out.clear();

  // Check offset on yaw
  ambRotator.setRotationOffset(M_PI_2, 0., -M_PI_2);
  ambRotator.step(out, in);
  compareTracks(in[SPEAKER_AMB_W], out[SPEAKER_AMB_W]);
  compareTracks(in[SPEAKER_AMB_X], out[SPEAKER_AMB_Y]);        // X ->  Y
  compareTracks(in[SPEAKER_AMB_Y], out[SPEAKER_AMB_X], -1.0);  // Y -> -X
  compareTracks(in[SPEAKER_AMB_Z], out[SPEAKER_AMB_Z]);        // Z ->  Z
  out.clear();
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  std::string testData(VideoStitch::Testing::getDataFolder());
  std::string inWav("data/snd/cos48k_mono.wav"), outWav(testData + "/tmp.wav");

  if (argc == 2) {
    inWav = argv[1];
  }

  if (argc == 3) {
    outWav = argv[2];
  }

  std::cout << "Run tests with input file " << inWav << " output file " << outWav << std::endl;

  // Open a wav file
  // Copy it in an other wav file
  std::cout << "RUN Test to copy a wav file into an other" << std::endl;
  VideoStitch::Testing::copyWavFile(inWav, outWav);
  // Compare the two files
  VideoStitch::Testing::compareWavFile(inWav, outWav, 1. / 32267.);
  std::cout << "Test to copy a wav file into an other : PASSED" << std::endl;

  // Generate a sin at 1 kHz and compare it to a signal generated
  std::cout << "RUN Test sine generator" << std::endl;
  VideoStitch::Testing::cosineTest();
  std::cout << "Test sine generator: PASSED" << std::endl;

  std::cout << "RUN Test conversion Samples to AudioBlock" << std::endl;
  VideoStitch::Testing::testSamplesToAudioBlock();

  std::cout << "RUN Test conversion AudioBlock to Samples" << std::endl;
  VideoStitch::Testing::testAudioBlkToSamples();

  // Test AudioSamples::drop
  std::cout << "RUN Test AudioSamples::drop(n)" << std::endl;
  VideoStitch::Testing::testDrop();
  std::cout << "RUN Test AudioSamples::drop(n) PASSED" << std::endl;

  // Test AudioSamples::append
  std::cout << "RUN Test AudioSamples::append(samples)" << std::endl;
  VideoStitch::Testing::testAppend();
  std::cout << "RUN Test AudioSamples::append(samples) PASSED" << std::endl;

  // Test envelope detector
  std::cout << "RUN Test EnvelopeDetector" << std::endl;
  VideoStitch::Testing::testSimpleEnvDetector();
  VideoStitch::Testing::testEnvDetector();
  std::cout << "RUN Test EnvelopeDetector PASSED" << std::endl;

  // Test audio block sum
  std::cout << "RUN Test Audio Block Summer" << std::endl;
  VideoStitch::Testing::testAudioSum();
  std::cout << "RUN Test Audio Block Summer PASSED" << std::endl;

  // Test ambisonic encoder
  std::cout << "RUN Test First Order Ambisonic Encoder" << std::endl;
  VideoStitch::Testing::testAmbisonicEncoderFirstOrder();
  std::cout << "RUN Test First Order Ambisonic Encoder PASSED" << std::endl;

  std::cout << "RUN Test Second Order Ambisonic Encoder" << std::endl;
  VideoStitch::Testing::testAmbisonicEncoderSecondOrder();
  std::cout << "RUN Test Second Order Ambisonic Encoder PASSED" << std::endl;

  std::cout << "RUN Test Ambisonic Rotator" << std::endl;
  VideoStitch::Testing::testAmbisonicRotator();
  std::cout << "RUN Test Ambisonic Rotator PASSED" << std::endl;

  return 0;
}
