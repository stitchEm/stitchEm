// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <cassert>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <android/log.h>
#include <unistd.h>
#include <stdlib.h> /* getenv() */
#include <libgen.h> /* dirname() */

#include "NvGLUtils/NvGLSLProgram.h"
#include "NvAssetLoader/NvAssetLoader.h"

#include <libgpudiscovery/backendLibHelper.hpp>

#include <libvideostitch/algorithm.hpp>
#include <libvideostitch/audioPipeDef.hpp>
#include <libvideostitch/controller.hpp>
#include <libvideostitch/context.hpp>
#include <libvideostitch/imageMergerFactory.hpp>
#include <libvideostitch/imageWarperFactory.hpp>
#include <libvideostitch/imageFlowFactory.hpp>
#include <libvideostitch/inputFactory.hpp>
#include <libvideostitch/logging.hpp>
#include <libvideostitch/output.hpp>
#include <libvideostitch/panoDef.hpp>
#include <libvideostitch/parse.hpp>
#include <libvideostitch/profile.hpp>
#include <libvideostitch/ptv.hpp>
#include <libvideostitch/stitchOutput.hpp>
#include <libvideostitch/frame.hpp>

#define APPNAME "OrahDemo"
static std::string ANDROIDTag(APPNAME);

#include "NativeApp.h"

extern void NvInitSharedFoundation(void);
extern void NvReleaseSharedFoundation(void);

static bool stop = false;

typedef struct {
  double x;
  double y;
  double z;
} vector3d;

static std::string ASSETSDIR = "/data/user/videostitch/assets/";

void launchStitcher(void* outBuffer, std::mutex& outMutex, const char* fileName);

static double SphereRadius = 0.0;

double degToRad(double v) { return M_PI * (v / 180.0); }

double radToDeg(double v) { return v * (180.0 / M_PI); }

void equirectangular2sphere(vector2d uv, vector3d& vec, double sphereRadius) {
  uv.x /= sphereRadius;
  uv.y /= sphereRadius;
  double phi = uv.x;
  double theta = -uv.y + M_PI / 2.0;
  // Pass above the north pole
  if (theta < 0.0) {
    theta = -theta;
    phi += M_PI;
  }
  // Pass above the south pole
  if (theta > M_PI) {
    theta = 2.0 * M_PI - theta;
    phi += M_PI;
  }
  double sinTheta = sin(theta);
  vector2d v;
  v.x = sinTheta * sin(phi);
  v.y = cos(theta);
  double r = sqrt(v.x * v.x + v.y * v.y);
  if (r != 0.0f) {
    // Normal case, atan2f is defined in this domain.
    uv.x = (atan2(r, sinTheta * cos(phi)) / r) * v.x;
    uv.y = (atan2(r, sinTheta * cos(phi)) / r) * v.y;
  } else {
    // atan2f is not defined around (0,0) and (pi + k pi, pi/2).
    // The result is taken to be 0 at (0,0) and defined by continuity modulo (2 pi) along the phi axis to be (pi at (pi
    // + k pi, pi/2).
    if (std::abs(phi) < 0.001f) {  // <==> sin(theta) * cos(phi) >= 0
      uv.x = 0.0;
      uv.y = 0.0;
    } else {
      uv.x = M_PI;
      uv.y = 0.0;
    }
  }
  theta = sqrt(uv.x * uv.x + uv.y * uv.y);
  vec.x = (uv.x * sin(theta) / theta);
  vec.y = (uv.y * sin(theta) / theta);
  vec.z = (cos(theta));
}

void quaternion2euler(double q0, double q1, double q2, double q3, double& yaw, double& pitch, double& roll) {
  yaw = atan2(2.0 * (q1 * q3 - q0 * q2), q3 * q3 - q2 * q2 - q1 * q1 + q0 * q0);
  pitch = -asin(2.0 * (q2 * q3 + q0 * q1));
  roll = atan2(2.0 * (q1 * q2 - q0 * q3), q2 * q2 - q3 * q3 + q0 * q0 - q1 * q1);
}

void sphere2orientation(const vector3d& v1, const vector3d& v2, double& yaw, double& pitch, double& roll) {
  double dotProduct = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
  if (dotProduct > 1.0) {
    dotProduct = 1.0;
  }
  double theta_2 = acos(dotProduct) / 2.0;

  vector3d nu;
  nu.x = (v1.y * v2.z) - (v1.z * v2.y);
  nu.y = (v1.z * v2.x) - (v1.x * v2.z);
  nu.z = (v1.x * v2.y) - (v1.y * v2.x);

  double sintheta_2 = sin(theta_2);
  double q0 = cos(theta_2);
  double q1 = sintheta_2 * nu.x;
  double q2 = sintheta_2 * nu.y;
  double q3 = sintheta_2 * nu.z;
  quaternion2euler(q0, q1, q2, q3, yaw, pitch, roll);
}

void NVPlatformLog(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  __android_log_vprint(ANDROID_LOG_INFO, APPNAME, fmt, ap);
  va_end(ap);
}

// redirect stdout & stderr to android logcat
static int pfd[2];
static pthread_t thr;

static void* thread_func(void*) {
  ssize_t rdsz;
  char buf[128];
  while ((rdsz = read(pfd[0], buf, sizeof buf - 1)) > 0) {
    if (buf[rdsz - 1] == '\n') --rdsz;
    buf[rdsz] = 0; /* add null-terminator */
    __android_log_write(ANDROID_LOG_DEBUG, APPNAME, buf);
  }
  return 0;
}

static int start_logger() {
  /* make stdout line-buffered and stderr unbuffered */
  setvbuf(stdout, 0, _IOLBF, 0);
  setvbuf(stderr, 0, _IONBF, 0);

  /* create the pipe and redirect stdout and stderr */
  pipe(pfd);
  dup2(pfd[1], 1);
  dup2(pfd[1], 2);

  /* spawn the logging thread */
  if (pthread_create(&thr, 0, thread_func, 0) == -1) return -1;
  pthread_detach(thr);
  return 0;
}

NativeApp::NativeApp(android_app* app, NvEGLUtil* egl)
    : mTextureWidth(1920),
      mTextureHeight(960),
      mEgl(egl),
      mNativeAppInstance(app),
      mStitcherBuffer(NULL),
      mStitcherThread(NULL),
      mOrientationFlag(false),
      mColorChannels{1.0, 1.0, 1.0} {
  app->userData = this;
  app->onAppCmd = HandleCommand;
  app->onInputEvent = HandleInput;

  mCurrentApplicationState = INITIALIZATION;
  mScreenPressed = false;
  mYaw = 0.0f;
  mPitch = 0.0f;
  mRoll = 0.0f;
  mAlphaStep = -0.05f;
  mPos.x = 0.0f;
  mPos.y = 0.0f;
}

NativeApp::~NativeApp() {
  mStitcherThread->join();
  delete[] mStitcherBuffer;
}

/**
 *  Called every frame
 */
void NativeApp::renderFrame(void) {
  if (!mEgl->isReadyToRender(true)) {
    return;
  }

  if (mEgl->checkWindowResized()) {
    glViewport(0, 0, mEgl->getWidth(), mEgl->getHeight());
  }

  switch (mCurrentApplicationState) {
    case INITIALIZATION: {
      // vertex shader
      const char* vertexShaderSource =
          "attribute vec2 aPosition;\n"
          "attribute vec2 aTexCoord;\n"
          "varying vec2 vTexCoord;\n"
          "void main(void)\n"
          "{\n"
          "   vTexCoord = aTexCoord;\n"
          "   gl_Position = vec4(aPosition.x, aPosition.y, 0.0, 1.0);\n"
          "}\n";

      // fragment shader
      const char* fragmentShaderSource =
          "precision mediump float;\n"
          "uniform sampler2D uSourceTex;\n"
          "varying vec2 vTexCoord;\n"
          "void main(void)\n"
          "{\n"
          "    gl_FragColor = texture2D(uSourceTex, vTexCoord);\n"
          "}\n";

      //                mPlainTextureProgram = NvGLSLProgram::createFromFiles("plain.vert", "plain.frag",
      //                true)->getProgram();
      mPlainTextureProgram =
          NvGLSLProgram::createFromStrings(vertexShaderSource, fragmentShaderSource, true)->getProgram();

      // output from Stitcher is stored in mStitcherBuffer
      mStitcherBuffer = new unsigned int[mTextureWidth * mTextureHeight];

      // generate output texture
      glGenTextures(1, &mOutputImageTexture);
      glBindTexture(GL_TEXTURE_2D, mOutputImageTexture);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mTextureWidth, mTextureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
      mAlphaStep = -0.05f;
      mColorChannels[0] = 1.0;
      mColorChannels[1] = mColorChannels[0];
      mColorChannels[2] = mColorChannels[0];
      glClearColor(mColorChannels[0], mColorChannels[1], mColorChannels[2], 1.0);
      glClear(GL_COLOR_BUFFER_BIT);
      glFinish();
      //                mCurrentApplicationState = MAIN_LOOP;
      mCurrentApplicationState = SELECTION;
      break;
    }

    case SELECTION: {
      mColorChannels[0] += mAlphaStep;
      mColorChannels[1] += mAlphaStep;
      mColorChannels[2] += mAlphaStep;
      if ((mColorChannels[0] < 0.0) || (mColorChannels[0] > 1.0)) {
        mAlphaStep = -mAlphaStep;
        mColorChannels[0] += mAlphaStep;
        mColorChannels[1] += mAlphaStep;
        mColorChannels[2] += mAlphaStep;
      }
      glClearColor(mColorChannels[0], mColorChannels[1], mColorChannels[2], 1.0);
      glClear(GL_COLOR_BUFFER_BIT);
      glFinish();
    } break;

    case MAIN_LOOP: {
      // bind output texture
      glBindTexture(GL_TEXTURE_2D, mOutputImageTexture);
      mStitcherMutex.lock();
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, mTextureWidth, mTextureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                   (unsigned char*)mStitcherBuffer);
      mStitcherMutex.unlock();

      float const vertexPosition[] = {1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
      float const textureCoord[] = {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};

      // Setup fragment program uniforms
      int program = mPlainTextureProgram;
      glUseProgram(program);

      // Setup uniforms
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, mOutputImageTexture);

      glUniform1i(glGetUniformLocation(program, "uSourceTex"), 0);

      // Rendering quad
      int aPosCoord = glGetAttribLocation(program, "aPosition");
      int aTexCoord = glGetAttribLocation(program, "aTexCoord");

      glVertexAttribPointer(aPosCoord, 2, GL_FLOAT, GL_FALSE, 0, vertexPosition);
      glVertexAttribPointer(aTexCoord, 2, GL_FLOAT, GL_FALSE, 0, textureCoord);
      glEnableVertexAttribArray(aPosCoord);
      glEnableVertexAttribArray(aTexCoord);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      glDisableVertexAttribArray(aPosCoord);
      glDisableVertexAttribArray(aTexCoord);
      break;
    }

    case EXIT:
      mStitcherThread->join();
      //            delete [] mStitcherBuffer;
      // NOP
      break;
  }

  mEgl->swap();
}

/**
 *  returns current Sate of the application
 */
NativeApp::State NativeApp::getState(void) const { return mCurrentApplicationState; }

/**
 * Handle commands.
 */
void NativeApp::handleCommand(int cmd) {
  switch (cmd) {
    case APP_CMD_INIT_WINDOW:
    case APP_CMD_WINDOW_RESIZED:
      mEgl->setWindow(mNativeAppInstance->window);
      break;

    case APP_CMD_TERM_WINDOW:
      mEgl->setWindow(NULL);
      break;

    default:
      break;
  }
}

/**
 * Handle inputs.
 */
int NativeApp::handleInput(AInputEvent const* event) {
  // Here we only handle key back event
  int32_t type = AInputEvent_getType(event);

  if (type == AINPUT_EVENT_TYPE_KEY) {
    if (AKeyEvent_getKeyCode(event) == AKEYCODE_BACK) {
      ANativeActivity_finish(mNativeAppInstance->activity);
      __android_log_print(ANDROID_LOG_INFO, APPNAME, "ANativeActivity_finish()");
      stop = true;
      return 1;
    }
  } else {
    int32_t action = AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK;

    switch (action) {
      case AMOTION_EVENT_ACTION_DOWN:
        if (mCurrentApplicationState == MAIN_LOOP) {
          mPos.x = (AMotionEvent_getX(event, 0) - (mEgl->getWidth() / 2)) * (mTextureWidth * 1.0f / mEgl->getWidth());
          mPos.y =
              (AMotionEvent_getY(event, 0) - (mEgl->getHeight() / 2)) * (mTextureHeight * 1.0f / mEgl->getHeight());
          mScreenPressed = true;
        } else if (mCurrentApplicationState == SELECTION) {
          float x = AMotionEvent_getX(event, 0);
          __android_log_print(ANDROID_LOG_INFO, APPNAME, "AMotionEvent_getX() %d / %d", (int)x, mEgl->getWidth());
          float fX = x / mEgl->getWidth();
          if (fX < 0.4f) {
            mStitcherThread =
                new std::thread([=]() { launchStitcher((void*)mStitcherBuffer, mStitcherMutex, "av-rtmp.ptv"); });
            glClearColor(1.0, 0.0, 0.0, 1.0);
          } else if (fX < 0.7f) {
            mStitcherThread =
                new std::thread([=]() { launchStitcher((void*)mStitcherBuffer, mStitcherMutex, "procedural.ptv"); });
            glClearColor(0.0, 1.0, 0.0, 1.0);
          } else {
            mStitcherThread =
                new std::thread([=]() { launchStitcher((void*)mStitcherBuffer, mStitcherMutex, "profiling.ptv"); });
            glClearColor(0.0, 0.0, 1.0, 1.0);
          }
          glClear(GL_COLOR_BUFFER_BIT);
          glFinish();
          mCurrentApplicationState = MAIN_LOOP;
        }
        break;

      case AMOTION_EVENT_ACTION_UP:
        if (mCurrentApplicationState == MAIN_LOOP) {
          mScreenPressed = false;
        }
        break;

      case AMOTION_EVENT_ACTION_MOVE:
        if (mCurrentApplicationState == MAIN_LOOP) {
          vector2d pos;
          pos.x = (AMotionEvent_getX(event, 0) - (mEgl->getWidth() / 2)) * (mTextureWidth * 1.0f / mEgl->getWidth());
          pos.y = (AMotionEvent_getY(event, 0) - (mEgl->getHeight() / 2)) * (mTextureHeight * 1.0f / mEgl->getHeight());

          //                point2pano(curPos);
          //            	  QPoint topLeftCorner((width - texWidth * zoom / 100) / (2 * devicePixelRatio),
          //            	                       (height - texHeight * zoom / 100) / (2 * devicePixelRatio));
          //            	  p -= topLeftCorner;
          //            	  p *= 100 / zoom * devicePixelRatio;
          //            	  p.rx() -= texWidth / 2;
          //            	  p.ry() -= texHeight / 2;

          vector3d v1, v2;
          equirectangular2sphere(pos, v2, SphereRadius);
          equirectangular2sphere(mPos, v1, SphereRadius);
          double y, p, r;
          sphere2orientation(v2, v1, y, p, r);
          mOrientationMutex.lock();
          mRoll = radToDeg(r);
          mYaw = radToDeg(y);
          mPitch = radToDeg(p);
          mPos = pos;
          mOrientationFlag = true;
          mOrientationMutex.unlock();
        }
        break;
    }
    return 1;
  }
  return 0;
}

/**
 * Wrapper to handle commands generated by the UI.
 */
void NativeApp::HandleCommand(android_app* app, int32_t cmd) {
  static_cast<NativeApp*>(app->userData)->handleCommand(cmd);
}

/**
 * Wrapper to handle input events generated by the UI.
 */
int32_t NativeApp::HandleInput(android_app* app, AInputEvent* event) {
  return static_cast<NativeApp*>(app->userData)->handleInput(event);
}

class AndroidOutputWriter : public VideoStitch::Output::VideoWriter {
 public:
  AndroidOutputWriter(const std::string& name, void* outBuffer, std::mutex& outMutex, unsigned int w, unsigned int h)
      : Output(name),
        VideoWriter(w, h, {30, 1}, VideoStitch::PixelFormat::RGBA, AddressSpace::Host),
        dst(outBuffer),
        mutex(outMutex) {
    __android_log_print(ANDROID_LOG_INFO, APPNAME, "AndroidOutputWriter::AndroidOutputWriter()");
    //		frame = std::make_shared<Frame>(w, h);
  }

  ~AndroidOutputWriter() {
    __android_log_print(ANDROID_LOG_INFO, APPNAME, "AndroidOutputWriter::~AndroidOutputWriter()");
  }

  virtual void pushVideo(const Frame& videoFrame) {  // int frameId, const char* video, size_t, uint8_t* const*) {
    mutex.lock();
    memcpy(dst, videoFrame.planes[0], videoFrame.width * videoFrame.height * 4);
    mutex.unlock();
  }

 private:
  //  std::shared_ptr<Frame> frame;
  void* dst;
  std::mutex& mutex;
};

struct ParsingError {
  explicit ParsingError(std::string const& msg) : msg_(msg) {
    __android_log_print(ANDROID_LOG_INFO, APPNAME, "ParsingError: %s", msg_.c_str());
  }

  const std::string msg_;
};

struct StitchError {
  explicit StitchError(std::string const& msg) : msg_(msg) {
    __android_log_print(ANDROID_LOG_INFO, APPNAME, "StitchError: %s", msg_.c_str());
  }

  const std::string msg_;
};

void normalizeOutputConfig(VideoStitch::Ptv::Value* outputConfig, VideoStitch::Core::Controller const* controller) {
  // If the input frame rate is known, set the output frame rate.
  VideoStitch::Logger::get(VideoStitch::Logger::Verbose)
      << "Setting output frame rate to " << controller->getReaderSpec(0).frameRate << "." << std::endl;
  outputConfig->get("fps")->asDouble() =
      controller->getReaderSpec(0).frameRate.num * 1.0 / controller->getReaderSpec(0).frameRate.den;
}

bool normalizeFrameBoundaries(VideoStitch::Core::Controller const* controller, const int firstFrame, int& lastFrame) {
  lastFrame = (lastFrame < 0) ? controller->getLastStitchableFrame()
                              : std::min(lastFrame, controller->getLastStitchableFrame());
  if (lastFrame == (int)NO_LAST_FRAME && lastFrame < 0) {
    VideoStitch::Logger::get(VideoStitch::Logger::Error) << "Last frame autodetection was enabled, but all the readers "
                                                            "are unbounded (Are you using only procedural readers ?)."
                                                         << std::endl;
    return false;
  }

  if (lastFrame < firstFrame) {
    VideoStitch::Logger::get(VideoStitch::Logger::Error)
        << "Nothing to stitch: last_frame = " << lastFrame << " < first_frame = " << firstFrame << "." << std::endl;
    return false;
  }

  VideoStitch::Logger::get(VideoStitch::Logger::Info)
      << "Will stitch " << lastFrame - firstFrame + 1 << " images." << std::endl;
  return true;
}

template <typename Controller>
Output::Output* makeCallback(Controller* controller, Core::PanoDefinition& pano, Ptv::Value* outputConfig,
                             Audio::SamplingRate outRate, Audio::SamplingDepth outDepth,
                             Audio::ChannelLayout outLayout) {
  size_t width = pano.getWidth(), height = pano.getHeight();
  if (pano.getProjection() == VideoStitch::Core::PanoProjection::Cubemap ||
      pano.getProjection() == VideoStitch::Core::PanoProjection::EquiangularCubemap) {
    width = 3 * pano.getLength();
    height = 2 * pano.getLength();
  } else {
    width = pano.getWidth();
    height = pano.getHeight();
  }

  Potential<Output::Output> pot =
      Output::create(*outputConfig, outputConfig->has("filename")->asString(), (unsigned)width, (unsigned)height,
                     controller->getFrameRate(), outRate, outDepth, outLayout);
  if (!pot.ok()) {
    Logger::get(Logger::Error) << "Output writer creation failed!" << std::endl;
    return nullptr;
  }
  return pot.release();
}

VideoStitch::Potential<VideoStitch::Core::StitchOutput> createStitchOutput(
    VideoStitch::Core::Controller* controller, VideoStitch::Core::PanoDefinition const& pano,
    std::shared_ptr<Output::VideoWriter> writer) {
  std::vector<std::shared_ptr<Core::PanoSurface>> surfs;

  for (int i = 0; i < 2; ++i) {
    auto panoSurface = Core::OffscreenAllocator::createPanoSurface(pano.getWidth(), pano.getHeight(), "StitchOutput");
    FAIL_RETURN(panoSurface.status());
    surfs.push_back(std::shared_ptr<Core::PanoSurface>(panoSurface.release()));
  }

  // TODO: allow only one thread for this one; this is not thread-safe.
  return controller->createAsyncStitchOutput(surfs, writer);
}

/**
 * The stitcher loop for one device.
 */
int NativeApp::stitchRun(VideoStitch::Core::Controller* controller, VideoStitch::Core::StitchOutput* stitchOutput,
                         const int firstFrame, const int lastFrame) {
  auto stitcher = controller->createStitcher();
  if (!stitcher.ok()) throw StitchError("Could not create stitcher for GPU 0.");

  int curFrame = firstFrame;
  while (!stop && (curFrame <= lastFrame)) {
    double yaw, pitch, roll;
    int frame = (++curFrame) - 1;
    yaw = 0.0;
    pitch = 0.0;
    roll = 0.0;
    if (mOrientationFlag) {
      mOrientationMutex.lock();
      yaw = mYaw;
      //    	pitch = mPitch;
      //    	roll = mRoll;
      mOrientationFlag = false;
      mOrientationMutex.unlock();
      controller->applyRotation(yaw, pitch, roll);
    }
    if (!controller->stitch(stitchOutput).ok()) {
      std::ostringstream str;
      str << "Failed to stitch frame " << frame;
      throw StitchError(str.str());
    }
    usleep(10000);
  }
  return 0;
}

void NativeApp::launchStitcher(void* outBuffer, std::mutex& outMutex, const char* fileName) {
  Potential<Ptv::Parser> parser = Ptv::Parser::create();
  if (!parser.ok()) throw ParsingError("Error: Cannot create parser.");
  __android_log_print(ANDROID_LOG_INFO, APPNAME, "void NativeApp::launchStitcher() starting");

  // Load the project and parse it.
  if (!parser->parse(fileName)) {
    std::ostringstream l_str;
    l_str << "Error: Cannot parse PTV file: " << parser->getErrorMessage();
    throw ParsingError(l_str.str());
  }

  bool needToRestart = false;
  VideoStitch::Discovery::Framework selectedFramework = VideoStitch::BackendLibHelper::getBestFrameworkAndBackend();
  if (VideoStitch::BackendLibHelper::selectBackend(selectedFramework, &needToRestart)) {
    __android_log_print(ANDROID_LOG_INFO, APPNAME, "%s backend selected",
                        VideoStitch::Discovery::getFrameworkName(selectedFramework).c_str());
  } else {
    throw StitchError("No capable GPU detected on your system");
  }

  int device;
  if ((!Discovery::getBackendDeviceIndex(0, device)) || (!GPU::setDefaultBackendDeviceVS(device).ok()) ||
      (!GPU::checkDefaultBackendDeviceInitialization().ok())) {
    throw StitchError("Cannot set default device");
  }

  Core::PanoDefinition* panoDef = Core::PanoDefinition::create(*parser->getRoot().has("pano"));
  if (!panoDef) {
    throw StitchError("Invalid panorama definition!");
  }
  std::unique_ptr<Core::PanoDefinition> pano(panoDef);

  SphereRadius = panoDef->getWidth() / (7.0f * /*tanf*/ (degToRad(pano->getHFOV()) /*/ 2.0f*/));
  //   	SphereRadius = 960.0 / (8.0 * degToRad(pano->getHFOV()));
  //	__android_log_print(ANDROID_LOG_INFO, APPNAME, "HFOV: %f (%f)", float(pano->getHFOV()),
  //float(degToRad(pano->getHFOV())));
  //	__android_log_print(ANDROID_LOG_INFO, APPNAME, "SphereRadius: %f", float(SphereRadius));

  Potential<Core::ImageMergerFactory> imageMergerFactory =
      Core::ImageMergerFactory::createMergerFactory(*parser->getRoot().has("merger"));
  const Ptv::Value* output = parser->getRoot().has("output");
  if (!output) {
    __android_log_print(ANDROID_LOG_INFO, APPNAME, "Missing output config");
  }
  Potential<Core::ImageWarperFactory> imageWarperFactory =
      Core::ImageWarperFactory::createWarperFactory(parser->getRoot().has("warper"));

  Potential<Core::ImageFlowFactory> imageFlowFactory =
      Core::ImageFlowFactory::createFlowFactory(parser->getRoot().has("flow"));

  std::unique_ptr<Core::AudioPipeDefinition> audioPipeDef;
  if (parser->getRoot().has("audio_pipe")) {
    audioPipeDef = std::unique_ptr<Core::AudioPipeDefinition>(
        Core::AudioPipeDefinition::create(*parser->getRoot().has("audio_pipe")));
  } else {
    audioPipeDef = std::unique_ptr<Core::AudioPipeDefinition>(
        Core::AudioPipeDefinition::createAudioPipeFromPanoInputs(pano.get()));
  }

  const int firstFrame = 0;
  int lastFrame = 5000;
  Input::ReaderFactory* readerFactory = new Input::DefaultReaderFactory(firstFrame, lastFrame);
  if (!readerFactory) throw StitchError("Reader factory creation failed!");

  Core::PotentialController controller =
      Core::createController(*pano, *imageMergerFactory.object(), *imageWarperFactory.object(),
                             *imageFlowFactory.object(), readerFactory, *audioPipeDef.get());
  if (!controller.ok()) throw StitchError("Controller creation failed!");
  assert(controller->getPano().numInputs());

  const std::unique_ptr<Ptv::Value> outputConfig((*output).clone());
  normalizeOutputConfig(outputConfig.get(), controller.object());

  std::shared_ptr<Output::Output> sharedWriter(makeCallback(controller.object(), *pano, outputConfig.get(),
                                                            Audio::SamplingRate::SR_48000, Audio::SamplingDepth::FLT_P,
                                                            Audio::STEREO));
  if (!sharedWriter) {
    throw StitchError("Output writer creation failed!");
  }

  normalizeFrameBoundaries(controller.object(), firstFrame, lastFrame);

  //	PROBLEM HERE BELOW!!! (40 seconds wait)
  // Create (possibly buffered) StitchOutput(s).
  const Potential<Core::StitchOutput> stitchOutput(createStitchOutput(
      controller.object(), *pano, std::dynamic_pointer_cast<VideoStitch::Output::VideoWriter>(sharedWriter)));

  if (panoDef->getWidth() <= mTextureWidth && panoDef->getHeight() <= mTextureHeight) {
    VideoStitch::Potential<Output::VideoWriter> writerDisplay(
        new AndroidOutputWriter("Suzanne", outBuffer, outMutex, mTextureWidth, mTextureHeight));
    if (!writerDisplay.ok()) throw StitchError("Output Display creation failed!");
    std::shared_ptr<Output::Output> sharedWriterDisplay(writerDisplay.release());

    stitchOutput.object()->addWriter(std::dynamic_pointer_cast<VideoStitch::Output::VideoWriter>(sharedWriterDisplay));
  }

  stitchRun(controller.object(), stitchOutput.object(), firstFrame, lastFrame);

  //	delete readerFactory;
  delete panoDef;
}

/**
 * This is the main entry point of a native application that is using
 * android_native_app_glue.  It runs in its own thread, with its own
 * event loop for receiving input events and doing other things.
 */
void android_main(android_app* androidApp) {
  // Make sure glue isn't stripped.
  app_dummy();

  NvInitSharedFoundation();

  NvAssetLoaderInit((void*)androidApp->activity->assetManager);

  // redirect Videostitch library logs to android logging
  Logger::setLevel(Logger::Info);
  start_logger();

  NvEGLUtil* egl = NvEGLUtil::create();
  if (egl == 0) {
    // If we have a basic EGL failure, we need to exit immediately; nothing else we can do
    nv_app_force_quit_no_cleanup(androidApp);
    return;
  }

  // some cache directory will be created to store Opencl compiled code while default HOME is "/"
  setenv("HOME", androidApp->activity->obbPath, 0);
  NativeApp* instance = new NativeApp(androidApp, egl);

  JNIEnv* env = nullptr;
  androidApp->activity->vm->AttachCurrentThread(&env, NULL);

  jclass android_content_Context = env->GetObjectClass(androidApp->activity->clazz);
  jmethodID midGetPackageCodePath =
      env->GetMethodID(android_content_Context, "getPackageCodePath", "()Ljava/lang/String;");
  jstring packageCodePath = (jstring)env->CallObjectMethod(androidApp->activity->clazz, midGetPackageCodePath);

  std::string dir = dirname(env->GetStringUTFChars(packageCodePath, 0));

  int count = VideoStitch::Plugin::loadPlugins(dir + "/lib/" + ANDROID_ARCH_NAME);
  __android_log_print(ANDROID_LOG_INFO, APPNAME, "Loaded %d plugin(s) in %s.", count, dir.c_str());

  if (chdir(ASSETSDIR.c_str()) != 0) {
    // getExternalFilesDir should correspond to /storage/emulated/0/Android/data/co.orah.stitch360/files
    jmethodID midGetFilesDir =
        env->GetMethodID(android_content_Context, "getExternalFilesDir", "(Ljava/lang/String;)Ljava/io/File;");
    jobject objectFile = env->CallObjectMethod(androidApp->activity->clazz, midGetFilesDir, NULL);
    // Call method on File object to retrieve String object.
    jclass classFile = env->GetObjectClass(objectFile);
    jmethodID methodIDgetAbsolutePath2 = env->GetMethodID(classFile, "getAbsolutePath", "()Ljava/lang/String;");
    jstring stringPath = (jstring)env->CallObjectMethod(objectFile, methodIDgetAbsolutePath2);

    // Extract a C string from the String object, and chdir() to it.
    const char* wpath = env->GetStringUTFChars(stringPath, NULL);

    if (chdir(wpath) != 0) {
      __android_log_print(ANDROID_LOG_INFO, APPNAME, "Unable to change working directory to %s.", wpath);
    } else {
      __android_log_print(ANDROID_LOG_INFO, APPNAME, "Working directory set to %s.", wpath);
    }

    env->ReleaseStringUTFChars(stringPath, wpath);
  } else {
    __android_log_print(ANDROID_LOG_INFO, APPNAME, "Working directory set to %s.", ASSETSDIR.c_str());
  }

  androidApp->activity->vm->DetachCurrentThread();

  while (nv_app_status_running(androidApp)) {
    // Read all pending events.
    int ident, events;
    struct android_poll_source* source;

    // If not rendering, we will block forever waiting for events.
    // If animating, we loop until all events are read, then continue
    // to draw the next frame of animation.
    while ((ident = ALooper_pollAll((nv_app_status_focused(androidApp) ? 1 : 250), NULL, &events, (void**)&source)) >=
           0) {
      // If we timed out, then there are no pending messages.
      if (ident == ALOOPER_POLL_TIMEOUT) {
        break;
      }

      // Process this event.
      if (source != NULL) {
        source->process(androidApp, source);
      }

      // Check if we are exiting.  If so, dump out.
      if (!nv_app_status_running(androidApp)) {
        break;
      }
    }

    if (nv_app_status_interactable(androidApp)) {
      instance->renderFrame();
    }
  }

  // Remove application instance
  delete instance;
  // Remove EGL instance
  delete egl;

  NvReleaseSharedFoundation();

  NvAssetLoaderShutdown();
}
