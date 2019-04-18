// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "../../include/libvideostitch/allocator.hpp"
#include "../../include/libvideostitch/logging.hpp"

#include <GL/glew.h>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#ifdef __linux__
#include <GLFW/glfw3.h>
#else
#include <glfw/glfw3.h>
#endif

#include <functional>
#include <queue>
#include <mutex>
#include <chrono>
#include <limits>
#include <thread>
#include <condition_variable>

static std::string RENDERERtag = "OpenGLRenderer";
static std::string GLFWtag = "GLFW";

/** @brief Render for OpenGL
 *
 * - Keeps a queue of frames to be rendered.
 * - Makes sure that at least bufferSize frames are available in the queue to
 * mitigate the variation in frame generation time.
 * - Pops textures according to the frame timestamps
 * - Displays the texture on the given screen
 */
class OpenGLRenderer : public VideoStitch::Core::PanoRenderer {
 public:
  OpenGLRenderer(GLFWwindow* window, int _rollingWindowSize, int _width, int _height)
      : state(disabled),
        previousTick(0),
        tickCount(0),
        inputRate(3000),
        outputRate(3000),
        flushFrames(false),
        rollingWindowSize(_rollingWindowSize),
        width(_width),
        height(_height),
        currentContext(nullptr),
        outputWindow(nullptr),
        outputLoop(&OpenGLRenderer::loop, this) {
    glfwSetErrorCallback(error);
    createShaders(window);
  }

  virtual ~OpenGLRenderer() { stop(); }

  virtual std::string getName() const { return "OpenGLRenderer"; }

  void stop() {
    if (setState(stopping)) {
      outputLoop.join();
      std::lock_guard<std::mutex> _(framesMutex);
      clearQueues();
    }
  }

  /** @brief Called by the output thread when a new frame is ready to be rendered
   *
   *  @param surf: surface
   *  @param timestamp: render timestamp
   */
  virtual void render(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface> surf, mtime_t timestamp) {
    pushFrame(surf.get(), timestamp);
  }

  /** @brief Enables or disable the display
   */
  void enableOutput(bool enable) {
    if (enable) {
      setState(enabling, enabled);
    } else {
      setState(disabling, disabled);
    }
  }

  /** @brief Sets the viewport for the display
   */
  void setViewport(int w, int h) {
    outputWidth = w;
    outputHeight = h;
  }

  void setRefreshRate(int input, int output) {
    inputRate = input;
    outputRate = output;
  }

  /** @brief Sets the output window.
   * @note The window is created Python side
   */
  void setOutputWindow(GLFWwindow* window) {
    outputWindow = window;
    setState(starting, enabled);
  }

  virtual void renderCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf, mtime_t date) {}

  virtual void renderEquiangularCubemap(std::shared_ptr<VideoStitch::Core::CubemapOpenGLSurface> surf, mtime_t date) {}

 private:
  enum State {
    none,
    starting,
    enabling,
    enabled,
    disabling,
    disabled,
    stopping,
  };

  bool isState(State _state) {
    std::lock_guard<std::mutex> _(stateMutex);
    return state == _state;
  }

  static void error(int code, const char* message) {
    Logger::error(GLFWtag) << message << " (" << code << ")" << std::endl;
  }

  void clearQueues() {
    std::queue<Frame>().swap(acquiredFrames);
    std::queue<Frame>().swap(releasedFrames);
  }

  /** @brief Changes internal automata state, eventually waiting for it to reach a final sate.
   *  @return true if state has changed
   */
  bool setState(State newState, State finalState = none) {
    std::lock_guard<std::mutex> _(setStateMutex);

    std::unique_lock<std::mutex> lock(stateMutex);
    if (state != newState) {
      state = newState;
    } else {
      return false;
    }

    if (finalState != none) {
      stateCV.wait(lock, [this, finalState] { return state == finalState; });
    }
    return true;
  }

  /** @brief Creates shaders and objects used for rendering.
   */
  void createShaders(GLFWwindow* window) {
    int err;

    glfwMakeContextCurrent(window);

    glewInit();

    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    const char* vs =
        " \n\
    #version 120 \n\
    attribute vec3 in_Position; \n\
    attribute vec2 in_TextureCoord; \n\
    varying vec2 pass_TextureCoord; \n\
    void main() { \n\
        gl_Position = gl_ModelViewProjectionMatrix * vec4(in_Position, 1.0); \n\
        pass_TextureCoord = in_TextureCoord; \n\
    }";

    int vsLength = strlen(vs);
    glShaderSource(vertexShader, 1, &vs, &vsLength);

    const char* fs =
        " \n\
    #version 120 \n\
    uniform sampler2D sampler; \n\
    in vec2 pass_TextureCoord; \n\
    void main() { \n\
        gl_FragColor = texture2D(sampler, pass_TextureCoord); \n\
    }";

    int fsLength = strlen(fs);
    glShaderSource(fragmentShader, 1, &fs, &fsLength);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    pos = glGetAttribLocation(program, "in_Position");
    texcoord = glGetAttribLocation(program, "in_TextureCoord");

    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    const GLfloat data[] = {
        -1, -1, 0, 0, 1, 1, -1, 0, 1, 1, -1, 1, 0, 0, 0, -1, 1, 0, 0, 0, 1, -1, 0, 1, 1, 1, 1, 0, 1, 0,
    };

    glGenBuffers(1, &vb);
    glGenTextures(1, &texture);
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glfwMakeContextCurrent(nullptr);
  }

  /** @brief Binds the shader for the current context.
   */
  void setShaders() {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glUseProgram(program);
    glBindBuffer(GL_ARRAY_BUFFER, vb);

    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GL_FLOAT), 0);
    glEnableVertexAttribArray(texcoord);
    glVertexAttribPointer(texcoord, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GL_FLOAT), (void*)(3 * sizeof(GL_FLOAT)));

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
  }

  /** @brief Called by the stiching thread to push a frame in the queue
   */
  void pushFrame(VideoStitch::Core::PanoOpenGLSurface* surface, mtime_t timestamp) {
    if (!isState(enabled)) {
      return;
    }

    // Push frame for rendering
    std::lock_guard<std::mutex> lock(framesMutex);
    acquiredFrames.push(Frame(surface));
  }

  /** @brief Called by the rendering thread to pop a frame from the queue and set the corresponding texture
   */
  void popFrame() {
    VideoStitch::Core::PanoOpenGLSurface* surface = nullptr;
    {
      std::lock_guard<std::mutex> _(framesMutex);

      if (flushFrames) {
        flushFrames = false;
        clearQueues();
      }

      if (acquiredFrames.size() > rollingWindowSize) {
        // Checks if the frame must be displayed for this tick
        int64_t tick = (inputRate * tickCount) / outputRate;
        int64_t diff = tick - previousTick;
        while (diff > 0 && !acquiredFrames.empty()) {
          diff--;
          // Sets that frame for display
          Frame frame = acquiredFrames.front();
          surface = frame.surface;
          acquiredFrames.pop();

          // Delay-release the frame to make it availabe back to the stitcher.
          // The delay prevents the stitcher to immediatly overwrite the frame that is going to be displayed.
          releasedFrames.push(frame);
          if (releasedFrames.size() > rollingWindowSize) {
            releasedFrames.pop();
          }
        }
        previousTick = tick;
        tickCount++;
      }
    }

    // If a new texture has been popped, binds that texture for drawing
    if (surface != nullptr) {
      glBindTexture(GL_TEXTURE_2D, texture);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, surface->pixelbuffer);
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    }
  }

  /** @brief Draws the texture.
   */
  void draw() {
    popFrame();
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLES, 0, 6);
  }

  /** @brief Unbinds current context.
   */
  void clearContext(bool cleanup = false) {
    if (currentContext) {
      glClearColor(0, 0, 0, 0);
      glClear(GL_COLOR_BUFFER_BIT);
      glfwSwapBuffers(outputWindow);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindTexture(GL_TEXTURE_2D, 0);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      if (cleanup) {
        glDeleteBuffers(1, &vb);
        glDeleteTextures(1, &texture);
        glUseProgram(0);
        glDeleteProgram(program);
      }
      glfwMakeContextCurrent(nullptr);
      currentContext = nullptr;
    }
  }

  void changeState(State newState) {
    state = newState;
    stateCV.notify_one();  // Keep the notify before the unlock
    stateMutex.unlock();
  }

  /** @brief Rendering automata loop.
   */
  void loop() {
    bool stop = false;

    while (!stop) {
      stateMutex.lock();

      switch (state) {
        case starting:
          glfwMakeContextCurrent(outputWindow);
          currentContext = outputWindow;
          glfwSwapInterval(1);
          setShaders();

        case enabling:
          if (currentContext == nullptr) {
            glfwMakeContextCurrent(outputWindow);
            currentContext = outputWindow;
          }
          glViewport(0, 0, outputWidth, outputHeight);
          flushFrames = true;

        case enabled:
          changeState(enabled);
          draw();
          glfwSwapBuffers(outputWindow);
          break;

        case disabling:
          clearContext();

        case disabled:
          changeState(disabled);
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          break;

        case stopping:
          clearContext(true);
          stop = true;
          changeState(stopping);
          break;
      }
    }
    std::this_thread::yield();
  }

 private:
  struct Frame {
    VideoStitch::Core::PanoOpenGLSurface* surface;

    Frame(VideoStitch::Core::PanoOpenGLSurface* s) : surface(s) { surface->acquire(); }

    Frame(const Frame& frame) : surface(frame.surface) { surface->acquire(); }

    ~Frame() { surface->release(); }
  };

  int width, height;
  int rollingWindowSize;
  std::queue<Frame> acquiredFrames;
  std::queue<Frame> releasedFrames;
  std::mutex framesMutex;
  bool flushFrames;
  int64_t inputRate;
  int64_t outputRate;
  int64_t previousTick;
  int64_t tickCount;

  // Display
  GLFWwindow* outputWindow;
  GLFWwindow* currentContext;
  int outputWidth, outputHeight;
  std::mutex setStateMutex;
  std::mutex stateMutex;
  std::condition_variable stateCV;
  State state;
  std::thread outputLoop;

  GLuint vertexShader, fragmentShader;
  GLuint program;
  GLuint pos, texcoord;
  GLuint vb;
  GLuint texture;
};
