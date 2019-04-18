// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include <EGL/egl.h>
#include <GLES2/gl2.h>

class NvEGLUtil;
static int engine_init_display(struct NvEGLUtil *engine, ANativeWindow *window);
static void engine_draw_frame(struct NvEGLUtil *engine);
static void engine_term_display(struct NvEGLUtil *engine);

class NvEGLUtil {
 public:
  static NvEGLUtil *create() { return (NvEGLUtil *)calloc(1, sizeof(NvEGLUtil)); }
  bool isReadyToRender(bool allocateIfNeeded = false) { return true; }
  bool checkWindowResized() { return false; }
  int32_t getWidth() { return width; }
  int32_t getHeight() { return height; }
  bool swap() {
    engine_draw_frame(this);
    return true;
  }
  bool setWindow(ANativeWindow *window) {
    if (window == nullptr) {
      engine_term_display(this);
      return true;
    } else {
      return engine_init_display(this, window) >= 0;
    }
  }

  // EGL Display, surface, and context
  EGLDisplay display;
  EGLSurface surface;
  EGLContext context;
  // States and touch locations
  bool animating;
  int width;
  int height;
  int x_touch;
  int y_touch;
};

/**
 * Initialize an EGL context for the current display
 */
static int engine_init_display(struct NvEGLUtil *engine, ANativeWindow *window) {
  // initialize OpenGL ES and EGL
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  eglInitialize(display, 0, 0);

  // Specify the attributes of the desired configuration.
  // We select an EGLConfig with at least 8 bits per color component
  // that is compatible with on-screen windows.
  const EGLint attribs[] = {EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8,
                            EGL_NONE};

  // Here, the application chooses the configuration it desires.
  // eglChooseConfig in general returns all the configurations compatible
  // with the attributes passed. In this sample, we have a very simplified
  // selection process, where we pick the first EGLConfig that matches
  // our criteria (by setting the third argument to 1).
  EGLConfig config;
  EGLint numConfigs;
  eglChooseConfig(display, attribs, &config, 1, &numConfigs);

  // EGL_NATIVE_VISUAL_ID is an attribute of the EGLConfig that is
  // guaranteed to be accepted by ANativeWindow_setBuffersGeometry().
  // We can use it to make the ANativeWindow buffers to match.
  EGLint format;
  eglGetConfigAttrib(display, config, EGL_NATIVE_VISUAL_ID, &format);

  // Set a native Android window to have the format configured by EGL
  ANativeWindow_setBuffersGeometry(window, 0, 0, format);

  // Create EGL surface and context
  int attrib_list[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};

  EGLSurface surface = eglCreateWindowSurface(display, config, window, NULL);
  EGLContext context = eglCreateContext(display, config, NULL, attrib_list);

  // Use the surface and context we just created and configure the engine
  if (eglMakeCurrent(display, surface, surface, context) == EGL_FALSE) {
    VideoStitch::Logger::warning(ANDROIDTag) << "Unable to eglMakeCurrent" << std::endl;
    return -1;
  }

  // Get width and height of the surface
  EGLint w, h;
  eglQuerySurface(display, surface, EGL_WIDTH, &w);
  eglQuerySurface(display, surface, EGL_HEIGHT, &h);

  // Store the app variables so the callbacks can access the data
  engine->display = display;
  engine->context = context;
  engine->surface = surface;
  engine->width = w;
  engine->height = h;

  // Initialize GL state
  glEnable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);

  return 0;
}

/**
 * Tear down the EGL context currently associated with the display
 */
static void engine_term_display(struct NvEGLUtil *engine) {
  if (engine->display != EGL_NO_DISPLAY) {
    eglMakeCurrent(engine->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (engine->context != EGL_NO_CONTEXT) {
      eglDestroyContext(engine->display, engine->context);
    }
    if (engine->surface != EGL_NO_SURFACE) {
      eglDestroySurface(engine->display, engine->surface);
    }
    eglTerminate(engine->display);
  }
  engine->animating = false;
  engine->display = EGL_NO_DISPLAY;
  engine->context = EGL_NO_CONTEXT;
  engine->surface = EGL_NO_SURFACE;
}

/**
 * Draw the current frame on the display
 */
static void engine_draw_frame(struct NvEGLUtil *engine) {
  if (engine->display == NULL) {
    return;
  }  // No display

  // Set the clear color based on the touch location from the engine
  glClearColor(((float)engine->x_touch) / engine->width,   // Red channel
               0,                                          // Green channel
               ((float)engine->y_touch) / engine->height,  // Blue channel
               1);                                         // Alpha channel
  // Clear the screen to the color we just set
  glClear(GL_COLOR_BUFFER_BIT);

  // Swap the buffers, which indicates we're done with rendering this frame
  eglSwapBuffers(engine->display, engine->surface);
}
