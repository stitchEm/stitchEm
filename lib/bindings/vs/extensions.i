//LibVideoStitch extensions

%{
#include "../../include/libvideostitch/ptv.hpp"
%}
%include "../../include/libvideostitch/ptv.hpp"

%extend VideoStitch::Ptv::Value {

  void setObject(const VideoStitch::Ptv::Value& val) {
    $self->asObject() = val;
  }

  const std::string& getString() const {
    return $self->asString();
  }

  void setString(const std::string& val) {
    $self->asString() = val;
  }

  int64_t getInt() const {
    return $self->asInt();
  }

  void setInt(const int64_t val) {
    $self->asInt() = val;
  }

  bool getBool() const {
    return $self->asBool();
  }

  void setBool(const bool val) {
    $self->asBool() = val;
  }


  float getDouble() const {
    return $self->asDouble();
  }

  void setDouble(const float val) {
    $self->asDouble() = val;
  }

  int getNumbItems() const {
    return $self->asList().size();
  }

  Value* getItem(const int item) {
    return $self->asList().at(item);
  }
}

//Logger to File extention

%{
#include "../../include/libvideostitch/logging.hpp"
#include <fstream>



class timeStampStreambuf : public std::streambuf {
public:
  timeStampStreambuf(std::basic_ios<char> * _stream, const char * _suffix) :
    suffix(_suffix),
    newline(true)
  {
    sink = _stream->rdbuf();
  }

protected:

  int_type overflow(int_type c = traits_type::eof()) {
    if (traits_type::eq_int_type(c, traits_type::eof()))
      return traits_type::not_eof(c);

    if (newline) {
      struct timeval tmnow;
      struct tm * tm;
      gettimeofday(&tmnow, NULL);
      tm = localtime(&tmnow.tv_sec);
      char buffer[80], mbuffer[16];
      strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", tm);
      strcat(buffer, ",");
      sprintf(mbuffer, "%03d ", (int)tmnow.tv_usec / 1000);
      strcat(buffer, mbuffer);
      std::ostream str(sink);
      if (! (str << buffer << suffix) ) {
        return traits_type::eof();
      }
    }
    newline = traits_type::to_char_type(c) == '\n';
    return sink->sputc(c);
  }

  int sync() {
    std::streambuf::int_type result = this->overflow(traits_type::eof());
    sink->pubsync();
    return traits_type::eq_int_type(result, traits_type::eof()) ? -1 : 0;
  }


private:
  const char * suffix;
  std::streambuf * sink;
  bool newline;
};


std::ofstream* fileStream = nullptr;

%}
%include "../../include/libvideostitch/logging.hpp"


%extend VideoStitch::Logger {

  static void setLogFile(const char* filename, bool rotate) {
    // Note : We are leaking a few pointers here
    std::ofstream* oldFileStream = fileStream;
    fileStream = new std::ofstream();
    fileStream->open(filename, std::ofstream::out | (rotate ? std::ofstream::app: std::ofstream::trunc));
    VideoStitch::Logger::setLogStream(VideoStitch::Logger::LogLevel::Error,   new std::ostream(new timeStampStreambuf(fileStream, "ERROR  ")));
    VideoStitch::Logger::setLogStream(VideoStitch::Logger::LogLevel::Warning, new std::ostream(new timeStampStreambuf(fileStream, "WARNING  ")));
    VideoStitch::Logger::setLogStream(VideoStitch::Logger::LogLevel::Info,    new std::ostream(new timeStampStreambuf(fileStream, "INFO  ")));
    VideoStitch::Logger::setLogStream(VideoStitch::Logger::LogLevel::Verbose, new std::ostream(new timeStampStreambuf(fileStream, "VERBOSE  ")));
    VideoStitch::Logger::setLogStream(VideoStitch::Logger::LogLevel::Debug,   new std::ostream(new timeStampStreambuf(fileStream, "DEBUG  ")));
    if (oldFileStream) {
        delete oldFileStream;
    }
  }

  static void log(LogLevel level, const char* message) {
    VideoStitch::Logger::get(level) << message << std::endl;
  }

}

/*
 *
 *  Extends processor intensive functions to release the Python GIL
 *
 */

%{
#include "../../include/libvideostitch/output.hpp"

#include "../../include/libvideostitch/audio.hpp"

using namespace VideoStitch;
using namespace VideoStitch::Audio;
using namespace VideoStitch::Output;

#include "../../src/output/profilingOutput.hpp"
%}
%extend VideoStitch::Output::Output {

  static Potential<Output> createNoGIL(
                                        const Ptv::Value& writerConfig,
                                        const std::string& name,
                                        unsigned width, unsigned height,
                                        FrameRate framerate,
                                        int outRate, const char* outDepth, const char* outLayout) {
    PyThreadState *_save;
    _save = PyEval_SaveThread();
    VideoStitch::Potential<VideoStitch::Output::Output> r = create(writerConfig, name, width, height,
                                         framerate,
                                         getSamplingRateFromInt(outRate),
                                         getSamplingDepthFromString(outDepth),
                                         getChannelLayoutFromString(outLayout));
    PyEval_RestoreThread(_save);
    return r;
  }

  static Potential<Output> profiling(const std::string& name,
                                     unsigned width, unsigned height,
                                     FrameRate framerate,
                                     VideoStitch::PixelFormat fmt)
  {
    PyThreadState *_save = PyEval_SaveThread();
    VideoStitch::Output::ProfilingWriter *r =
        new VideoStitch::Output::ProfilingWriter(name, width, height, framerate, fmt);
    PyEval_RestoreThread(_save);
    return (VideoStitch::Potential<VideoStitch::Output::Output>)r;
  }

  static void reset(VideoStitch::Output::Output * writer)
  {
    PyThreadState *_save = PyEval_SaveThread();
    ((VideoStitch::Output::ProfilingWriter *) writer->getVideoWriter())->reset();
    PyEval_RestoreThread(_save);
  }

  static double getFps(VideoStitch::Output::Output * writer)
  {
    PyThreadState *_save = PyEval_SaveThread();
    double fps = ((VideoStitch::Output::ProfilingWriter *) writer->getVideoWriter())->getFps();
    PyEval_RestoreThread(_save);
    return fps;
  }
}

%{
#include "../../include/libvideostitch/controller.hpp"
#include "../../include/libvideostitch/stitchOutput.hpp"

// Impossible to get SWIG extend the stitcher class because of the complexity of templates. Had to fall back to a global function.

VideoStitch::Status stitchAndExtractNoGIL(
  VideoStitch::Core::PotentialController& controller,
  VideoStitch::Core::StitchOutput* output,
  std::vector<VideoStitch::Core::ExtractOutput*> extracts,
  VideoStitch::Core::AlgorithmOutput* algo,
  bool readFrame)
{
  auto threadInfo = PyEval_SaveThread();
  auto status = controller->stitchAndExtract(output, extracts, algo, readFrame);
  PyEval_RestoreThread(threadInfo);
  if (status.getCode() == VideoStitch::Core::ControllerStatusCode::EndOfStream) {
    return { Origin::Input, ErrType::RuntimeError, "Could not load input frames, reader reported end of stream" };
  }
  return status.getStatus();
}
//Todo: consider creating generic "nogil" wrapper

%}

VideoStitch::Status stitchAndExtractNoGIL(
  VideoStitch::Core::PotentialController& controller,
  VideoStitch::Core::StitchOutput* output,
  std::vector<VideoStitch::Core::ExtractOutput*> extracts,
  VideoStitch::Core::AlgorithmOutput* algo,
  bool readFrame);





%include <std_shared_ptr.i>
%include <std_vector.i>



%shared_ptr(VideoStitch::Output::Output)
%shared_ptr(VideoStitch::Output::AudioWriter)
%shared_ptr(VideoStitch::Output::VideoWriter)
%shared_ptr(VideoStitch::Output::StereoWriter)
%shared_ptr(VideoStitch::Core::SourceRenderer)
%shared_ptr(VideoStitch::Core::PanoRenderer)
%shared_ptr(VideoStitch::Core::SourceSurface)
%template(SourceSurfaceVec) std::vector<std::shared_ptr<VideoStitch::Core::SourceSurface>>;
%shared_ptr(VideoStitch::Core::PanoSurface)
%template(PanoSurfaceVec) std::vector<std::shared_ptr<VideoStitch::Core::PanoSurface>>;
%shared_ptr(VideoStitch::Core::SourceOpenGLSurface)
%shared_ptr(VideoStitch::Core::PanoOpenGLSurface)
%shared_ptr(VideoStitch::Core::CubemapSurface)
%shared_ptr(VideoStitch::Core::CubemapOpenGLSurface)
%shared_ptr(VideoStitch::Core::Overlayer)

%{
#include <memory>
%}
%include "CppCallback.hpp"
%include "renderer.hpp"
%include "compositor.hpp"
%include "stitch_loop.hpp"
%include "../../include/libvideostitch/allocator.hpp"


%{
#include "../../include/libvideostitch/controller.hpp"
#include "../../include/libvideostitch/stitchOutput.hpp"
#include "../../include/libvideostitch/allocator.hpp"
#include "renderer.hpp"
#include "compositor.hpp"
#include "stitch_loop.hpp"

std::shared_ptr<VideoStitch::Output::Output> writerSharedPtr(VideoStitch::Output::Output* originalPtr) {
  return std::shared_ptr<VideoStitch::Output::Output>(originalPtr);
}

std::shared_ptr<VideoStitch::Output::VideoWriter> videoWriterSharedPtr(const std::shared_ptr<VideoStitch::Output::Output>& ptr) {
  if (ptr->getVideoWriter()) {
    return std::dynamic_pointer_cast<VideoStitch::Output::VideoWriter>(ptr);
  }
  return std::shared_ptr<VideoStitch::Output::VideoWriter>();
}

std::shared_ptr<VideoStitch::Output::AudioWriter> audioWriterSharedPtr(const std::shared_ptr<VideoStitch::Output::Output>& ptr) {
  if (ptr->getAudioWriter()) {
    return std::dynamic_pointer_cast<VideoStitch::Output::AudioWriter>(ptr);
  }
  return std::shared_ptr<VideoStitch::Output::AudioWriter>();
}

std::shared_ptr<VideoStitch::Core::PanoRenderer> panoRendererSharedPtr(OpenGLRenderer* originalPtr) {
  return std::shared_ptr<VideoStitch::Core::PanoRenderer>(originalPtr);
}

std::shared_ptr<VideoStitch::Core::PanoSurface> panoSurfaceSharedPtr(VideoStitch::Core::PanoSurface* ptr) {
  return std::shared_ptr<VideoStitch::Core::PanoSurface>(ptr);
}

std::shared_ptr<VideoStitch::Core::SourceSurface> sourceSurfaceSharedPtr(VideoStitch::Core::SourceSurface* ptr) {
  return std::shared_ptr<VideoStitch::Core::SourceSurface>(ptr);
}

VideoStitch::Core::PanoOpenGLSurface * getPanoOpenGLSurface(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface> ptr) {
  return ptr.get();
}

std::shared_ptr<VideoStitch::GPU::Overlayer> overlayerSharedPtr(Compositor* originalPtr) {
  return std::shared_ptr<VideoStitch::GPU::Overlayer>(originalPtr);
}

%}

std::shared_ptr<VideoStitch::Output::Output>           writerSharedPtr(VideoStitch::Output::Output*);
std::shared_ptr<VideoStitch::Output::VideoWriter> videoWriterSharedPtr(const std::shared_ptr<VideoStitch::Output::Output>&);
std::shared_ptr<VideoStitch::Output::AudioWriter> audioWriterSharedPtr(const std::shared_ptr<VideoStitch::Output::Output>&);

std::shared_ptr<VideoStitch::Core::PanoRenderer>        panoRendererSharedPtr(OpenGLRenderer*);

std::shared_ptr<VideoStitch::Core::SourceSurface> sourceSurfaceSharedPtr(VideoStitch::Core::SourceSurface*);
std::shared_ptr<VideoStitch::Core::PanoSurface> panoSurfaceSharedPtr(VideoStitch::Core::PanoSurface*);

VideoStitch::Core::PanoOpenGLSurface * getPanoOpenGLSurface(std::shared_ptr<VideoStitch::Core::PanoOpenGLSurface> ptr);

std::shared_ptr<VideoStitch::GPU::Overlayer> overlayerSharedPtr(Compositor* originalPtr);

%{
#include "algorithmListenerGIL.hpp"
%}
%include "algorithmListenerGIL.hpp"

%extend VideoStitch::Core::PanoramaDefinitionUpdater {
    std::function<Potential<PanoDefinition>(const PanoDefinition&)> getCloneUpdater() {
        // because swig won't show interface from DeferredUpdater
        return $self->getCloneUpdater();
    }
}

%{
#include "CppCallback.hpp"
#include "renderer.hpp"

GLFWwindow * castToSwigGLFWwindow(unsigned long addr) {
    return reinterpret_cast<GLFWwindow *>(addr);
}
%}

GLFWwindow * castToSwigGLFWwindow(unsigned long addr);

%extend VideoStitch::Core::StitcherOutput {
    bool removeWriterNoGIL(const std::string& name) {
    PyThreadState *_save;
    _save = PyEval_SaveThread();
    auto result = $self->removeWriter(name);
    PyEval_RestoreThread(_save);
    return result;
  }
}

%{
#include "../../include/libvideostitch/quaternion.hpp"
%}
struct YawPitchRoll {
    double yaw, pitch, roll;
};
%{
struct YawPitchRoll {
    double yaw, pitch, roll;
};
%}
%extend VideoStitch::Quaternion<double> {
  YawPitchRoll toEuler_py() {
    YawPitchRoll result;
    $self->toEuler(result.yaw, result.pitch, result.roll);
    return result;
  }
}

%include <std_array.i>
%template(std_array_u16_256) std::array<uint16_t, 256>;
