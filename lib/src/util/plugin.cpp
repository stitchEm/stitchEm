// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "plugin.hpp"

#include "libvideostitch/inputFactory.hpp"

#ifdef _MSC_VER
#include <codecvt>
#include <Windows.h>
#include <VersionHelpers.h>
#endif
#include <iostream>
#include <cassert>
#include <memory>

#ifndef _MSC_VER
#include <dirent.h>
#include <dlfcn.h>
#else
#include "libvideostitch/win32/dirent.h"
#endif

#ifdef _MSC_VER  // win32
namespace {
class DllDirectoryGuard {
 public:
  DllDirectoryGuard(const std::string& dir, bool mandatory = true) : cookie(nullptr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::wstring wdir = converter.from_bytes(dir);

    typedef DLL_DIRECTORY_COOKIE(WINAPI * PADD)(PCWSTR);
    PADD pADD = (PADD)GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")), "AddDllDirectory");
    if (pADD) {
      cookie = pADD(wdir.c_str());
    } else {
      VideoStitch::Logger::get(VideoStitch::Logger::Error)
          << "Error: AddDllDirectory is not supported on your Windows" << std::endl;
    }

    if (!cookie) {
      LPTSTR lpErrorText = NULL;
      FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_IGNORE_INSERTS, 0,
                    GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), lpErrorText, 0, NULL);
      if (mandatory) {
        VideoStitch::Logger::get(VideoStitch::Logger::Error)
            << "Error: could not set the module directory " << dir << " : "
            << (lpErrorText ? lpErrorText : "unknown error") << std::endl;
      } else {
        VideoStitch::Logger::get(VideoStitch::Logger::Warning)
            << "Warning: could not set the module directory " << dir << " : "
            << (lpErrorText ? lpErrorText : "unknown warning") << std::endl;
      }
      LocalFree(lpErrorText);
    }
  }
  ~DllDirectoryGuard() {
    if (cookie) {
      typedef BOOL(WINAPI * PRDD)(DLL_DIRECTORY_COOKIE);
      PRDD pRDD = (PRDD)GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")), "RemoveDllDirectory");
      if (pRDD) {
        pRDD(cookie);
      } else {
        VideoStitch::Logger::get(VideoStitch::Logger::Error)
            << "Error: RemoveDllDirectory is not supported on your Windows" << std::endl;
      }
    }
  }

  bool ok() const { return cookie != NULL; }

 private:
  DLL_DIRECTORY_COOKIE cookie;
};
}  // namespace
#endif

namespace VideoStitch {
namespace Plugin {
std::mutex pluginsMutex;

template <typename T>
typename VectorInstanceManaged<T>::InstanceVector& VectorInstanceManaged<T>::instances() {
  static InstanceVector lReturn;
  return lReturn;
}

template class VectorInstanceManaged<VSDiscoveryPlugin>;
template class VectorInstanceManaged<VSReaderPlugin>;
template class VectorInstanceManaged<VSProbeReaderPlugin>;
template class VectorInstanceManaged<VSWriterPlugin>;

// Track the reader instances only to delete them at program exit
// This should not be used to get access to the plugins,
// use ::Instances() instead
static std::vector<std::unique_ptr<VSDiscoveryPlugin>> freeDiscoveryAtExit;
static std::vector<std::unique_ptr<VSReaderPlugin>> freeReaderAtExit;
static std::vector<std::unique_ptr<VSProbeReaderPlugin>> freeProbeReaderAtExit;
static std::vector<std::unique_ptr<VSWriterPlugin>> freeWriterAtExit;

VSProbeReaderPlugin::VSProbeReaderPlugin(char const* name, HandlesFnT handlesFn, CreateFnT createFn, ProbeFnT pProbeFn)
    : VSReaderPlugin(name, handlesFn, createFn), VectorInstanceManaged<VSProbeReaderPlugin>(this), probeFn(pProbeFn) {}

VSProbeReaderPlugin::ProbeResult VSProbeReaderPlugin::probe(const std::string& p) const { return probeFn(p); }

namespace {
/**
 * List files with a certain extension in a directory.
 * @param directory: directory to list.
 * @param ext: the file prefix (may begin with '.').
 * @return A list of matching files.
 */
std::vector<std::string> listDirectory(const std::string& directory, const std::string& ext) {
  size_t extSize = ext.size();
  std::vector<std::string> filenames;
  DIR* dir = opendir(directory.c_str());
  if (dir) {
    /*print all the files and directories within directory*/
    struct dirent* ent = readdir(dir);
    while (ent) {
      std::string str(ent->d_name);
      ent = readdir(dir);
      if (str.size() > extSize) {
        if (str.substr(str.size() - extSize) == ext) {
          Logger::get(Logger::Verbose) << "Plugin: found potential plugin \"" << str << "\"" << std::endl;
          filenames.push_back(str);
          continue;
        }
      }
      Logger::get(Logger::Debug) << "Plugin: discard potential plugin \"" << str << "\"" << std::endl;
    }
    closedir(dir);
  }

  return filenames;
}

#ifdef _MSC_VER
typedef HMODULE handle_type;

std::string getLastErrorAsString() {
  // Get the error message, if any.
  DWORD errorMessageID = GetLastError();
  if (errorMessageID == 0) {
    return std::string();  // No error message has been recorded
  }

  LPSTR messageBuffer = nullptr;
  size_t size =
      FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                     errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

  std::string message(messageBuffer, size);
  // Free the buffer.
  LocalFree(messageBuffer);

  return message;
}
#else
typedef void* handle_type;
#endif

/**
 * Looks for symbol named p_name matching p_fct's signature.
 * \returns true if symbol loading has succeeded. */
template <typename _Fct>
bool loadSymbol(/** [out] target function */ _Fct& p_fct,
                /** [in] shared library handle */ handle_type p_handle,
                /** [in] symbol name */ const std::string& p_name) {
  bool l_return = false;
#ifdef _MSC_VER
  p_fct = (_Fct)GetProcAddress(p_handle, p_name.c_str());
  if (p_fct) {
    l_return = true;
  }
#else
  *(void**)(&p_fct) = dlsym(p_handle, p_name.c_str());
  const char* l_error = 0;
  if ((l_error = dlerror()) == 0)
    l_return = true;
  else {
    Logger::get(Logger::Verbose) << "Failed to load symbol: " << l_error << std::endl;
  }
#endif
  return l_return;
}

/** Try to load plug-ins from p_directory/p_sharedLib.
 *
 * \returns true if at least one plug-in has been created.
 *
 * \note On Windows, p_directory is not
 * passed. SetDllDirectory(p_directory) is supposed to have been
 * called successfully. */
bool loadPlugin(const std::string& p_directory, const std::string& p_sharedLib) {
#ifndef __clang_analyzer__
  bool l_return = false;
  std::string fullPath = p_directory;
#ifdef _MSC_VER
  std::replace(fullPath.begin(), fullPath.end(), '/', '\\');
  fullPath = fullPath + '\\' + p_sharedLib;
#else
  fullPath = fullPath + '/' + p_sharedLib;
#endif
  Logger::get(Logger::Verbose) << "Trying to load plugin " << fullPath << "." << std::endl;

#ifndef _MSC_VER
  dlerror();
#endif
  handle_type libHandle =
#ifdef _MSC_VER
      LoadLibraryEx(fullPath.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
#else
      dlopen(fullPath.c_str(), RTLD_NOW)
#endif
      ;
  if (!libHandle) {
#ifdef _MSC_VER
    std::string l_msg = getLastErrorAsString();
#else
    std::string l_msg = dlerror();
#endif
    Logger::get(Logger::Error) << "Debug: could not load plug-in \"" << fullPath << "\": " << l_msg << std::endl;
    return l_return;
  }
  VSDiscoveryPlugin::DiscoverFnT discoverFn = NULL;
  VSReaderPlugin::HandlesFnT handleRFn = NULL;
  VSReaderPlugin::CreateFnT createRFn = NULL;
  VSWriterPlugin::HandlesFnT handleWFn = NULL;
  VSWriterPlugin::CreateFnT createWFn = NULL;
  if (loadSymbol(handleRFn, libHandle, "handleReaderFn") && loadSymbol(createRFn, libHandle, "createReaderFn")) {
    VSProbeReaderPlugin::ProbeFnT probeRFn = NULL;
    l_return = true;
    if (loadSymbol(probeRFn, libHandle, "probeReaderFn")) {
      freeProbeReaderAtExit.emplace_back(new VSProbeReaderPlugin(p_sharedLib.c_str(), handleRFn, createRFn, probeRFn));
    } else {
      freeReaderAtExit.emplace_back(new VSReaderPlugin(p_sharedLib.c_str(), handleRFn, createRFn));
    }
  }
  if (loadSymbol(handleWFn, libHandle, "handleWriterFn") && loadSymbol(createWFn, libHandle, "createWriterFn")) {
    l_return = true;
    freeWriterAtExit.emplace_back(new VSWriterPlugin(p_sharedLib.c_str(), handleWFn, createWFn));
  }
  if (loadSymbol(discoverFn, libHandle, "discoverFn")) {
    l_return = true;
    freeDiscoveryAtExit.emplace_back(discoverFn());
  }
  if (!l_return) {
#ifdef _MSC_VER
    FreeLibrary(libHandle);
#else
    dlclose(libHandle);
#endif
  }
  return l_return;
#else
  return false;
#endif
}
}  // namespace
#ifdef _MSC_VER
#define VS_PLUGINS_EXT ".dll"
#else
#ifdef __APPLE__  // macosx
#define VS_PLUGINS_EXT ".dylib"
#else  // linux
#define VS_PLUGINS_EXT ".so"
#endif
#endif

int loadPlugins(const std::string& dir) {
  std::unique_lock<std::mutex> lock(pluginsMutex);
  int addedPlugins = 0;
  Logger::get(Logger::Verbose) << "Scanning '" << dir << "' for plug-ins." << std::endl;
#ifdef _MSC_VER  // win32
  DllDirectoryGuard dirGuard(dir);
  if (!dirGuard.ok()) {
    return 0;
  }
#endif
  std::vector<std::string> sharedLibs = listDirectory(dir, VS_PLUGINS_EXT);
  for (std::vector<std::string>::const_iterator l_it = sharedLibs.begin(), l_last = sharedLibs.end(); l_it != l_last;
       ++l_it) {
    if (loadPlugin(dir, *l_it)) {
      ++addedPlugins;
    }
  }
  Logger::get(Logger::Verbose) << addedPlugins << " plug-ins loaded from " << dir << "." << std::endl;
  return addedPlugins;
}

VSDiscoveryPlugin::~VSDiscoveryPlugin() {}

AutoDetection::~AutoDetection() {}

std::vector<DiscoveryDevice> VSDiscoveryPlugin::devices() {
  std::vector<DiscoveryDevice> devices = inputDevices();
  for (const auto& device : outputDevices()) {
    // Because some devices can be input AND output
    if (std::find(devices.cbegin(), devices.cend(), device) == devices.cend()) {
      devices.push_back(device);
    }
  }
  return devices;
}

VSReaderPluginBase::~VSReaderPluginBase() {}

VSWriterPluginBase::~VSWriterPluginBase() {}

}  // namespace Plugin
}  // namespace VideoStitch
