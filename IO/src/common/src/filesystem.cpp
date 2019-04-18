// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "filesystem.hpp"

#ifdef _MSC_VER
#include "libvideostitch/win32/dirent.h"
#else
#include <dirent.h>
#endif

namespace VideoStitch {
namespace Util {

void getBaseDir(const std::string& path, std::string* dir, std::string* filename) {
  // TODO: handle escaping.
  const size_t found = path.find_last_of("/\\");
  if (found == std::string::npos) {
    // Assume that we were given a file, try the current directory.
    if (dir) {
      dir->assign(".");
    }
    if (filename) {
      *filename = path;
    }
  } else {
    if (dir) {
      *dir = path.substr(0, found);
    }
    if (filename) {
      *filename = path.substr(found + 1, std::string::npos);
    }
  }
}

DirectoryLister::DirectoryLister(const std::string& directory) : dirp(opendir(directory.c_str())), isOk(dirp != NULL) {}

DirectoryLister::~DirectoryLister() {
  if (dirp) {
    closedir((DIR*)dirp);
  }
}

bool DirectoryLister::ok() const { return isOk; }

bool DirectoryLister::done() const { return dirp == NULL; }

void DirectoryLister::next() {
  if (done()) {
    return;
  }
#ifdef _MSC_VER
  // TODO: make it thread-safe.
  struct dirent* pentry = readdir((DIR*)dirp);
  if (pentry == NULL) {
    closedir((DIR*)dirp);
    dirp = NULL;
    return;
  }
  curFilename = pentry->d_name;
#else
  struct dirent* success;
  struct dirent entry;
  if (readdir_r((DIR*)dirp, &entry, &success) || !success) {
    closedir((DIR*)dirp);
    dirp = NULL;
    return;
  }
  curFilename = entry.d_name;
#endif
}

const std::string& DirectoryLister::file() const { return curFilename; }

}  // namespace Util
}  // namespace VideoStitch
