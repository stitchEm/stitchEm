// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef FILESYSTEM_HPP_
#define FILESYSTEM_HPP_

#include <string>

namespace VideoStitch {
namespace Util {

/**
 * Returns the base directory for a file path.
 * @param path file name
 * @param dir If not null, contains the directory name, without trailing separator, on output.
 * @param filename If not null, contains the directory name, without trailing separator, on output.
 */
void getBaseDir(const std::string& path, std::string* dir, std::string* filename);

/**
 * A directory lister. No specific order.
 * Not thread safe, but several instances can be used concurrently.
 */
class DirectoryLister {
 public:
  explicit DirectoryLister(const std::string& directory);

  ~DirectoryLister();

  /**
   * Returns true if the directory was successfully opened.
   */
  bool ok() const;

  /**
   * Returns true if there are no more files.
   */
  bool done() const;

  /**
   * Returns the current file name, relative to the base directory.
   */
  const std::string& file() const;

  /**
   * Goes to the next file.
   */
  void next();

 private:
  void* dirp;
  const bool isOk;
  std::string curFilename;
};

}  // namespace Util
}  // namespace VideoStitch
#endif
