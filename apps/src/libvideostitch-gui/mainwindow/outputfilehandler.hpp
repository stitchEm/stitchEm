// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef OUTPUTFILEHANDLER_HPP
#define OUTPUTFILEHANDLER_HPP

#include <QMutex>
#include "libvideostitch-base/file.hpp"
#include "libvideostitch-base/singleton.hpp"

/**
 * @brief A thread-safe class to access the output filename.
 */
class VS_GUI_EXPORT ProjectFileHandler : public Singleton<ProjectFileHandler> {
  friend class Singleton<ProjectFileHandler>;

 public:
  QString getFilename() const;
  void setFilename(QString newFilename);
  QString getWorkingDirectory() const;
  void resetFilename();

 private:
  ProjectFileHandler();

  QString filename;
  QString workingDirectory;

  /**
   * @brief Mute protecting the output filename.
   */
  mutable QMutex filenameMutex;
};

#endif  // OUTPUTFILEHANDLER_HPP
