// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef FILES_HPP
#define FILES_HPP

#include "common-config.hpp"

#include <QList>

class QString;
class QDir;

/**
 * @brief Class used to have an enum to list all the types of files.
 * It can also be used to manage them
 */
class VS_COMMON_EXPORT File {
 public:
  /**
   * @brief List of the different file types used in VideoStitch
   */
  enum Type { PTV = 0x01, VAH = 0x02, CALIBRATION = 0x03, VIDEO = 0x04, STILL_IMAGE = 0x05, UNKNOWN = 0x00 };
  typedef Type quint8;

  /**
   * @brief Gets the file type from a file (assuming its extension is representing its type)
   *        In can be using using either a full path or just the file name
   *
   * @param file File you want to obtain the type.
   * @return Type of the file.
   */
  static Type getTypeFromFile(const QString &file);
  static QDir getFirstCommonDirectory(QList<QDir> directories);
  static QString suffixIfAlreadyExists(const QString &basename, const QString &extension);
  /**
   * @brief Gets the folder where you can store persistent files/settings (application scope)
   * @return Returns C:\Program Data\app_name on windows, or ~/.app_name/ on Unix systems.
   */
  static QString getAppDataFolder();
  /**
   * @brief Gets the folder where you can store persistent files/settings (organization scope)
   * @return Returns C:\Program Data\VideoStitch on windows, or ~/.VideoStitch/ on Unix systems.
   */
  static QString getVSDataFolder();
  /**
   * @brief Returns the location of the "Documents" directory of the user using VS.
   * @return Location of the "documents" directory.
   */
  static QString getDocumentsLocation();
  /**
   * @brief Shows the given path in the file explorer and highlights it.
   * @param parent Parent QWidget (needed for heal allocation).
   * @param path Path to open and select in the explorer.
   */
  static void showInShellExporer(const QString &pathIn);

  static QString strippedName(const QString &fullFileName);

  static bool fileExists(const QString &filename);

  static QString normalizePath(const QString &filename);
};

#endif
