// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "libvideostitch-gui/common.hpp"

#include <QVector>
#include <QString>

#define FRAME_EXTENSION QString("-%frame%")

/**
 * @brief The ExtensionHandler is a helper class to set and manage file format names.
 */
class VS_GUI_EXPORT ExtensionHandler {
 public:
  void init();
  ExtensionHandler();
  virtual ~ExtensionHandler();

  /**
   * @brief Returns a string with the filename and the extension.
   * @param filename The output filename.
   * @param format The output format.
   * @return The filename containing the extension.
   */
  virtual QString handle(const QString &filename, const QString &format) const;

  /**
   * @brief Given a full filename and a format, returns the file name without the associated extension.
   * @param filename The filename with its extension.
   * @param format The output format.
   * @return Returns the filename without the formated extension.
   */
  virtual QString stripBasename(const QString &inputText, const QString &format) const;

 protected:
  /**
   * @brief Adds a specific handler.
   * @param handler An extension handler.
   */
  void addHandler(ExtensionHandler *handler);
  QString extension;
  QString format;

 private:
  QVector<ExtensionHandler *> handlers;
};
