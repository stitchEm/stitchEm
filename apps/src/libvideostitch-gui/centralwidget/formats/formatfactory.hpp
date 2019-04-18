// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QString>
#include "format.hpp"

/**
 * @brief The FormatFactory class creates formats using a given string.
 */
class VS_GUI_EXPORT FormatFactory {
 public:
  static Format* create(const QString& key, QWidget* const parent);
};
