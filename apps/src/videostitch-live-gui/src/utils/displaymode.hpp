// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef DISPLAYMODE_HPP
#define DISPLAYMODE_HPP

#include "libvideostitch/plugin.hpp"
#include <QString>
#include <vector>

/**
 * @brief Converts a struct of type DisplayMode into a readeable string
 * @param mode A struct representing the DisplayMode
 * @return A string in the form: 1080x720 30fps (interleaved)
 */
QString displayModeToString(const VideoStitch::Plugin::DisplayMode& mode);

/**
 * @brief Compare 2 display modes to display them nicely
 * @param lhs First display mode to compare
 * @param rhs Second display mode to compare
 */
bool lessThan(const VideoStitch::Plugin::DisplayMode& lhs, const VideoStitch::Plugin::DisplayMode& rhs);

Q_DECLARE_METATYPE(VideoStitch::Plugin::DisplayMode);

#endif  // DISPLAYMODE_HPP
