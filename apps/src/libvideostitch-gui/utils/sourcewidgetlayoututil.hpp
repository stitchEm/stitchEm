// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SOURCEWIDGETLAYOUTUTIL_HPP
#define SOURCEWIDGETLAYOUTUTIL_HPP

#include "libvideostitch-gui/common.hpp"

namespace SourceWidgetLayoutUtil {

const int MaxColumnsNumber = 3;

VS_GUI_EXPORT int getColumnsNumber(int itemsNumber);
VS_GUI_EXPORT int getLinesNumber(int itemsNumber);
VS_GUI_EXPORT int getItemLine(int itemId, int columnsNumber);
VS_GUI_EXPORT int getItemColumn(int itemId, int columnsNumber);
}  // namespace SourceWidgetLayoutUtil

#endif  // SOURCEWIDGETLAYOUTUTIL_HPP
