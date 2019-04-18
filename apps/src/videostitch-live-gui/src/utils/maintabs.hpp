// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <QString>

namespace GuiEnums {

// Main tabs
enum class Tab : int { TabSources = 0, TabOutPut = 1, TabInteractive = 2, TabConfiguration = 3, TabCount };

QString getTabName(Tab tab);
QString getPixmapPath(Tab tab, bool enabled);
}  // namespace GuiEnums
