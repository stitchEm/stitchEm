// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "sourcewidgetlayoututil.hpp"

namespace SourceWidgetLayoutUtil {

int getColumnsNumber(int itemsNumber) {
  if (itemsNumber <= MaxColumnsNumber) {
    return itemsNumber;
  }

  return itemsNumber == MaxColumnsNumber + 1 ? (MaxColumnsNumber + 1) / 2 : MaxColumnsNumber;
}

int getLinesNumber(int itemsNumber) { return (itemsNumber - 1) / getColumnsNumber(itemsNumber) + 1; }

int getItemLine(int itemId, int columnsNumber) { return itemId / columnsNumber; }

int getItemColumn(int itemId, int columnsNumber) { return itemId % columnsNumber; }

}  // namespace SourceWidgetLayoutUtil
