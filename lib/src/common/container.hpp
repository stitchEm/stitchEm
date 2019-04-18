// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm
//
// This file contains common utilities for containers.
// It must not include any header outside of the current directory.

#ifndef CONTAINER_HPP_
#define CONTAINER_HPP_

#include <vector>
#include <sstream>

namespace VideoStitch {
/**
 * @brief Delete all elements.
 * @param container Container whose element to delete. Will be empty ion return.
 */
template <class ContainerT>
void deleteAll(ContainerT& container) {
  for (typename ContainerT::iterator it = container.begin(); it != container.end(); ++it) {
    delete *it;
  }
  container.clear();
}

/**
 * @brief Delete all values.
 * @param container Container whose element to delete. Will be empty on return.
 */
template <class ContainerT>
void deleteAllValues(ContainerT& container) {
  for (typename ContainerT::iterator it = container.begin(); it != container.end(); ++it) {
    delete it->second;
  }
  container.clear();
}

/**
 * @brief Helper function to swap one container with another one if a pointer is provided
 * @param containerPtr Pointer to container, can be nullptr
 * @param container Container to be swapped
 */
template <class ContainerT>
void containerSwapIfPtr(ContainerT* containerPtr, ContainerT& container) {
  if (containerPtr) {
    containerPtr->swap(container);
  }
}

/**
 * @brief Helper function to pretty-print containers
 * @param container Container whose elements will be assembled to a string
 */
template <class ContainerT>
std::string containerToString(const ContainerT& container) {
  std::stringstream stream;
  stream << "(";
  for (auto it = container.begin(); it != container.end();) {
    stream << *it;
    stream << ((++it != container.end()) ? ", " : "");
  }
  stream << ")";
  return stream.str();
}

}  // namespace VideoStitch

#endif
