// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef SINGLETON_HPP
#define SINGLETON_HPP

#include <cstddef>

template <typename T>

/**
 * @brief Singleton template class.
 */
class Singleton {
 protected:
  Singleton() {}
  ~Singleton() {}

 public:
  /**
   * @brief Instantiates the static instance if it hasn't been instantiated.
   * @return Static instance.
   */
  static T *getInstance() {
    if (nullptr == _singleton) {
      _singleton = new T;
    }
    return (static_cast<T *>(_singleton));
  }

  static T &the() { return *getInstance(); }

  /**
   * @brief Destroys the static instance and sets it to nullptr.
   */
  static void destroy() {
    if (nullptr != _singleton) {
      delete _singleton;
      _singleton = nullptr;
    }
  }

 protected:
  static T *_singleton;
};

template <typename T>
T *Singleton<T>::_singleton = nullptr;

#endif  // SINGLETON_HPP
