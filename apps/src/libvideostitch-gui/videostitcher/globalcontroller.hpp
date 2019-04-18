// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "stitchercontroller.hpp"

/**
 * The Controller is unique in every application, so it is accessible through a global variable.
 * Since the real type of the Controller is up to the application to
 * decide, the global variable is actually a pointer.
 *
 * The application is responsible for implementing the real factory method.
 * The global factory can be configured at any time by setting
 * the global variable 'impl'.
 * The singleton pattern ensures correct deletion at shutdown.
 */
class VS_GUI_EXPORT GlobalController {
 public:
  class Impl {
   public:
    virtual ~Impl() {}
    virtual StitcherController* getController() const = 0;
    virtual void createController(int device) = 0;
    virtual void deleteController() = 0;
  };

  static GlobalController& getInstance() {
    static GlobalController theInstance;
    return theInstance;
  }
  ~GlobalController();

  void deleteController();

  StitcherController* getController() const;

 protected:
  void createController(int device);

 private:
  GlobalController();
  GlobalController(GlobalController const&);
  GlobalController& operator=(GlobalController const&);

  bool configure(Impl* i) {
    // once and for all
    if (impl == nullptr) {
      impl = i;
      return true;
    }
    return false;
  }

  Impl* impl;

  template <typename T>
  friend class GlobalControllerImpl;
};
