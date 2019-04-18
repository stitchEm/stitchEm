#pragma once

#include "libvideostitch-gui/videostitcher/globalcontroller.hpp"


template <typename Controller>
class GlobalControllerImpl {
public:
  static GlobalControllerImpl& getInstance() {
    static GlobalControllerImpl theInstance;
    return theInstance;
  }
  ~GlobalControllerImpl() {}

  Controller* getController() const {
    return dynamic_cast<Controller*>(GlobalController::getInstance().getController());
  }

  void createController(int device) {
    GlobalController::getInstance().createController(device);
  }

  void deleteController() {
    GlobalController::getInstance().deleteController();
  }

  GlobalControllerImpl() {
    Impl *i = new Impl();
    if (!GlobalController::getInstance().configure(i)) {
      delete i;
      i = nullptr;
    }
  }

private:
  class Impl : public GlobalController::Impl {
  public:
    StitcherController* getController() const override;
    void createController(int device) override;
    void deleteController() override;

    Impl();
    virtual ~Impl();

  private:
    Controller* stitcher;
  };
};

template <typename Controller>
GlobalControllerImpl<Controller>::Impl::Impl()
  : stitcher(nullptr) {
}

template <typename Controller>
GlobalControllerImpl<Controller>::Impl::~Impl() {
}

template <typename Controller>
StitcherController* GlobalControllerImpl<Controller>::Impl::getController() const {
  return stitcher;
}

template <typename Controller>
void GlobalControllerImpl<Controller>::Impl::createController(int device) {
  Q_ASSERT(stitcher == nullptr); // Assert that we have no memory leak

  VideoStitch::Core::PanoDeviceDefinition dd;
  dd.device = device;
  stitcher = new Controller(dd);
  VideoStitch::Helper::LogManager::getInstance()->writeToLogFile("stitcher controller created");
}

template <typename Controller>
void GlobalControllerImpl<Controller>::Impl::deleteController() {
  if (stitcher) {
    stitcher->deleteLater();
    stitcher = nullptr;
  }
}