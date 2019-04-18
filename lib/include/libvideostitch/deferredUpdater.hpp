// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#ifndef DEFERREDUPDATER_HPP
#define DEFERREDUPDATER_HPP

#include <memory>
#include <future>
#include <vector>
#include <functional>
#include <chrono>
#include <algorithm>
#include <iterator>

#include "config.hpp"
#include "status.hpp"
#include "delayedAction.hpp"
#include "logging.hpp"

namespace VideoStitch {
namespace Core {

// Workaround for bug in vscpp2013. Can be removed when we move to vscpp2015
// origin:
// https://stackoverflow.com/questions/16140293/stdasync-decayinglosing-rvalue-reference-in-visual-studio-2012-update-2-any
// also on move to vscpp2105 preserved features can be replaced with just lambdas
template <typename T>
struct rvref_wrapper {
  rvref_wrapper(T&& value) : value_(std::move(value)) {}
  rvref_wrapper(rvref_wrapper const& other) : value_(other.get()) {}
  T&& get() const { return std::move(value_); }

  using WrappedType = typename T::element_type;
  WrappedType* release() { return value_.release(); }

  mutable T value_;
};

template <typename T>
auto rvref(T&& x) -> rvref_wrapper<typename std::decay<T>::type> {
  return std::move(x);
}

// Make sure that params are preserved (deep copy and correct cleanup afterwards)
#define PRESERVE_ACTION(actionName, baseObjectName, ...)                                                       \
  {                                                                                                            \
    auto futureCopy = updateObjectFuture;                                                                      \
    actionsToRepeat.emplace_back(std::async(                                                                   \
        std::launch::deferred, [futureCopy, ##__VA_ARGS__]() { futureCopy.get()->actionName(__VA_ARGS__); })); \
    baseObjectName->actionName(__VA_ARGS__);                                                                   \
  }

// Make sure that params are preserved (deep copy and correct cleanup afterwards)
#define PRESERVE_ACTION_RETURN(actionName, baseObjectName, ...)                                                \
  {                                                                                                            \
    auto futureCopy = updateObjectFuture;                                                                      \
    actionsToRepeat.emplace_back(std::async(                                                                   \
        std::launch::deferred, [futureCopy, ##__VA_ARGS__]() { futureCopy.get()->actionName(__VA_ARGS__); })); \
    return baseObjectName->actionName(__VA_ARGS__);                                                            \
  }

// Todo: Can be simplified after switch to vs2015
#define PRESERVE_ACTION_MANAGED(actionName, baseObjectName, smartPointerType, releaseStatement, cleaunUpStatement, \
                                returnStatement, smartPointer, originalPointer, ...)                               \
  {                                                                                                                \
    auto futureCopy = updateObjectFuture;                                                                          \
    auto action_future = std::async(                                                                               \
        std::launch::deferred,                                                                                     \
        [futureCopy, ##__VA_ARGS__](rvref_wrapper<smartPointerType> capturedSmartPointer) mutable {                \
          cleaunUpStatement futureCopy.get()->actionName(capturedSmartPointer.releaseStatement(), ##__VA_ARGS__);  \
        },                                                                                                         \
        rvref(std::move(smartPointer)));                                                                           \
    actionsToRepeat.emplace_back(std::move(action_future));                                                        \
    returnStatement baseObjectName->actionName(originalPointer, ##__VA_ARGS__);                                    \
  }

#define PRESERVE_ACTION_CLONEABLE(actionName, baseObjectName, cleaunUpStatement, returnStatement, type,    \
                                  originalPointer, ...)                                                    \
  {                                                                                                        \
    auto preservedClonnable = std::unique_ptr<type>(originalPointer->clone());                             \
    PRESERVE_ACTION_MANAGED(actionName, baseObjectName, std::unique_ptr<type>, release, cleaunUpStatement, \
                            returnStatement, preservedClonnable, originalPointer, ##__VA_ARGS__)           \
  }

#define PRESERVE_ACTION_CURVE(actionName, baseObjectName, cleaunUpStatement, returnStatement, curveType, \
                              originalPointer, ...)                                                      \
  {                                                                                                      \
    PRESERVE_ACTION_CLONEABLE(actionName, baseObjectName, cleaunUpStatement, returnStatement,            \
                              CurveTemplate<curveType>, originalPointer, ##__VA_ARGS__)                  \
  }

static const std::string API_MISUSE_MESSAGE(
    "Misuse of the deferredUpdater API, you can't setToUpdate multiple different objects for same updater in current "
    "implementation");

/** On how the "remember and reapply" mechanism works:
 * Every call to the function that mutates *definition is wrapped in a macro, that
 * 1) Makes a deep copy of any parameter function has.
 * 2) Creates std::async task with std::launch::deferred policy (executed when required and not concurrently).
 * In this task we call same function with copied parameters on "promised object" (It's represented by shared_future,
 * that would be set later). 3) Adds future for the created task to the actions vector (with timestamp to identify in
 * what order the actions were performed). 4) Calls the underlying function by forwarding it to a wrapped object.
 *
 * (There could be some variation e.g. for sub-updaters we don't call the same function, but set a promise on
 * subupdater)
 *
 * When we want to later apply a set of changes, we call apply function with an object that should be modified.
 * The apply function:
 * 1) If threre were any sub-updaters it merges all actions into 1 ordered list (works only with 1 level of subupdaters
 * for now) 2) It sets the promise for the shared_future mentioned above making the object to update available for
 * preserved tasks. 3) It goes through the list of tasks and executes them (by calling .get() on the future).
 *
 * P.S. some ugliness is sponsored by VS2013
 *
 */

template <typename T>
class VS_EXPORT DeferredUpdater {
 public:
  DeferredUpdater() : updateObjectFuture(updateObjectPromise.get_future()) {}

  DeferredUpdater(const DeferredUpdater& rhs) = delete;
  DeferredUpdater(DeferredUpdater&& rhs)
      : actionsToRepeat(std::move(rhs.actionsToRepeat)),
        subUpdatersActionLists(std::move(rhs.subUpdatersActionLists)),
        updateObjectPromise(std::move(rhs.updateObjectPromise)),
        updateObjectFuture(std::move(rhs.updateObjectFuture)) {}  // Todo can be defaulted when we switch to vs2015

  virtual ~DeferredUpdater() = default;

  /** @brief setToUpdate Set object on which updates would be applide
   */
  virtual Status setToUpdate(T& updateValue) {
    /** We want to set value only once (otherwise exception will be thrown).
     * Correspondingly with this approach we can update only one extra object with give set of changes.
     * We can change this in the future if we want to by changing the media through which we pass value to actions or
     * by passing future as a parameter to them and not capturing it on creation.
     **/
    if (updateObjectFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
      if (updateObjectFuture.get() != &updateValue) {
        return Status(Origin::PanoramaConfiguration, ErrType::ImplementationError, API_MISUSE_MESSAGE);
      }
    } else {
      updateObjectPromise.set_value(&updateValue);
    }
    return Status();
  }

  /**
   * @brief apply Apply preserved set of changes on top of the target object
   * @param updateValue Target object
   */
  virtual void apply(T& updateValue) {
    mergeActions();
    setToUpdate(updateValue);
    for (auto& action : actionsToRepeat) {
      action.execute();
    }
  }

  /**
   * @brief getActions Get list of preserved actions
   * @return List of preserved actions
   */
  virtual std::vector<DelayedAction>& getActions() { return actionsToRepeat; }

  /**
   * @brief getCloneUpdater Creates updater functor that would copy the given object, apply changes on top of the copy
   * and return it.
   * @return Functor with described properties.
   */
  virtual std::function<Potential<T>(const T&)> getCloneUpdater() {  // Not valid after updater have been destroyed
    return [this](const T& clonnable) {
      auto result = VideoStitch::Potential<T>(clonnable.clone());
      this->apply(*result.object());
      return result;
    };
  }

 protected:
  /**
   * @brief mergeActions Merge actions list from parent object and subupdaters into one list.
   */
  virtual void mergeActions() {
    if (subUpdatersActionLists.empty()) {
      return;
    }

    /** We could do merge more effectively if it would be required (see: heap merge)
     **/
    for (const auto& actionList : subUpdatersActionLists) {
      std::copy(begin(actionList.get()), end(actionList.get()), std::back_inserter(actionsToRepeat));
    }

    std::sort(begin(actionsToRepeat), end(actionsToRepeat));
  }

  std::vector<DelayedAction> actionsToRepeat;
  std::vector<std::reference_wrapper<std::vector<DelayedAction>>> subUpdatersActionLists;
  std::promise<T*> updateObjectPromise;
  std::shared_future<T*> updateObjectFuture;
};

}  // namespace Core
}  // namespace VideoStitch

#endif  // DEFERREDUPDATER_HPP
