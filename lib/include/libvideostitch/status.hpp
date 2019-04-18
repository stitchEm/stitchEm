// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "config.hpp"

#include <errno.h>

#include <cassert>
#include <cstring>
#include <map>
#include <string>

namespace VideoStitch {

enum class Origin {
  // algorithms
  BlendingMaskAlgorithm,
  CalibrationAlgorithm,
  CropAlgorithm,
  ExposureAlgorithm,
  PhotometricCalibrationAlgorithm,
  MaskInterpolationAlgorithm,
  ScoringAlgorithm,
  StabilizationAlgorithm,
  SynchronizationAlgorithm,
  MotionEstimationAlgorithm,
  EpipolarCurvesAlgorithm,

  // audio
  AudioPipeline,
  AudioPreProcessor,

  // blending
  ImageFlow,
  ImageWarper,

  // stitching pipeline
  Input,
  PreProcessor,
  Stitcher,
  PostProcessor,
  Output,
  Surface,

  // project handling
  PanoramaConfiguration,
  AudioPipelineConfiguration,

  // low level GPU backend
  GPU,

  // utils
  GeometryProcessingUtils,

  ExternalModule,
  Unspecified,
};

enum class ErrType {
  // unable to acquire a crucial resource
  OutOfResources,

  // a component needed to finish the operation could not be set up correctly,
  // usually has an underlying cause
  SetupFailure,

  InvalidConfiguration,

  // encountered an irrecoverable, unexpected error on runtime
  RuntimeError,

  // unable to compute a valid result
  AlgorithmFailure,

  // did not finish as user requested to stop
  OperationAbortedByUser,

  // a feature or configuration that has not been implemented
  UnsupportedAction,

  // runtime sanity check. asserts false.
  ImplementationError,

  None,
};

/**
 * An object that is either indicating normal operation (.ok())
 * or holds an error description on the encountered failure
 */
class VS_EXPORT Status {
 public:
  /**
   * Create an Ok Status
   */
  Status();

  /**
   * Create an Ok Status
   */
  static Status OK() { return Status(); }

  /**
   * Create an error Status
   */
  Status(Origin o, ErrType t, const std::string& message);

  /**
   * Create an error Status with an underlying cause
   */
  Status(Origin o, ErrType t, const std::string& message, Status underlyingCause);

  /**
   * Create an error Status
   */
  Status(Origin o, ErrType t, const std::string& tag, const std::string& message);

  /**
   * Create an error Status with an underlying cause
   */
  Status(Origin o, ErrType t, const std::string& tag, const std::string& message, Status underlyingCause);

  /**
   * Return the module or point of origin
   */
  Origin getOrigin() const;

  /**
   * Return the type
   */
  ErrType getType() const;

  /**
   * Return the runtime error information (English string)
   */
  std::string getErrorMessage() const;

  // these convenience methods don't require access to Status internals
  // and could be moved outside
  // ---
  std::string getOriginString() const;
  std::string getTypeString() const;
  // ---

  /**
   * Copy contructor.
   * @param other copy source
   */
  Status(const Status& other);

  /**
   * Copy assignment constructor
   * @param other copy source
   */
  Status& operator=(Status other) {
    std::swap(description, other.description);
    return *this;
  }

  ~Status();

  /**
   * Is the status Ok?
   */
  bool ok() const { return !description; }

  /**
   * Is there information on the underlying cause?
   */
  bool hasCause() const;

  /**
   * Access the underlying cause. Only valid if hasCause().
   */
  const Status& getCause() const;

  /**
   * Check the error type recursively in all underlying causes
   * @param t The error type to check
   * @return true if the status or one of its causes has this error type
   */
  bool hasUnderlyingCause(ErrType t) const;

 private:
  class Description;
  Description* description;
};

/** Result<CustomStatusCode> provides a more customizable Status
 *  Valid cases are not just 'Ok', but also the additional states
 *  defined in the enum class CustomStatusCode.
 *
 *  This is useful for operations that can result in a certain state
 *  that is acted upon by the caller, without this state being a runtime error.
 *
 *  Result<T> thus is either:
 *  1) Ok: still the default case. Create with Result<T>::OK(), test with .ok()
 *  2) custom code: in a custom state defined in enum class T
 *  3) ErrorWithStatus: in an error state, declared with an enclosed `Status`
 *
 *  Note: a CustomCode is not ok() and does not enclose an error Status
 */
template <typename CustomStatusCode>
class Result {
 public:
  typedef CustomStatusCode StatusCode;

  static_assert(sizeof StatusCode::Ok, "CustomStatusCode must be an enum class and must contain `Ok`");
  static_assert(sizeof StatusCode::ErrorWithStatus,
                "CustomStatusCode must an enum class must contain `ErrorWithStatus`");

  /**
   * Create an Ok Result, the default state
   */
  Result() : status(), code(StatusCode::Ok) {}

  /**
   * Create an Ok Result, the default state
   */
  static Result OK() { return Result(); }

  /**
   * Create an Result with a custom defined state
   *
   * The parameter state must not be Ok or ErrorWithStatus
   * as they define special states.
   */
  template <StatusCode state>
  static Result fromCode() {
    static_assert(state != StatusCode::Ok, "Use Result::OK()");
    static_assert(state != StatusCode::ErrorWithStatus, "Use fromError()");
    return Result(state);
  }

  /**
   * Create an Result with a custom defined state from an error status
   */
  static Result fromError(const Status& errorStatus) { return Result(errorStatus); }

  /**
   * Create an error Result, enclosing a Status
   */
  Result(Status genericStatus) : status(genericStatus) {
    if (genericStatus.ok()) {
      code = StatusCode::Ok;
    } else {
      code = StatusCode::ErrorWithStatus;
    }
  }

  bool ok() const { return code == StatusCode::Ok; }

  /**
   * Returns Ok, ErrorWithStatus or a custom state
   */
  StatusCode getCode() const { return code; }

  /**
   * Always Status::OK(), unless `getCode()` returns ErrorWithStatus
   */
  Status getStatus() const { return status; }

 private:
  // Construction should go through ::fromCode which disallows ErrorWithStatus without a Status
  Result(StatusCode state) : status(), code(state) {}

  Status status;
  StatusCode code;
};

/**
 * A deleter.
 */
template <class T>
struct DefaultDeleter {
  /**
   * Create a default deleter.
   */
  DefaultDeleter() {}
  /**
   * Copy constructor.
   */
  DefaultDeleter(const DefaultDeleter<T>& /*other*/) {}

  /**
   * Deletes the given object.
   * @param t Object to delete. Can be NULL.
   */
  void operator()(T* t) const { delete t; }
};

/**
 * An object that provides RAII on a factory-created object with an error status.
 * See DefaultDeleter for the constraints on Deleter.
 */
template <class T, class Deleter = DefaultDeleter<T>, class StatusClass = Status>
class Potential {
 public:
  /**
   * Move constructor. Should use Rvalue references, but we want to handle non-c++11 compilers.
   */
  Potential(const Potential& other) : object_(other.object_), status_(other.status_), deleter_(other.deleter_) {
    const_cast<T*&>(other.object_) = nullptr;
  }

  /**
   * Create a Potential with a null object and a given error
   */
  Potential(Origin o, ErrType t, const std::string& message) : object_(nullptr), status_(o, t, message), deleter_() {}
  Potential(Origin o, ErrType t, const std::string& tag, const std::string& message)
      : object_(nullptr), status_(o, t, tag, message), deleter_() {}

  /**
   * Create a Potential with a null object and a given error and an underlying error cause
   */
  Potential(Origin o, ErrType t, const std::string& message, Status underlyingCause)
      : object_(nullptr), status_(o, t, message, underlyingCause), deleter_() {}
  Potential(Origin o, ErrType t, const std::string& tag, const std::string& message, Status underlyingCause)
      : object_(nullptr), status_(o, t, tag, message, underlyingCause), deleter_() {}

  /**
   * Creates an Potential with a null object and a given status.
   * @param status root status code
   */
  Potential(const StatusClass& status) : object_(NULL), status_(status), deleter_() {
    // TODOSTATUS enable this assertion, it's confusing to have an .ok Potential without a value
    // it's currently used in the algorithms to optionally return a value
    // assert(!status_.ok());
  }

  /**
   * Creates an Potential from the given object
   * @param object Factory-created object. Ownership is transferred to the Potential.
   * @param deleter Custom deleter to be used for deleting @a object.
   */
  Potential(T* object, const Deleter& deleter = Deleter())
      : object_(object), status_(StatusClass::OK()), deleter_(deleter) {
    // Potential(nullptr) used to create a potential with an OutOfResources Status code
    // this feature was removed, create a Potential with a custom Status message if you are out of resources
    assert(object != nullptr);
  }

  ~Potential() { deleter_(object_); }

  /**
   * Returns the object.
   */
  T* operator->() { return object_; }

  /**
   * Returns the object
   */
  const T* operator->() const { return object_; }

  /**
   * Returns the object. Ownership is NOT transfered.
   */
  T* object() const { return object_; }

  /**
   * Returns the status for this object.
   */
  const StatusClass& status() const { return status_; }

  /**
   * Returns whether the Status of this Potential object is Ok
   */
  bool ok() const { return status_.ok(); }

  /**
   * Releases the created object and passes ownership to the caller.
   */
  T* release() {
    T* tmp = object_;
    object_ = NULL;
    return tmp;
  }

 private:
  Potential& operator=(const Potential&);
  T* object_;
  const StatusClass status_;
  const Deleter deleter_;
};

/**
 * An object that stores an optional value of type T and a Status code.
 * Similar to usage of std::optional.
 * A Status is always provided, even if the optional value is present.
 */
template <class T>
class PotentialValue {
 public:
  /**
   * @brief Creates an PotentialValue with a default constructed value and a an error
   */
  PotentialValue(const Status& status) : value_(), status_(status) {}

  /**
   * @brief Creates an PotentialValue with a value and a given status.
   * Useful when there's no default constructor for the value
   * @param status root status code
   * @param value The value to be stored
   */
  PotentialValue(const Status& status, T value) : value_(value), status_(status) {}

  /**
   * @brief Creates an PotentialValue from the given value. Status will be 'Ok', as value is present.
   * @param value The value to be stored
   */
  PotentialValue(const T& value) : value_(value), status_(Status::OK()) {}

  /**
   * @brief Creates a PotentialValue with move semantics
   * @param value The value to be stored
   */
  PotentialValue(T&& value) : value_(std::move(value)), status_(Status::OK()) {}

  /**
   * @brief Affectation operator
   */
  PotentialValue& operator=(const PotentialValue&) = default;

  /**
   * @brief Affectation operator with move semantics
   */
  PotentialValue& operator=(PotentialValue&& other) {
    value_ = std::move(other.value_);
    status_ = other.status_;
    return *this;
  }

  /**
   * @brief Copy constructor
   */
  PotentialValue(const PotentialValue& other) = default;

  /**
   * @brief Copy constructor with move semantics
   */
  PotentialValue(PotentialValue&& other) : value_(std::move(other.value_)), status_(other.status_) {}

  ~PotentialValue() {}

  /**
   * Returns the value
   */
  T value() const { return value_; }

  /**
   * Returns the value
   */
  const T& ref() const { return value_; }

  /**
   * Returns the value
   */
  T&& releaseValue() { return std::move(value_); }

  /**
   * Returns the status for this object.
   */
  const Status& status() const { return status_; }

  /**
   * Returns whether the Status of this potential value is Ok.
   */
  bool ok() const { return status_.ok(); }

 protected:
  T value_;        ///< Stored (optional) value. Default constructed if missing on init.
  Status status_;  ///< Stored status, not optional
};

#define PROPAGATE_FAILURE_STATUS(call)         \
  {                                            \
    const VideoStitch::Status status = (call); \
    if (!status.ok()) {                        \
      return status;                           \
    }                                          \
  }
#define FAIL_RETURN PROPAGATE_FAILURE_STATUS

#define PROPAGATE_FAILURE_CAUSE(call, origin, type, msg) \
  {                                                      \
    const VideoStitch::Status status = (call);           \
    if (!status.ok()) {                                  \
      return Status(origin, type, msg, status);          \
    }                                                    \
  }
#define FAIL_CAUSE PROPAGATE_FAILURE_CAUSE

#define PROPAGATE_FAILURE_MSG(call, msg)                                \
  {                                                                     \
    const VideoStitch::Status status = (call);                          \
    if (!status.ok()) {                                                 \
      return Status(status.getOrigin(), status.getType(), msg, status); \
    }                                                                   \
  }
#define FAIL_MSG PROPAGATE_FAILURE_MSG

#define PROPAGATE_FAILURE_CONDITION(condition, origin, type, msg) \
  {                                                               \
    if (!(condition)) {                                           \
      return Status(origin, type, msg);                           \
    }                                                             \
  }
#define FAIL_CONDITION PROPAGATE_FAILURE_CONDITION
}  // namespace VideoStitch
