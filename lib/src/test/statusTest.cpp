// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/testing.hpp"

#include <gpu/buffer.hpp>
#include "libvideostitch/status.hpp"

#include <sstream>

namespace VideoStitch {
namespace Testing {

void testStatus() {
  ENSURE(Status::OK());

  // ensure that ENSURE is working as intended, too
  ENSURE(Status::OK().ok());

  Status defaultInitialized;
  ENSURE(defaultInitialized.ok());

  Status fail{Origin::ExternalModule, ErrType::RuntimeError, "Test error message"};
  ENSURE(!fail.ok());

  Status copy = fail;
  ENSURE(!fail.ok());
  ENSURE(!copy.ok());

  ENSURE(copy.getType() == ErrType::RuntimeError);
  ENSURE(copy.getOrigin() == Origin::ExternalModule);
  ENSURE_EQ(copy.getErrorMessage(), std::string("Test error message"));
}

void testPotentialManualDelete() {
  auto makeIntPotential = [](int *ptr) { return Potential<int>{ptr}; };

  Potential<int> pot = makeIntPotential(new int(982448879));
  ENSURE(pot.status().ok());
  ENSURE(pot.ok());
  ENSURE_EQ(*pot.object(), 982448879);
  int *ptr = pot.release();
  ENSURE_EQ(*ptr, 982448879);
  delete ptr;
}

void testPotentialAutoDelete() {
  auto makeIntPotential = [](int *ptr) { return Potential<int>{ptr}; };

  auto pot = makeIntPotential(new int(982448879));
  ENSURE(pot.status().ok());
  ENSURE(pot.ok());
  ENSURE_EQ(*pot.object(), 982448879);
}

void testPotentialValue() {
  auto makeIntPotentialValue = [](int val) { return PotentialValue<int>{val}; };

  auto pot = makeIntPotentialValue(982448879);
  ENSURE(pot.status().ok());
  ENSURE(pot.ok());
  ENSURE_EQ(pot.value(), 982448879);
}

void testPotentialImplicitConversion() {
  // implicit constructor with pointer
  auto makeIntPotentialValid = [](int *ptr) -> Potential<int> { return ptr; };

  auto potValid = makeIntPotentialValid(new int(982448879));
  ENSURE(potValid.status().ok());
  ENSURE(potValid.ok());
  ENSURE_EQ(*potValid.object(), 982448879);

  // implicit constructor with Status
  auto makeIntPotentialError = []() -> Potential<int> {
    return {Origin::ExternalModule, ErrType::RuntimeError, "Test error message"};
  };

  auto potError = makeIntPotentialError();
  ENSURE(!potError.status().ok());
  ENSURE(!potError.ok());

  ENSURE(potError.status().getOrigin() == Origin::ExternalModule);
  ENSURE(potError.status().getType() == ErrType::RuntimeError);
  ENSURE_EQ(potError.status().getErrorMessage(), std::string{"Test error message"});
}

void testStatusCause() {
  const Status a = Status::OK();
  ENSURE(!a.hasCause());
  ENSURE(&a.getCause() == &a);

  // An OK cause is not a cause.
  const Status b{Origin::ExternalModule, ErrType::RuntimeError, "Root error message", a};
  ENSURE(!b.hasCause());
  ENSURE(&b.getCause() == &b);

  // Test causes
  const Status c{Origin::ExternalModule, ErrType::RuntimeError, "Top error message", b};
  ENSURE(c.hasCause());
  ENSURE(c.getCause().getErrorMessage() == b.getErrorMessage());

  const Status d{Origin::PanoramaConfiguration, ErrType::AlgorithmFailure, "Top error message", c};
  ENSURE(d.hasUnderlyingCause(ErrType::AlgorithmFailure));
  ENSURE(d.hasUnderlyingCause(ErrType::RuntimeError));
  ENSURE(!d.hasUnderlyingCause(ErrType::SetupFailure));
}

void testCustomStatus() {
  enum class TestCode {
    Ok,
    ErrorWithStatus,
    CodeA,
    CodeB,
  };

  typedef Result<TestCode> TestStatus;

  ENSURE(TestStatus::OK().ok());
  ENSURE(!TestStatus::fromCode<TestCode::CodeA>().ok());

  ENSURE(TestStatus::fromCode<TestCode::CodeA>().getCode() == TestCode::CodeA);
  ENSURE(TestStatus::fromCode<TestCode::CodeB>().getCode() == TestCode::CodeB);

  // Result<T> custom states don't have a Status
  ENSURE(TestStatus::fromCode<TestCode::CodeA>().getStatus().ok());
  ENSURE(TestStatus::fromCode<TestCode::CodeB>().getStatus().ok());

  ENSURE(TestStatus(Status::OK()).ok());

  const Status someError{Origin::ExternalModule, ErrType::RuntimeError, "Top error message"};
  ENSURE(!TestStatus(someError).ok());
  ENSURE(TestStatus(someError).getCode() == TestCode::ErrorWithStatus);
}

}  // namespace Testing
}  // namespace VideoStitch

int main(int argc, char **argv) {
  VideoStitch::Testing::initTest();
  VideoStitch::Testing::testStatus();
  VideoStitch::Testing::testPotentialManualDelete();
  VideoStitch::Testing::testPotentialAutoDelete();
  VideoStitch::Testing::testPotentialValue();
  VideoStitch::Testing::testPotentialImplicitConversion();
  VideoStitch::Testing::testStatusCause();
  VideoStitch::Testing::testCustomStatus();
  return 0;
}
