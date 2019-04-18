// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include <assert.h>

namespace VideoStitch {
namespace Util {
/**
 * Instances are indexed by a Key.
 *
 * Usage: subclass as follows.
 * \code{.cpp}
 * class MyClass : public Dictionary<MyKey, MyClass> {
 *   MyClass (MyKey const& k) : Dictionary<MyKey, MyClass> (k);
 * };
 * \endcode
 *
 */
template <typename Key, class Class>
class Dictionary {
 public:
  typedef std::map<Key, Class*> InstanceMap;

  virtual ~Dictionary() { instances().erase(key); }

  /**
   * \returns the instance with key name. 0 if not found.
   */
  static Class* getInstance(const Key& key) {
    Class* lReturn = 0;
    typename InstanceMap::const_iterator it = instances().find(key);
    if (it != instances().end()) {
      lReturn = it->second;
    }
    return lReturn;
  }

  static InstanceMap const& getInstances() { return instances(); }

 protected:
  explicit Dictionary(const Key& key, Class* object) : key(key) {
    assert(instances().find(key) == instances().end());
    instances()[key] = object;
  }

 private:
  /** \name Non-copyable */
  //\{
  Dictionary(const Dictionary&);
  Dictionary& operator=(const Dictionary&);
  //\}

  const Key key;

  static InstanceMap& instances() {
    static InstanceMap lReturn;
    return lReturn;
  }
};

/**
 * Traits class used to distinguish between Algorithm and OnlineAlgorithm.
 */
template <bool Online>
struct RegisteredTraits;

template <>
struct RegisteredTraits<false> {
  typedef Algorithm AlgoType;
};

template <>
struct RegisteredTraits<true> {
  typedef OnlineAlgorithm AlgoType;
};

/**
 * Provides a mechanism to register Algorithm''s.
 *
 * Expects registering classes to have:
 *   - a static member: const char* docString
 *   - and a constructor taking a parameter: Ptv::Value* config.
 *
 * Instances are indexed by name.
 *
 * Usage : instantiate RegisteredAlgoBase<MyAlgoClass> in MyAlgoClass.cpp.
 */
template <bool Online>
class RegisteredAlgoBase : public Dictionary<std::string, RegisteredAlgoBase<Online> > {
 protected:
  typedef typename RegisteredTraits<Online>::AlgoType AlgoType;

 public:
  /** \name Mandatory services. */
  //\{
  virtual Potential<AlgoType> create(const Ptv::Value* config) const = 0;
  virtual const char* getDocString() const = 0;
  //\}
 protected:
  explicit RegisteredAlgoBase(const std::string& name) : Dictionary<std::string, RegisteredAlgoBase>(name, this) {}

 private:
};

template <class AlgoClass, bool Online = false>
class RegisteredAlgo : public RegisteredAlgoBase<Online> {
  typedef RegisteredAlgoBase<Online> BaseType;
  typedef typename BaseType::AlgoType AlgoType;

 public:
  explicit RegisteredAlgo(const std::string& name) : BaseType(name) {}

  /** \name Mandatory services. */
  //\{
  virtual Potential<AlgoType> create(const Ptv::Value* config) const {
    AlgoType* algo = new AlgoClass(config);
    if (algo == nullptr) {
      return {Origin::Unspecified, ErrType::OutOfResources, "Cannot create algorithm"};
    }
    return Potential<AlgoType>(algo);
  }
  virtual const char* getDocString() const { return AlgoClass::docString; }
  //\}
 private:
};
}  // namespace Util
}  // namespace VideoStitch
