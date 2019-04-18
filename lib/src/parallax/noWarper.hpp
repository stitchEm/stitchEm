// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once
#include "./imageWarper.hpp"

#include "libvideostitch/status.hpp"
#include "libvideostitch/imageWarperFactory.hpp"

namespace VideoStitch {
namespace Core {

class NoWarper : public ImageWarper {
 public:
  class Factory : public ImageWarperFactory {
   public:
    static Potential<ImageWarperFactory> parse(const Ptv::Value& value);
    explicit Factory() {}
    virtual std::string getImageWarperName() const override;
    virtual Potential<ImageWarper> create() const override;
    virtual bool needsInputPreProcessing() const override;
    virtual Ptv::Value* serialize() const override;
    virtual ImageWarperFactory* clone() const override;
    virtual std::string hash() const override;
    virtual ~Factory() {}
  };

  static std::string getName();

  friend class ImageWarper;

  virtual bool needImageFlow() const override;

  virtual ImageWarperAlgorithm getWarperAlgorithm() const override;

 private:
  explicit NoWarper(const std::map<std::string, float>& parameters);
};
}  // namespace Core
}  // namespace VideoStitch
