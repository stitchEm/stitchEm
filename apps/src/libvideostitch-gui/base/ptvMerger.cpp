// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "ptvMerger.hpp"

#include "libvideostitch/ptv.hpp"
#include "libvideostitch/parse.hpp"

namespace VideoStitch {
namespace Helper {

void PtvMerger::mergeValue(Ptv::Value* originalValue, Ptv::Value* templateValue) {
  for (int i = 0; i < templateValue->size(); i++) {
    std::pair<const std::string*, const Ptv::Value*> pair = templateValue->get(i);

    switch (pair.second->getType()) {
      case Ptv::Value::NIL:
        originalValue->get(*pair.first);
        break;
      case Ptv::Value::BOOL:
        originalValue->get(*pair.first)->asBool() = pair.second->asBool();
        break;
      case Ptv::Value::INT:
        originalValue->get(*pair.first)->asInt() = pair.second->asInt();
        break;
      case Ptv::Value::DOUBLE:
        originalValue->get(*pair.first)->asDouble() = pair.second->asDouble();
        break;
      case Ptv::Value::STRING:
        originalValue->get(*pair.first)->asString() = pair.second->asString();
        break;
      case Ptv::Value::OBJECT: {
        Ptv::Value* templateObject = templateValue->get(*pair.first);
        Ptv::Value& originalObject = originalValue->get(*pair.first)->asObject();
        if (templateObject->size() >= 1) {
          mergeValue(&originalObject, templateObject);
          originalValue->push(*pair.first, &originalObject);
        }
        break;
      }
      case Ptv::Value::LIST:
        // merge lists recursively
        std::vector<Ptv::Value*>& originalList = originalValue->get(*pair.first)->asList();
        std::vector<Ptv::Value*> templateList = templateValue->get(*pair.first)->asList();
        if (originalList.size() == templateList.size()) {
          for (size_t j = 0; j < originalList.size(); j++) {
            mergeValue(originalList[j], templateList[j]);
          }
        } else {
          // recursive merging not possible, just replace
          for (auto valPtr : originalList) {
            delete valPtr;
          }
          originalList.clear();
          for (Ptv::Value* val : templateList) {
            originalList.push_back(val->clone());
          }
        }
        break;
    }
  }
}

void PtvMerger::removeFrom(Ptv::Value* originalValue, Ptv::Value* toRemove) {
  for (int i = 0; i < toRemove->size(); i++) {
    std::pair<const std::string*, const Ptv::Value*> pair = toRemove->get(i);
    if (originalValue->has(*pair.first)) {
      originalValue->remove(*pair.first);
    }
  }
}

std::unique_ptr<Ptv::Value> PtvMerger::getMergedValue(const std::string& currentPtv, const std::string& templatePtv) {
  Potential<Ptv::Parser> currentParser = Ptv::Parser::create();
  if (!currentParser.ok()) {
    Logger::get(Logger::Error) << "Error: Could not create a PTV parser for " << currentPtv << std::endl;
    return std::unique_ptr<Ptv::Value>(Ptv::Value::emptyObject());
  }
  bool ret = currentParser->parse(currentPtv);
  if (!ret) {
    Logger::get(Logger::Error) << "Error: Cannot parse PTV file: " << currentPtv << std::endl;
    Logger::get(Logger::Error) << currentParser->getErrorMessage() << std::endl;
    return std::unique_ptr<Ptv::Value>(Ptv::Value::emptyObject());
  }

  Potential<Ptv::Parser> templateParser = Ptv::Parser::create();
  if (!templateParser.ok()) {
    Logger::get(Logger::Error) << "Error: Could not create a PTV parser for " << templatePtv << std::endl;
    return std::unique_ptr<Ptv::Value>(Ptv::Value::emptyObject());
  }
  ret = templateParser->parse(templatePtv);
  if (!ret) {
    Logger::get(Logger::Error) << "Error: Cannot parse PTV file: " << templatePtv << std::endl;
    Logger::get(Logger::Error) << templateParser->getErrorMessage() << std::endl;
    return std::unique_ptr<Ptv::Value>(Ptv::Value::emptyObject());
  }

  std::unique_ptr<Ptv::Value> currentProjectRoot(currentParser->getRoot().clone());
  std::unique_ptr<Ptv::Value> templateProjectRoot(templateParser->getRoot().clone());

  mergeValue(currentProjectRoot.get(), templateProjectRoot.get());
  return currentProjectRoot;
}

void PtvMerger::saveMergedPtv(const std::string& currentPtv, const std::string& templatePtv,
                              const std::string& outputPtv) {
  std::string outputFile;
  outputFile = (outputPtv == "") ? currentPtv : outputPtv;

  std::unique_ptr<Ptv::Value> root(getMergedValue(currentPtv, templatePtv));
  std::unique_ptr<Ptv::Value> empty(Ptv::Value::emptyObject());
  if (root.get() == empty.get()) {
    Logger::get(Logger::Error) << "Error: could not merge " << currentPtv << " with " << templatePtv << std::endl;
    return;
  }

  std::ofstream ofs(outputFile.c_str(), std::ios_base::out);
  if (!ofs.is_open()) {
    Logger::get(Logger::Error) << "Error: cannot open '" << outputFile << "' for writing." << std::endl;
    return;
  }
  assert(root);
  root->printJson(ofs);
}

}  // namespace Helper
}  // namespace VideoStitch
