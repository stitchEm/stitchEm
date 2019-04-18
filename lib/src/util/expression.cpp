// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "expression.hpp"

#include <cstdlib>
#include <iostream>

namespace VideoStitch {
namespace Util {

Expr* Expr::parse(const std::string& expr) {
  // TODO: real parser.
  char* endConv = NULL;
  int64_t v = strtoll(expr.c_str(), &endConv, 10);
  if (endConv == expr.c_str() + expr.size()) {
    return new ConstantExpr(v);
  } else {
    return new ContextExpr(expr);
  }
}

}  // namespace Util
}  // namespace VideoStitch
