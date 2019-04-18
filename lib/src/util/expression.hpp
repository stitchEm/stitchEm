// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#include "libvideostitch/logging.hpp"

#include <string>
#include <stdint.h>
#include <ostream>

namespace VideoStitch {

namespace Util {

class Expr;

/**
 * Evaluation result of an expression.
 */
class EvalResult {
 public:
  EvalResult() : valid(false), v(0) {}
  explicit EvalResult(int64_t v) : valid(true), v(v) {}

  bool isValid() const { return valid; }

  int64_t getInt() const { return v; }

  struct Plus {
    EvalResult operator()(const EvalResult& r1, const EvalResult& r2) const {
      return EvalResult(r1.getInt() + r2.getInt());
    }
  };

  struct Subtr {
    EvalResult operator()(const EvalResult& r1, const EvalResult& r2) const {
      return EvalResult(r1.getInt() - r2.getInt());
    }
  };

  struct Mult {
    EvalResult operator()(const EvalResult& r1, const EvalResult& r2) const {
      return EvalResult(r1.getInt() * r2.getInt());
    }
  };

  struct Div {
    EvalResult operator()(const EvalResult& r1, const EvalResult& r2) const {
      return EvalResult(r1.getInt() / r2.getInt());
    }
  };

 private:
  const bool valid;
  const int64_t v;
};

/**
 * Evaluation context for an expression.
 */
class Context {
 public:
  /**
   * Returns the value of a given variable,
   */
  virtual EvalResult get(const std::string& var) const = 0;
};

/**
 * A simple evaluable expression.
 */
class Expr {
 public:
  /**
   * Parses an expression and returns NULL on failure.
   * Right now we support:
   *  + - * /
   */
  static Expr* parse(const std::string& expr);

  virtual ~Expr() {}

  /**
   * Evaluates the expression in the given context.
   */
  virtual EvalResult eval(const Context& context) const = 0;

  /**
   * Prints out the expression.
   * @param os Output stream.
   */
  virtual void print(std::ostream& os) = 0;

 protected:
  Expr() {}
};

/**
 * A constant.
 */
class ConstantExpr : public Expr {
 public:
  explicit ConstantExpr(int64_t v) : v(v) {}

  virtual EvalResult eval(const Context& /*context*/) const { return EvalResult(v); }

  virtual void print(std::ostream& os) { os << v; };

 private:
  const int64_t v;
};

/**
 * A expr whose value is stored in the context by name.
 */
class ContextExpr : public Expr {
 public:
  explicit ContextExpr(const std::string& name) : name(name) {}

  virtual EvalResult eval(const Context& context) const { return context.get(name); }

  virtual void print(std::ostream& os) { os << name; };

 private:
  const std::string name;
};

/**
 * Plus
 */
template <class Op>
class BinaryExpr : public Expr {
 public:
  BinaryExpr(Expr* e1, Expr* e2) : e1(e1), e2(e2) {}

  virtual ~BinaryExpr() {
    delete e1;
    delete e2;
  }

  virtual EvalResult eval(const Context& context) const {
    EvalResult r1(e1->eval(context));
    EvalResult r2(e2->eval(context));
    if (!r1.isValid() || !r2.isValid()) {
      return EvalResult();
    }
    return EvalResult(op(r1, r2));
  }

  virtual void print(std::ostream& os) {
    e1->print(os);
    os << " " << Op::display << " ";
    e2->print(os);
  };

 private:
  Op op;
  Expr* e1;
  Expr* e2;
};

typedef BinaryExpr<EvalResult::Plus> AddExpr;
typedef BinaryExpr<EvalResult::Subtr> SubtrExpr;
typedef BinaryExpr<EvalResult::Mult> MultExpr;
typedef BinaryExpr<EvalResult::Div> DivExpr;
}  // namespace Util
}  // namespace VideoStitch
