// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

%skeleton "lalr1.cc"
%defines
%define parser_class_name {JsonParser}
%define api.namespace  {VideoStitch::Parse}

%code requires {
  #ifdef __GNUC__
  #ifdef __clang__
  # pragma GCC diagnostic ignored "-Wdeprecated-register"
  #endif // __clang__
  # pragma GCC diagnostic ignored "-Wunused-variable"
  # pragma GCC diagnostic ignored "-Wunused-parameter"
  # pragma GCC diagnostic ignored "-Wunused-function"
  # pragma GCC diagnostic ignored "-Wextra"
  # pragma GCC diagnostic ignored "-Wtype-limits"
  # pragma GCC diagnostic ignored "-Wconversion"
  #elif defined(_MSC_VER)
  # pragma warning( disable : 4005 )
  # pragma warning( disable : 4127 )
  # pragma warning( disable : 4129 )
  # pragma warning( disable : 4146 )
  # pragma warning( disable : 4244 )
  # pragma warning( disable : 4267 )
  #endif
  # include <string>
  namespace VideoStitch {
  namespace Parse {
  class JsonDriver;
  }
  }
}

%locations
%initial-action
{
  // Initialize the initial location.
  @$.begin.filename = @$.end.filename = &driver.filename;
};

%debug
%error-verbose

%{
/* WARNING: AUTOMATICALLY GENERATED, DO NOT EDIT. */

/// @cond NEVER

#include "parse/jsonDriver.hpp"
#include "parse/json.hpp"

#include "util/strutils.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <string>

%}

%parse-param { JsonDriver& driver }
%lex-param   { JsonDriver& driver }

%union
{
  int64_t integer;
  double decimal;
  std::string *string;
  std::pair<std::string, JsonValue*>* pairType;
  JsonValue* jsonValueType;
  std::vector<Ptv::Value*>* listEltsType;
}

%token <integer> INT "integer"
%token <decimal> DECIMAL "decimal"
%token <string> STRING_LITERAL "string literal"
%token TRUEV FALSEV NULLV
%left OBRACE EBRACE OBRACKET EBRACKET
%left COMMA
%left COLON
%token END 0 "end of file"

%type <pairType> pair
%type <listEltsType> list_elts
%type <jsonValueType> members
%type <jsonValueType> json_list
%type <jsonValueType> json_set
%type <jsonValueType> value

%destructor { delete ($$)->second; delete $$; } pair
%destructor { for (size_t i = 0; i < ($$)->size(); ++i) { delete ((*($$))[i]); } delete $$; } list_elts
%destructor { delete $$; } members
%destructor { delete $$; } json_list
%destructor { delete $$; } json_set
%destructor { delete $$; } value
%destructor { delete $$; } STRING_LITERAL

%start root

%%

root: json_list {
  driver.root = $1;
}
| json_set {
  driver.root = $1;
};

json_set: OBRACE EBRACE {
  // Empty set.
  $$ = new JsonValue();
}
| OBRACE members EBRACE {
  $2->reverse();
  $$ = $2;
};

members: pair {
  $$ = new JsonValue();
  delete ($$)->push($1->first, $1->second);
  $1->second = NULL;
  delete $1;
}
| pair COMMA members {
  delete ($3)->push($1->first, $1->second);
  $1->second = NULL;
  delete $1;
  $$ = $3;
};

pair: STRING_LITERAL COLON value {
  $$ = new std::pair<std::string, JsonValue*>(*$1, $3);
  delete $1;
};

json_list: OBRACKET EBRACKET {
  $$ = new JsonValue((void*)NULL);
  $$->asList();
}
| OBRACKET list_elts EBRACKET {
  std::reverse(($2)->begin(), ($2)->end());
  $$ = new JsonValue($2);
};

list_elts: value {
  $$ = new std::vector<Ptv::Value*>();
  ($$)->push_back($1);
}
| value COMMA list_elts {
  ($3)->push_back($1);
  $$ = $3;
};

value: STRING_LITERAL {
  std::string res;
  bool success = Util::unescapeStr(*($1), res);
  delete $1;
  if (!success) {
    std::string msg("Error: invalid escape sequence. (did you forget to escape a backslash ?)");
    error(yylhs.location, msg);
  }
  $$ = new JsonValue(res);
}
| INT {
  $$ = new JsonValue($1);
}
| DECIMAL {
  $$ = new JsonValue($1);
}
| json_set {
  $$ = $1;
}
| json_list {
  $$ = $1;
}
| TRUEV {
  $$ = new JsonValue(true);
}
| FALSEV {
  $$ = new JsonValue(false);
}
| NULLV {
  $$ = new JsonValue((void*)NULL);
}
;

%%

namespace VideoStitch {
namespace Parse {
void JsonParser::error(const location_type& l, const std::string& m) {
  driver.error(l, m);
}
}
}

/// @endcond
