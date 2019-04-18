%{
/* WARNING: AUTOMATICALLY GENERATED, DO NOT EDIT. */

/// @cond NEVER

#ifdef __GNUC__
# pragma GCC diagnostic ignored "-Wunused-variable"
# pragma GCC diagnostic ignored "-Wunused-parameter"
# pragma GCC diagnostic ignored "-Wunused-function"
# pragma GCC diagnostic ignored "-Wextra"
# pragma GCC diagnostic ignored "-Wtype-limits"
# pragma GCC diagnostic ignored "-Wconversion"
#elif defined(_MSC_VER)
# pragma warning( disable : 4005 )
#endif

#ifdef _MSC_VER
#include <codecvt>
#endif
#include <cstdio>
#include <iostream>
#include <sstream>
#include <locale>

#include <parse/jsonDriver.hpp>
#include "parser-generated/jsonParser.hpp"

/* Work around an incompatibility in flex (at least versions
  2.5.31 through 2.5.33): it generates code that does
  not conform to C89.  See Debian bug 333231
  <http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=333231>.  */
# undef yywrap
# define yywrap() 1

typedef VideoStitch::Parse::JsonParser::token token;

# define YY_USER_ACTION  yylloc->columns (yyleng);

/* By default yylex returns int, we use token_type.
  Unfortunately yyterminate by default returns 0, which is
  not of token_type.  */
#define yyterminate() return token::END
%}

%option yylineno
%option noyywrap
%option nounput
%option batch
%option debug
%option never-interactive
%option nounistd

%x comment
%x string_lit

DIGIT1to9 [1-9]
DIGIT [0-9]
DIGITS {DIGIT}+
INT {DIGIT}|{DIGIT1to9}{DIGITS}|-{DIGIT}|-{DIGIT1to9}{DIGITS}
DECIMAL_PART [.]{DIGITS}
EXP {E}{DIGITS}
E [eE][+-]?
HEX_DIGIT [0-9a-f]
FLOAT {INT}{DECIMAL_PART}|{INT}{EXP}|{INT}{DECIMAL_PART}{EXP}
UNESCAPEDCHAR [ -!#-\[\]-~]
ESCAPEDCHAR \\["\\bfnrt/]
ESCAPEDUNICODECHAR \\u{HEX_DIGIT}{HEX_DIGIT}{HEX_DIGIT}{HEX_DIGIT}
DBL_QUOTE ["]
UNICODEHIGHCHAR [\x80-\xff]
CHAR {UNESCAPEDCHAR}|{ESCAPEDCHAR}|{ESCAPEDUNICODECHAR}|{UNICODEHIGHCHAR}
CHARS {CHAR}+

%%

%{
  yylloc->step ();
%}

{DBL_QUOTE}{DBL_QUOTE} |
{DBL_QUOTE}{CHARS}{DBL_QUOTE} {
    yylval->string = new std::string(yytext + 1, strlen(yytext) - 2);
    return token::STRING_LITERAL;
};
{INT} {
    yylval->integer = atol(yytext);
    return token::INT;
}
{FLOAT} {
    std::stringstream ss;
    ss.imbue(std::locale("C"));
    ss << yytext;
    ss >> yylval->decimal;
    return token::DECIMAL;
}
true {
    return token::TRUEV;
};
false {
    return token::FALSEV;
};
null {
    return token::NULLV;
};
\{ {
    return token::OBRACE;
};

\} {
    return token::EBRACE;
};

\[ {
    return token::OBRACKET;
};

\] {
    return token::EBRACKET;
};

, {
    return token::COMMA;
};
: {
    return token::COLON;
};
[ \n\r\f\t]+  /* eat whitespace and newline*/;

%%

namespace VideoStitch {
namespace Parse {
bool JsonDriver::scan_begin () {
  yy_flex_debug = trace_scanning;
  yyin = NULL;
  dataBuffer = NULL;
  switch (parserSource) {
  case FromStdin:
    yyin = stdin;
    return true;
  case FromFile:
    {
#ifdef _MSC_VER
      std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
      std::wstring wideFilename;
      try {
        wideFilename = converter.from_bytes(filename);
        yyin = _wfopen(wideFilename.c_str(), L"r");
      } catch (std::range_error) {
        yyin = fopen(filename.c_str(), "r");
      }
#else
      yyin = fopen(filename.c_str(), "r");
#endif
      if (!yyin) {
        error("cannot open " + filename + ": " + strerror(errno));
        return false;
      }
      return true;
    }
  case FromData:
    dataBuffer = yy_scan_string(dataToParse.data());
    if (!dataBuffer) {
      error("cannot start parser");
      return false;
    }
    return true;
  }
  return false;
}

void JsonDriver::scan_end () {
  if (dataBuffer) {
    yy_delete_buffer((YY_BUFFER_STATE)dataBuffer);
    dataBuffer = NULL;
  } else {
    fclose(yyin);
  }
  yylex_destroy();
}
}
}

/// @endcond
