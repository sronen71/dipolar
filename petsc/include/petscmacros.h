#ifndef PETSC_PREPROCESSOR_MACROS_H
#define PETSC_PREPROCESSOR_MACROS_H

#include <petscconf.h>
#include <petscconf_poison.h> /* for PetscDefined() error checking */

/* SUBMANSEC = Sys */

/* ========================================================================== */
/* This facilitates using the C version of PETSc from C++ and the C++ version from C. */
#if defined(__cplusplus)
#  define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_CXX
#else
#  define PETSC_FUNCTION_NAME PETSC_FUNCTION_NAME_C
#endif

/* ========================================================================== */
/* Since PETSc manages its own extern "C" handling users should never include PETSc include
 * files within extern "C". This will generate a compiler error if a user does put the include
 * file within an extern "C".
 */
#if defined(__cplusplus)
void assert_never_put_petsc_headers_inside_an_extern_c(int); void assert_never_put_petsc_headers_inside_an_extern_c(double);
#endif

#if defined(__cplusplus)
#  define PETSC_RESTRICT PETSC_CXX_RESTRICT
#else
#  define PETSC_RESTRICT restrict
#endif

#define PETSC_INLINE PETSC_DEPRECATED_MACRO("GCC warning \"PETSC_INLINE is deprecated (since version 3.17)\"") inline
#define PETSC_STATIC_INLINE PETSC_DEPRECATED_MACRO("GCC warning \"PETSC_STATIC_INLINE is deprecated (since version 3.17)\"") static inline

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES) /* For Win32 shared libraries */
#  define PETSC_DLLEXPORT __declspec(dllexport)
#  define PETSC_DLLIMPORT __declspec(dllimport)
#  define PETSC_VISIBILITY_INTERNAL
#elif defined(__cplusplus) && defined(PETSC_USE_VISIBILITY_CXX)
#  define PETSC_DLLEXPORT __attribute__((visibility ("default")))
#  define PETSC_DLLIMPORT __attribute__((visibility ("default")))
#  define PETSC_VISIBILITY_INTERNAL __attribute__((visibility ("hidden")))
#elif !defined(__cplusplus) && defined(PETSC_USE_VISIBILITY_C)
#  define PETSC_DLLEXPORT __attribute__((visibility ("default")))
#  define PETSC_DLLIMPORT __attribute__((visibility ("default")))
#  define PETSC_VISIBILITY_INTERNAL __attribute__((visibility ("hidden")))
#else
#  define PETSC_DLLEXPORT
#  define PETSC_DLLIMPORT
#  define PETSC_VISIBILITY_INTERNAL
#endif

#if defined(petsc_EXPORTS) /* CMake defines this when building the shared library */
#  define PETSC_VISIBILITY_PUBLIC PETSC_DLLEXPORT
#else  /* Win32 users need this to import symbols from petsc.dll */
#  define PETSC_VISIBILITY_PUBLIC PETSC_DLLIMPORT
#endif

/* Functions tagged with PETSC_EXTERN in the header files are always defined as extern "C" when
 * compiled with C++ so they may be used from C and are always visible in the shared libraries
 */
#if defined(__cplusplus)
#  define PETSC_EXTERN         extern "C" PETSC_VISIBILITY_PUBLIC
#  define PETSC_EXTERN_TYPEDEF extern "C"
#  define PETSC_INTERN         extern "C" PETSC_VISIBILITY_INTERNAL
#else
#  define PETSC_EXTERN         extern PETSC_VISIBILITY_PUBLIC
#  define PETSC_EXTERN_TYPEDEF
#  define PETSC_INTERN         extern PETSC_VISIBILITY_INTERNAL
#endif

#if defined(PETSC_USE_SINGLE_LIBRARY)
#  define PETSC_SINGLE_LIBRARY_INTERN PETSC_INTERN
#else
#  define PETSC_SINGLE_LIBRARY_INTERN PETSC_EXTERN
#endif

/*MC
  PetscHasAttribute - Determine whether a particular __attribute__ is supported by the compiler

  Synopsis:
  #include <petscmacros.h>
  boolean PetscHasAttribute(name)

  Input Parameter:
. name - The name of the attribute to test

  Notes:
  name should be identical to what you might pass to the __attribute__ declaration itself --
  plain, unbroken text.

  As PetscHasAttribute() is wrapper over the function-like macro __has_attribute(), the exact
  type and value returned is implementation defined. In practice however, it usually returns
  the integer literal 1 if the attribute is supported, and integer literal 0 if the attribute
  is not supported.

  Example Usage:
  Typical usage is using the preprocessor

.vb
  #if PetscHasAttribute(always_inline)
  #  define MY_ALWAYS_INLINE __attribute__((always_inline))
  #else
  #  define MY_ALWAYS_INLINE
  #endif

  void foo(void) MY_ALWAYS_INLINE;
.ve

  but it can also be used in regular code

.vb
  if (PetscHasAttribute(some_attribute)) {
    foo();
  } else {
    bar();
  }
.ve

  Level: intermediate

.seealso: `PetscDefined()`, `PetscLikely()`, `PetscUnlikely()`, `PETSC_ATTRIBUTE_FORMAT`
M*/
#if !defined(__has_attribute)
#  define __has_attribute(x) 0
#endif
#define PetscHasAttribute(name) __has_attribute(name)

/*MC
  PETSC_ATTRIBUTE_FORMAT - Indicate to the compiler that specified arguments should be treated
  as format specifiers and checked for validity

  Synopsis:
  #include <petscmacros.h>
  <attribute declaration> PETSC_ATTRIBUTE_FORMAT(int strIdx, int vaArgIdx)

  Input Parameters:
+ strIdx   - The (1-indexed) location of the format string in the argument list
- vaArgIdx - The (1-indexed) location of the first formattable argument in the argument list

  Notes:
  This function attribute causes the compiler to issue warnings when the format specifier does
  not match the type of the variable that will be formatted, or when there exists a mismatch
  between the number of format specifiers and variables to be formatted. It is safe to use this
  macro if your compiler does not support format specifier checking (though this is
  exceeedingly rare).

  Both strIdx and vaArgIdx must be compile-time constant integer literals and cannot have the
  same value.

  The arguments to be formatted (and therefore checked by the compiler) must be "contiguous" in
  the argument list, that is, there is no way to indicate gaps which should not be checked.

  Definition is suppressed by defining PETSC_SKIP_ATTRIBUTE_FORMAT prior to including PETSc
  header files. In this case the macro will expand empty.

  Example Usage:
.vb
  // format string is 2nd argument, variable argument list containing args is 3rd argument
  void my_printf(void *obj, const char *fmt_string, ...) PETSC_ATTRIBUTE_FORMAT(2,3)

  int    x = 1;
  double y = 50.0;

  my_printf(NULL,"%g",x);      // WARNING, format specifier does not match for 'int'!
  my_printf(NULL,"%d",x,y);    // WARNING, more arguments than format specifiers!
  my_printf(NULL,"%d %g",x,y); // OK
.ve

  Level: developer

.seealso: `PETSC_ATTRIBUTE_COLD`, `PetscHasAttribute()`
M*/
#if PetscHasAttribute(format) && !defined(PETSC_SKIP_ATTRIBUTE_FORMAT)
#  define PETSC_ATTRIBUTE_FORMAT(strIdx,vaArgIdx) __attribute__((format(printf,strIdx,vaArgIdx)))
#else
#  define PETSC_ATTRIBUTE_FORMAT(strIdx,vaArgIdx)
#endif

/*MC
  PETSC_ATTRIBUTE_COLD - Indicate to the compiler that a function is very unlikely to be
  executed

  Notes:
  The marked function is often optimized for size rather than speed and may be grouped alongside
  other equally frigid routines improving code locality of lukewarm or hotter parts of program.

  The paths leading to cold functions are usually automatically marked as unlikely by the
  compiler. It may thus be useful to mark functions used to handle unlikely conditions -- such
  as error handlers -- as cold to improve optimization of the surrounding temperate functions.

  Example Usage:
.vb
  void my_error_handler(...) PETSC_ATTRIBUTE_COLD;

  if (temperature < 0) {
    return my_error_handler(...); // chilly!
  }
.ve

  Level: intermediate

.seealso: `PetscUnlikely()`, `PetscUnlikelyDebug()`, `PetscLikely()`, `PetscLikelyDebug()`,
          `PetscUnreachable()`, `PETSC_ATTRIBUTE_FORMAT`
M*/
#if PetscHasAttribute(__cold__)
#  define PETSC_ATTRIBUTE_COLD __attribute__((__cold__))
#elif PetscHasAttribute(cold) /* some implementations (old gcc) use no underscores */
#  define PETSC_ATTRIBUTE_COLD __attribute__((cold))
#else
#  define PETSC_ATTRIBUTE_COLD
#endif

/*MC
  PETSC_NULLPTR - Standard way of indicating a null value or pointer

  Notes:
  Equivalent to NULL in C source, and nullptr in C++ source. Note that for the purposes of
  interoperability between C and C++, setting a pointer to PETSC_NULLPTR in C++ is functonially
  equivalent to setting the same pointer to NULL in C. That is to say that the following
  expressions are equivalent\:

.vb
  ptr == PETSC_NULLPTR
  ptr == NULL
  ptr == 0
  !ptr

  ptr = PETSC_NULLPTR
  ptr = NULL
  ptr = 0
.ve

  and for completeness' sake\:

.vb
  PETSC_NULLPTR == NULL
.ve

  Fortran Notes:
  Not available in Fortran

  Example Usage:
.vb
  // may be used in place of '\0' or other such teminators in the definition of char arrays
  const char *const MyEnumTypes[] = {
    "foo",
    "bar",
    PETSC_NULLPTR
  };

  // may be used to nullify objects
  PetscObject obj = PETSC_NULLPTR;

  // may be used in any function expecting NULL
  PetscInfo(PETSC_NULLPTR,"Lorem Ipsum Dolor");
.ve

  Developer Notes:
  PETSC_NULLPTR must be used in place of NULL in all C++ source files. Using NULL in source
  files compiled with a C++ compiler may lead to unexpected side-effects in function overload
  resolution and/or compiler warnings.

  Level: beginner

.seealso: `PETSC_CONSTEXPR_14`, `PETSC_NODISCARD`
M*/

/*MC
  PETSC_CONSTEXPR_14 - C++14 constexpr

  Notes:
  Equivalent to constexpr when using a C++ compiler that supports C++14. Expands to nothing
  if the C++ compiler does not suppport C++14 or when not compiling with a C++ compiler. Note
  that this cannot be used in cases where an empty expansion would result in invalid code. It
  is safe to use this in C source files.

  Fortran Notes:
  Not available in Fortran

  Example Usage:
.vb
  PETSC_CONSTEXPR_14 int factorial(int n)
  {
    int r = 1;

    do {
      r *= n;
    } while (--n);
    return r;
  }
.ve

  Level: beginner

.seealso: `PETSC_NULLPTR`, `PETSC_NODISCARD`
M*/

/*MC
  PETSC_NODISCARD - Mark the return value of a function as non-discardable

  Notes:
  Hints to the compiler that the return value of a function must be captured. A diagnostic may
  (but is not required) be emitted if the value is discarded. It is safe to use this in C
  and C++ source files.

  Fortran Notes:
  Not available in Fortran

  Example Usage:
.vb
  class Foo
  {
    int x;

  public:
    PETSC_NODISCARD Foo(int y) : x(y) { }
  };

  PETSC_NODISCARD int factorial(int n)
  {
    return n <= 1 ? 1 : (n * factorial(n - 1));
  }

  auto x = factorial(10); // OK, capturing return value
  factorial(10);          // Warning: ignoring return value of function declared 'nodiscard'

  auto f = Foo(x); // OK, capturing constructed object
  Foo(x);          // Warning: Ignoring temporary created by a constructor declared 'nodiscard'
.ve

  Developer Notes:
  It is highly recommended if not downright required that any PETSc routines written in C++
  returning a PetscErrorCode be marked PETSC_NODISCARD. Ignoring the return value of PETSc
  routines is not supported; unhandled errors may leave PETSc in an unrecoverable state.

  Level: beginner

.seealso: `PETSC_NULLPTR`, `PETSC_CONSTEXPR_14`
M*/

/* C++11 features */
#if defined(__cplusplus)
#  define PETSC_NULLPTR nullptr
#else
#  define PETSC_NULLPTR NULL
#endif

/* C++14 features */
#if defined(__cplusplus) && defined(PETSC_HAVE_CXX_DIALECT_CXX14) && __cplusplus >= 201402L
#  define PETSC_CONSTEXPR_14 constexpr
#else
#  define PETSC_CONSTEXPR_14
#endif

/* C++17 features */
/* We met cases that the host CXX compiler (say mpicxx) supports C++17, but nvcc does not
 * agree, even with -ccbin mpicxx! */
#if defined(__cplusplus) && defined(PETSC_HAVE_CXX_DIALECT_CXX17) && (!defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_CUDA_DIALECT_CXX17))
#  define PETSC_NODISCARD [[nodiscard]]
#else
#  if PetscHasAttribute(warn_unused_result)
#    define PETSC_NODISCARD __attribute__((warn_unused_result))
#  else
#    define PETSC_NODISCARD
#  endif
#endif

#include <petscversion.h>
#define PETSC_AUTHOR_INFO  "       The PETSc Team\n    petsc-maint@mcs.anl.gov\n https://petsc.org/\n"

/* designated initializers since C99 and C++20, MSVC never supports them though */
#if defined(_MSC_VER) || (defined(__cplusplus) && (__cplusplus < 202002L))
#  define PetscDesignatedInitializer(name,...) __VA_ARGS__
#else
#  define PetscDesignatedInitializer(name,...) .name = __VA_ARGS__
#endif

/*MC
  PetscUnlikely - Hints the compiler that the given condition is usually FALSE

  Synopsis:
  #include <petscmacros.h>
  bool PetscUnlikely(bool cond)

  Not Collective

  Input Parameter:
. cond - Boolean expression

  Notes:
  Not available from fortran.

  This returns the same truth value, it is only a hint to compilers that the result of cond is
  unlikely to be true.

  Example usage:
.vb
  if (PetscUnlikely(cond)) {
    foo(); // cold path
  } else {
    bar(); // hot path
  }
.ve

  Level: advanced

.seealso: `PetscLikely()`, `PetscUnlikelyDebug()`, `PetscCall()`, `PetscDefined()`, `PetscHasAttribute()`,
          `PETSC_ATTRIBUTE_COLD`
M*/

/*MC
  PetscLikely - Hints the compiler that the given condition is usually TRUE

  Synopsis:
  #include <petscmacros.h>
  bool PetscLikely(bool cond)

  Not Collective

  Input Parameter:
. cond - Boolean expression

  Notes:
  Not available from fortran.

  This returns the same truth value, it is only a hint to compilers that the result of cond is
  likely to be true.

  Example usage:
.vb
  if (PetscLikely(cond)) {
    foo(); // hot path
  } else {
    bar(); // cold path
  }
.ve

  Level: advanced

.seealso: `PetscUnlikely()`, `PetscDefined()`, `PetscHasAttribute()`
          `PETSC_ATTRIBUTE_COLD`
M*/
#if defined(PETSC_HAVE_BUILTIN_EXPECT)
#  define PetscUnlikely(cond) __builtin_expect(!!(cond),0)
#  define PetscLikely(cond)   __builtin_expect(!!(cond),1)
#else
#  define PetscUnlikely(cond) (cond)
#  define PetscLikely(cond)   (cond)
#endif

/*MC
  PetscUnreachable - Indicate to the compiler that a code-path is logically unreachable

  Synopsis:
  #include <petscmacros.h>
  void PetscUnreachable(void)

  Notes:
  Indicates to the compiler (usually via some built-in) that a particular code path is always
  unreachable. Behavior is undefined if this function is ever executed, the user can expect an
  unceremonious crash.

  Example usage:
  Useful in situations such as switches over enums where not all enumeration values are
  explicitly covered by the switch

.vb
  typedef enum {RED, GREEN, BLUE} Color;

  int foo(Color c)
  {
    // it is known to programmer (or checked previously) that c is either RED or GREEN
    // but compiler may not be able to deduce this and/or emit spurious warnings
    switch (c) {
      case RED:
        return bar();
      case GREEN:
        return baz();
      default:
        PetscUnreachable(); // program is ill-formed if executed
    }
  }
.ve

  Level: advanced

.seealso: `SETERRABORT()`, `PETSCABORT()`, `PETSC_ATTRIBUTE_COLD`
M*/
#if defined(__GNUC__)
/* GCC 4.8+, Clang, Intel and other compilers compatible with GCC (-std=c++0x or above) */
#  define PetscUnreachable() __builtin_unreachable()
#elif defined(_MSC_VER) /* MSVC */
#  define PetscUnreachable() __assume(0)
#else /* ??? */
#  define PetscUnreachable() SETERRABORT(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Code path explicitly marked as unreachable executed")
#endif

/*MC
  PetscExpand - Expand macro argument

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PetscExpand(x)

  Input Parameter:
. x - The preprocessor token to expand

  Level: beginner

.seealso: `PetscStringize()`, `PetscConcat()`
M*/
#define PetscExpand_(...) __VA_ARGS__
#define PetscExpand(...)  PetscExpand_(__VA_ARGS__)

/*MC
  PetscStringize - Stringize a token

  Synopsis:
  #include <petscmacros.h>
  const char* PetscStringize(x)

  Input Parameter:
. x - The token you would like to stringize

  Output Parameter:
. <return-value> - The string representation of x

  Notes:
  Not available from Fortran.

  PetscStringize() expands x before stringizing it, if you do not wish to do so, use
  PetscStringize_() instead.

  Example Usage:
.vb
  #define MY_OTHER_VAR hello there
  #define MY_VAR       MY_OTHER_VAR

  PetscStringize(MY_VAR)  -> "hello there"
  PetscStringize_(MY_VAR) -> "MY_VAR"

  int foo;
  PetscStringize(foo)  -> "foo"
  PetscStringize_(foo) -> "foo"
.ve

  Level: beginner

.seealso: `PetscConcat()`, `PetscExpandToNothing()`, `PetscExpand()`
M*/
#define PetscStringize_(x) #x
#define PetscStringize(x)  PetscStringize_(x)

/*MC
  PetscConcat - Concatenate two tokens

  Synopsis:
  #include <petscmacros.h>
  <macro-expansion> PetscConcat(x, y)

  Input Parameters:
+ x - First token
- y - Second token

  Notes:
  Not available from Fortran.

  PetscConcat() will expand both arguments before pasting them together, use PetscConcat_()
  if you don't want to expand them.

  Example usage:
.vb
  PetscConcat(hello,there) -> hellothere

  #define HELLO hello
  PetscConcat(HELLO,there)  -> hellothere
  PetscConcat_(HELLO,there) -> HELLOthere
.ve

  Level: beginner

.seealso: `PetscStringize()`, `PetscExpand()`
M*/
#define PetscConcat_(x,y) x ## y
#define PetscConcat(x,y)  PetscConcat_(x,y)

#define PETSC_INTERNAL_COMPL_0 1
#define PETSC_INTERNAL_COMPL_1 0

/*MC
  PetscCompl - Expands to the integer complement of its argument

  Synopsis:
  #include <petscmacros.h>
  int PetscCompl(b)

  Input Parameter:
. b - Preprocessor variable, must expand to either integer literal 0 or 1

  Output Parameter:
. <return-value> - Either integer literal 0 or 1

  Notes:
  Not available from Fortran.

  Expands to integer literal 0 if b expands to 1, or integer literal 1 if b expands to
  0. Behaviour is undefined if b expands to anything else. PetscCompl() will expand its
  argument before returning the complement.

  This macro can be useful for negating PetscDefined() inside macros e.g.

$ #define PETSC_DONT_HAVE_FOO PetscCompl(PetscDefined(HAVE_FOO))

  Example usage:
.vb
  #define MY_VAR 1
  PetscCompl(MY_VAR) -> 0

  #undef  MY_VAR
  #define MY_VAR 0
  PetscCompl(MY_VAR) -> 1
.ve

  Level: beginner

.seealso: `PetscConcat()`, `PetscDefined()`
M*/
#define PetscCompl(b) PetscConcat_(PETSC_INTERNAL_COMPL_,PetscExpand(b))

#if !defined(PETSC_SKIP_VARIADIC_MACROS)
/*MC
  PetscDefined - Determine whether a boolean macro is defined

  Synopsis:
  #include <petscmacros.h>
  int PetscDefined(def)

  Input Parameter:
. def - PETSc-style preprocessor variable (without PETSC_ prepended!)

  Outut Parameter:
. <return-value> - Either integer literal 0 or 1

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  PetscDefined() returns 1 if and only if "PETSC_ ## def" is defined (but empty) or defined to
  integer literal 1. In all other cases, PetscDefined() returns integer literal 0. Therefore
  this macro should not be used if its argument may be defined to a non-empty value other than
  1.

  The prefix "PETSC_" is automatically prepended to def. To avoid prepending "PETSC_", say to
  add custom checks in user code, one should use PetscDefined_().

$ #define FooDefined(d) PetscDefined_(PetscConcat(FOO_,d))

  Developer Notes:
  Getting something that works in C and CPP for an arg that may or may not be defined is
  tricky. Here, if we have "#define PETSC_HAVE_BOOGER 1" we match on the placeholder define,
  insert the "0," for arg1 and generate the triplet (0, 1, 0). Then the last step cherry picks
  the 2nd arg (a one). When PETSC_HAVE_BOOGER is not defined, we generate a (... 1, 0) pair,
  and when the last step cherry picks the 2nd arg, we get a zero.

  Our extra expansion via PetscDefined__take_second_expand() is needed with MSVC, which has a
  nonconforming implementation of variadic macros.

  Example Usage:
  Suppose you would like to call either "foo()" or "bar()" depending on whether PETSC_USE_DEBUG
  is defined then

.vb
  #if PetscDefined(USE_DEBUG)
    foo();
  #else
    bar();
  #endif

  // or alternatively within normal code
  if (PetscDefined(USE_DEBUG)) {
    foo();
  } else {
    bar();
  }
.ve

  is equivalent to

.vb
  #if defined(PETSC_USE_DEBUG)
  #  if MY_DETECT_EMPTY_MACRO(PETSC_USE_DEBUG) // assuming you have such a macro
       foo();
  #   elif PETSC_USE_DEBUG == 1
       foo();
  #   else
       bar();
  #  endif
  #else
  bar();
  #endif
.ve

  Level: intermediate

.seealso: `PetscHasAttribute()`, `PetscUnlikely()`, `PetscLikely()`, `PetscConcat()`,
          `PetscExpandToNothing()`, `PetscCompl()`
M*/
#define PetscDefined_arg_1 shift,
#define PetscDefined_arg_  shift,
#define PetscDefined__take_second_expanded(ignored, val, ...) val
#define PetscDefined__take_second_expand(args) PetscDefined__take_second_expanded args
#define PetscDefined__take_second(...) PetscDefined__take_second_expand((__VA_ARGS__))
#define PetscDefined__(arg1_or_junk)   PetscDefined__take_second(arg1_or_junk 1, 0, at_)
#define PetscDefined_(value)           PetscDefined__(PetscConcat_(PetscDefined_arg_,value))
#define PetscDefined(def)              PetscDefined_(PetscConcat(PETSC_,def))

/*MC
  PetscUnlikelyDebug - Hints the compiler that the given condition is usually FALSE, eliding
  the check in optimized mode

  Synopsis:
  #include <petscmacros.h>
  bool PetscUnlikelyDebug(bool cond)

  Not Collective

  Input Parameters:
. cond - Boolean expression

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  This returns the same truth value, it is only a hint to compilers that the result of cond is
  likely to be false. When PETSc is compiled in optimized mode this will always return
  false. Additionally, cond is guaranteed to not be evaluated when PETSc is compiled in
  optimized mode.

  Example usage:
  This routine is shorthand for checking both the condition and whether PetscDefined(USE_DEBUG)
  is true. So

.vb
  if (PetscUnlikelyDebug(cond)) {
    foo();
  } else {
    bar();
  }
.ve

  is equivalent to

.vb
  if (PetscDefined(USE_DEBUG)) {
    if (PetscUnlikely(cond)) {
      foo();
    } else {
      bar();
    }
  } else {
    bar();
  }
.ve

  Level: advanced

.seealso: `PetscUnlikely()`, `PetscLikely()`, `PetscCall()`, `SETERRQ`
M*/
#define PetscUnlikelyDebug(cond) (PetscDefined(USE_DEBUG) && PetscUnlikely(cond))

#if defined(PETSC_CLANG_STATIC_ANALYZER)
// silence compiler warnings when using -pedantic, this is only used by the linter and it cares
// not what ISO C allows
#  define PetscMacroReturns_(retexpr,...) __extension__ ({ __VA_ARGS__; retexpr; })
#else
#  define PetscMacroReturns_(retexpr,...) retexpr; do { __VA_ARGS__; } while (0)
#endif

/*MC
  PetscExpandToNothing - Expands to absolutely nothing at all

  Synopsis:
  #include <petscmacros.h>
  void PetscExpandToNothing(...)

  Input Parameter:
. __VA_ARGS__ - Anything at all

  Notes:
  Not available from Fortran, requires variadic macro support, definition is disabled by
  defining PETSC_SKIP_VARIADIC_MACROS.

  Must have at least 1 parameter.

  Example usage:
.vb
  PetscExpandToNothing(a,b,c) -> *nothing*
.ve

  Level: beginner

.seealso: `PetscConcat()`, `PetscDefined()`, `PetscStringize()`, `PetscExpand()`
M*/
#define PetscExpandToNothing(...)

/*MC
  PetscMacroReturns - Define a macro body that returns a value

  Synopsis:
  #include <petscmacros.h>
  return_type PetscMacroReturns(return_type retexpr, ...)

  Input Parameters:
+ retexpr     - The value or expression that the macro should return
- __VA_ARGS__ - The body of the macro

  Notes:
  Due to limitations of the C-preprocessor retexpr cannot depend on symbols declared in the
  body of the macro and should not depend on values produced as a result of the expression. The
  user should not assume that the result of this macro is equivalent to a single logical source
  line. It is not portable to use macros defined using this one in conditional or loop bodies
  without enclosing them in curly braces\:

.vb
  #define FOO(arg1) PetscMacroReturns(0,arg1+=10) // returns 0

  int err,x = 10;

  if (...) err = FOO(x);      // ERROR, body of FOO() executed outside the if statement
  if (...) { err = FOO(x); }  // OK

  for (...) err = FOO(x);     // ERROR, body of FOO() executed outside the loop
  for (...) { err = FOO(x); } // OK
.ve

  It is also not portable to use this macro directly inside function call, conditional, loop,
  or switch statements\:

.vb
  extern void bar(int);

  int ret = FOO(x);

  bar(FOO(x)); // ERROR, may not compile
  bar(ret);    // OK

  if (FOO(x))  // ERROR, may not compile
  if (ret)     // OK
.ve

  Example usage:
.vb
  #define MY_SIMPLE_RETURNING_MACRO(arg1) PetscMacroReturns(0,arg1+=10)

  int x = 10;
  int err = MY_SIMPLE_RETURNING_MACRO(x); // err = 0, x = 20

  // multiline macros allowed, but must declare with line continuation as usual
  #define MY_COMPLEX_RETURNING_MACRO(arg1) PetscMacroReturns(0, \
    if (arg1 > 10) {                                            \
      puts("big int!");                                         \
    } else {                                                    \
      return 7355608;                                           \
    }                                                           \
  )

  // if retexpr contains commas, must enclose it with braces
  #define MY_COMPLEX_RETEXPR_MACRO_1() PetscMacroReturns(x+=10,0,body...)
  #define MY_COMPLEX_RETEXPR_MACRO_2() PetscMacroReturns((x+=10,0),body...)

  int x = 10;
  int y = MY_COMPLEX_RETEXPR_MACRO_1(); // ERROR, y = x = 20 not 0
  int z = MY_COMPLEX_RETEXPR_MACRO_2(); // OK, y = 0, x = 20
.ve

  Level: intermediate

.seealso: `PetscExpand()`, `PetscConcat()`, `PetscStringize()`
M*/
#define PetscMacroReturns(retexpr,...) PetscMacroReturns_(retexpr,__VA_ARGS__)

#define PetscMacroReturnStandard(...) PetscMacroReturns(0,__VA_ARGS__)

#endif /* !PETSC_SKIP_VARIADIC_MACROS */

/*MC
  PETSC_STATIC_ARRAY_LENGTH - Return the length of a static array

  Level: intermediate
M*/
#define PETSC_STATIC_ARRAY_LENGTH(a) (sizeof(a)/sizeof((a)[0]))

#endif /* PETSC_PREPROCESSOR_MACROS_H */
