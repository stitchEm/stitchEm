// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

// Common definitions of functions that can be used in .gpu files
// to be shared between CUDA and OpenCL implementations
//
// ### COMMON ###
//

#define TOKENIZE_1(a) a
#define FUNCTION_NAME_1(a) TOKENIZE_1(a)

#define TOKENIZE_2(a, b) a##_##b
#define FUNCTION_NAME_2(a, b) TOKENIZE_2(a, b)

#define TOKENIZE_3(a, b, c) a##_##b##_##c
#define FUNCTION_NAME_3(a, b, c) TOKENIZE_3(a, b, c)

#define TOKENIZE_4(a, b, c, d) a##_##b##_##c##_##d
#define FUNCTION_NAME_4(a, b, c, d) TOKENIZE_4(a, b, c, d)

#define TOKENIZE_5(a, b, c, d, e) a##_##b##_##c##_##d##_##e
#define FUNCTION_NAME_5(a, b, c, d, e) TOKENIZE_5(a, b, c, d, e)

#define TOKENIZE_6(a, b, c, d, e, f) a##_##b##_##c##_##d##_##e##_##f
#define FUNCTION_NAME_6(a, b, c, d, e, f) TOKENIZE_6(a, b, c, d, e, f)

#define TOKENIZE_7(a, b, c, d, e, f, g) a##_##b##_##c##_##d##_##e##_##f##_##g
#define FUNCTION_NAME_7(a, b, c, d, e, f, g) TOKENIZE_7(a, b, c, d, e, f, g)

#define TOKENIZE_8(a, b, c, d, e, f, g, h) a##_##b##_##c##_##d##_##e##_##f##_##g##_##h
#define FUNCTION_NAME_8(a, b, c, d, e, f, g, h) TOKENIZE_8(a, b, c, d, e, f, g, h)

// Used in C++ backend
#define const_member

// No early return in GPU code
#define RETURN_3D_IF_INVALID_INVERSE_2D(uv)
#define RETURN_2D_IF_INVALID_INVERSE_3D(pt)
#define RETURN_3D_IF_INVALID_INVERSE_3D(pt)
