// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#pragma once

#ifndef __APPLE__
#ifndef __ANDROID__
#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/gl.h>
#else
#ifndef GLEWLIB_UNSUPPORTED
#define GLEW_STATIC
#include <GL/glew.h>
#else
#include <GLES3/gl3.h>
#endif
#endif
#else
#include <GL/glew.h>
#include <OpenGL/gl.h>
#endif

#include "libvideostitch/logging.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace VideoStitch {
namespace GPU {

static const std::string kGLSLShaderTag{"GLSL sharder compilation"};

class OGLShader {
 public:
  GLuint Program;

  OGLShader(const GLchar* vShaderCode, const GLchar* fShaderCode, const GLchar* gShaderCode = nullptr) {
    // Compile shaders
    GLuint vertex = 0;
    GLuint fragment = 0;
    GLuint geometry = 0;

    // Vertex Shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");

    // Fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");

    // Fragment Shader
    if (gShaderCode) {
      geometry = glCreateShader(GL_GEOMETRY_SHADER);
      glShaderSource(geometry, 1, &gShaderCode, NULL);
      glCompileShader(geometry);
      checkCompileErrors(geometry, "GEOMETRY");
    }

    // Shader Program
    this->Program = glCreateProgram();
    glAttachShader(this->Program, vertex);
    glAttachShader(this->Program, fragment);
    if (gShaderCode) {
      glAttachShader(this->Program, geometry);
    }
    glLinkProgram(this->Program);
    checkCompileErrors(this->Program, "PROGRAM");

    // Delete the shaders as they're linked into our program now and no longer necessery
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (gShaderCode) {
      glDeleteShader(geometry);
    }
  }

  // Uses the current shader
  void use() { glUseProgram(this->Program); }

 private:
  void checkCompileErrors(GLuint shader, std::string type) {
    GLint success;
    if (type != "PROGRAM") {
      glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
      if (!success) {
        GLint infoLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLength);
        std::vector<GLchar> infoLog(infoLength);
        glGetShaderInfoLog(shader, infoLength, &infoLength, &infoLog[0]);
        std::ostringstream oss;
        oss << "SHADER-COMPILATION-ERROR of " << type << ": " << infoLog.data();
        Logger::error(kGLSLShaderTag) << oss.str() << std::endl;
        assert(false);
      }
    } else {
      glGetProgramiv(shader, GL_LINK_STATUS, &success);
      if (!success) {
        GLint infoLength;
        glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &infoLength);
        std::vector<GLchar> infoLog(infoLength);
        glGetProgramInfoLog(shader, infoLength, NULL, &infoLog[0]);
        std::ostringstream oss;
        oss << "PROGRAM-LINKING-ERROR of " << type << ": " << infoLog.data();
        Logger::error(kGLSLShaderTag) << oss.str() << std::endl;
        assert(false);
      }
    }
  }
};

}  // namespace GPU
}  // namespace VideoStitch
