#include <algorithm>
#include <iostream>
#include <tuple>
#include <utility>

#include "alluvion/shading_program.hpp"

namespace alluvion {

ShadingProgram::ShadingProgram(const char* vertex_code,
                               const char* fragment_code,
                               std::vector<std::string> uniform_names,
                               ProgramCallback callback)
    : program_(0), callback_(callback) {
  if (vertex_code && fragment_code) {
    std::vector<GLuint> shaders;
    shaders.push_back(create_shader(GL_VERTEX_SHADER, vertex_code));
    shaders.push_back(create_shader(GL_FRAGMENT_SHADER, fragment_code));
    program_ = create_program(shaders);

    for (GLuint shader : shaders) {
      glDetachShader(program_, shader);
    }
    std::for_each(shaders.begin(), shaders.end(), glDeleteShader);

    for (std::string const& uniform_name : uniform_names) {
      GLint uniform_location =
          glGetUniformLocation(program_, uniform_name.c_str());
      if (uniform_location == GL_INVALID_VALUE) {
        std::cerr << "Uniform '" << uniform_name << "' not found in program."
                  << std::endl;
      } else {
        uniform_dict_.emplace(std::piecewise_construct,
                              std::forward_as_tuple(uniform_name),
                              std::forward_as_tuple(uniform_location));
      }
    }
  }
}

ShadingProgram::~ShadingProgram() {
  if (program_ != 0) glDeleteProgram(program_);
}

GLuint ShadingProgram::create_shader(GLenum shader_type, const char* code) {
  GLuint shader = glCreateShader(shader_type);
  glShaderSource(shader, 1, &code, NULL);
  glCompileShader(shader);

  GLint status;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (status == GL_FALSE) {
    GLint info_log_length;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
    GLchar* info_log = new GLchar[info_log_length + 1];
    glGetShaderInfoLog(shader, info_log_length, NULL, info_log);
    std::cerr << "=== Problem loading shader ===" << std::endl
              << code << std::endl
              << "========" << std::endl;
    std::cerr << info_log << std::endl;
    delete[] info_log;
  }
  return shader;
}

GLuint ShadingProgram::create_program(std::vector<GLuint> const& shaders) {
  GLuint program = glCreateProgram();
  for (GLuint shader : shaders) {
    glAttachShader(program, shader);
  }
  glLinkProgram(program);

  GLint status;
  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (status == GL_FALSE) {
    GLint info_log_length;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
    GLchar* info_log = new GLchar[info_log_length + 1];
    glGetProgramInfoLog(program, info_log_length, NULL, info_log);
    std::cerr << info_log << std::endl;
    delete[] info_log;
  }
  return program;
}

void ShadingProgram::update(Display& display) {
  if (!callback_) return;
  if (program_ != 0) glUseProgram(program_);
  callback_(*this, display);
}

GLint ShadingProgram::get_uniform_location(std::string const& name) {
  return uniform_dict_[name];
}
}  // namespace alluvion
