#ifndef ALLUVION_SHADING_PROGRAM_HPP
#define ALLUVION_SHADING_PROGRAM_HPP

#include <glad/glad.h>
// glad first

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "alluvion/unique_vao.hpp"

namespace alluvion {

class Display;
class ShadingProgram {
 public:
  using ProgramCallback = std::function<void(ShadingProgram&, Display&)>;
  using VertexAttribSpec = std::tuple<GLuint, GLint, GLsizei>;

 private:
  std::unordered_map<std::string, GLint> uniform_dict_;
  ProgramCallback callback_;
  UniqueVao vao_;

  static GLuint create_shader(GLenum shader_type, const char* code);
  static GLuint create_program(std::vector<GLuint> const& shaders);

 public:
  GLuint program_;
  ShadingProgram() = delete;
  ShadingProgram(const char* vertex_code, const char* fragment_code,
                 std::vector<std::string> uniform_names,
                 std::vector<VertexAttribSpec> vertex_attrib_specs,
                 ProgramCallback callback);
  ShadingProgram(const ShadingProgram&) = delete;
  virtual ~ShadingProgram();

  void update(Display& display);
  GLint get_uniform_location(std::string const& name) const;
};
}  // namespace alluvion

#endif /* ALLUVION_SHADING_PROGRAM_HPP */
