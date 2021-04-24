#ifndef ALLUVION_SHADING_PROGRAM_HPP
#define ALLUVION_SHADING_PROGRAM_HPP

#include <glad/glad.h>
// glad first
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace alluvion {

class Display;
class ShadingProgram {
 public:
  using ProgramCallback = std::function<void(ShadingProgram&, Display&)>;

 private:
  std::unordered_map<std::string, GLint> uniform_dict_;
  ProgramCallback callback_;

  static GLuint create_shader(GLenum shader_type, const char* code);
  static GLuint create_program(std::vector<GLuint> const& shaders);

 public:
  GLuint program_;
  ShadingProgram(const char* vertex_code, const char* fragment_code,
                 std::vector<std::string> uniform_names,
                 ProgramCallback callback);
  virtual ~ShadingProgram();

  void update(Display& display);
  GLint get_uniform_location(std::string const& name);
};
}  // namespace alluvion

#endif /* ALLUVION_SHADING_PROGRAM_HPP */
