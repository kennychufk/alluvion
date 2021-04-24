#include <chrono>
#include <glm/gtc/type_ptr.hpp>
#include <thread>

#include "alluvion/display.hpp"
#include "alluvion/shading_program.hpp"
#include "alluvion/store.hpp"

using namespace alluvion;

int main(void) {
  Store store;
  Display* display = store.create_display(800, 600, "test display");

  GraphicalVariable<1, F3> var = store.create_graphical<1, F3>({2});
  store.map_graphical_pointers();
  std::vector<F> v{0.0, 0.0, 0.0, 0.2, 0.2, 0.2};
  var.set_bytes(v.data(), v.size() * sizeof(F));
  store.unmap_graphical_pointers();

  display->add_shading_program(new ShadingProgram(
      R"CODE(
#version 330 core
layout(location = 0) in vec3 x;
uniform mat4 view_matrix;
uniform mat4 clip_matrix;
uniform vec2 screen_dimension;
uniform float point_scale;

out vec3 eyePos;
out float eyeRadius;

void main() {
  vec4 camera_space_x4 = view_matrix * vec4(x, 1.0);

  eyePos = camera_space_x4.xyz;
  eyeRadius = point_scale / -camera_space_x4.z / screen_dimension.y;
  gl_Position = clip_matrix * camera_space_x4;
  gl_PointSize = point_scale / -camera_space_x4.z;
}
)CODE",
      R"CODE(
#version 330 core
uniform vec4 base_color;

out vec4 output_color;

void main() {
  const vec3 light_direction = vec3(0.577, 0.577, 0.577);

  vec3 N;
  N.xy = gl_PointCoord * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
  float N_squared = dot(N.xy, N.xy);
  if (N_squared > 1.0) discard;
  N.z = sqrt(1.0 - N_squared);

  float diffuse = max(0.0, dot(light_direction, N));
  output_color = base_color * diffuse;
}
)CODE",
      {"view_matrix", "clip_matrix", "screen_dimension", "point_scale",
       "base_color"},
      [&var](ShadingProgram& program, Display& display) {
        glm::mat4 clip_matrix = glm::perspective(
            glm::radians(45.0f),
            display.width_ / static_cast<GLfloat>(display.height_), .01f,
            100.f);

        glUniformMatrix4fv(program.get_uniform_location("view_matrix"), 1,
                           GL_FALSE,
                           glm::value_ptr(display.camera_.getMatrix()));
        glUniformMatrix4fv(program.get_uniform_location("clip_matrix"), 1,
                           GL_FALSE, glm::value_ptr(clip_matrix));
        glUniform2f(program.get_uniform_location("screen_dimension"),
                    static_cast<GLfloat>(display.width_),
                    static_cast<GLfloat>(display.height_));
        glUniform1f(program.get_uniform_location("point_scale"), 5.2);
        glUniform4f(program.get_uniform_location("base_color"), 1.0, 1.0, 1.0,
                    1.0);

        glBindBuffer(GL_ARRAY_BUFFER, var.vbo_);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_POINTS, 0, 2);
      }));
  display->add_shading_program(
      new ShadingProgram(nullptr, nullptr, {}, nullptr));
  display->run();

  return 0;
}
