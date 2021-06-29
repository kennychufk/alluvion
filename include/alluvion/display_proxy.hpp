#ifndef ALLUVION_DISPLAY_PROXY_HPP
#define ALLUVION_DISPLAY_PROXY_HPP

#include <glm/gtc/type_ptr.hpp>

#include "alluvion/display.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/solver.hpp"

namespace alluvion {
// Motivations
// 1. cannot pass Display* to Python (because not copy-constructible?)
// 2. avoid circular dependencies: Pile -> Store -> Display
class DisplayProxy {
 public:
  DisplayProxy(Display* display);
  GLuint create_colormap_viridis();
  template <typename TF>
  void add_particle_shading_program(GLuint x_vbo, GLuint attr_vbo,
                                    GLuint colormap_tex, float particle_radius,
                                    Solver<TF> const& solver) {
#include "alluvion/glsl/particle.frag"
#include "alluvion/glsl/particle.vert"
    display_->add_shading_program(new ShadingProgram(
        kParticleVertexShaderStr, kParticleFragmentShaderStr,
        {"particle_radius", "screen_dimension", "M", "V", "P",
         "camera_worldspace", "material.specular", "material.shininess",
         "directional_light.direction", "directional_light.ambient",
         "directional_light.diffuse", "directional_light.specular",
         "point_lights[0].position", "point_lights[0].constant",
         "point_lights[0].linear", "point_lights[0].quadratic",
         "point_lights[0].ambient", "point_lights[0].diffuse",
         "point_lights[0].specular",
         //
         "point_lights[1].position", "point_lights[1].constant",
         "point_lights[1].linear", "point_lights[1].quadratic",
         "point_lights[1].ambient", "point_lights[1].diffuse",
         "point_lights[1].specular"

        },
        {std::make_tuple(x_vbo, 3, 0), std::make_tuple(attr_vbo, 1, 0)},
        [x_vbo, attr_vbo, colormap_tex, particle_radius, &solver](
            ShadingProgram& program, Display& display) {
          glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
          glUniformMatrix4fv(program.get_uniform_location("M"), 1, GL_FALSE,
                             glm::value_ptr(glm::mat4(1)));
          glUniformMatrix4fv(
              program.get_uniform_location("P"), 1, GL_FALSE,
              glm::value_ptr(display.camera_.getProjectionMatrix()));
          glUniformMatrix4fv(program.get_uniform_location("V"), 1, GL_FALSE,
                             glm::value_ptr(display.camera_.getViewMatrix()));
          glUniform2f(program.get_uniform_location("screen_dimension"),
                      static_cast<GLfloat>(display.width_),
                      static_cast<GLfloat>(display.height_));
          glUniform1f(program.get_uniform_location("particle_radius"),
                      particle_radius);

          glm::vec3 const& camera_worldspace = display.camera_.getCenter();
          glUniform3f(program.get_uniform_location("camera_worldspace"),
                      camera_worldspace[0], camera_worldspace[1],
                      camera_worldspace[2]);
          glUniform3f(
              program.get_uniform_location("directional_light.direction"), 0.2f,
              1.0f, 0.3f);
          glUniform3f(program.get_uniform_location("directional_light.ambient"),
                      0.05f, 0.05f, 0.05f);
          glUniform3f(program.get_uniform_location("directional_light.diffuse"),
                      0.4f, 0.4f, 0.4f);
          glUniform3f(
              program.get_uniform_location("directional_light.specular"), 0.5f,
              0.5f, 0.5f);

          glUniform3f(program.get_uniform_location("point_lights[0].position"),
                      2.0f, 2.0f, 2.0f);
          glUniform1f(program.get_uniform_location("point_lights[0].constant"),
                      1.0f);
          glUniform1f(program.get_uniform_location("point_lights[0].linear"),
                      0.09f);
          glUniform1f(program.get_uniform_location("point_lights[0].quadratic"),
                      0.032f);
          glUniform3f(program.get_uniform_location("point_lights[0].ambient"),
                      0.05f, 0.05f, 0.05f);
          glUniform3f(program.get_uniform_location("point_lights[0].diffuse"),
                      0.8f, 0.8f, 0.8f);
          glUniform3f(program.get_uniform_location("point_lights[0].specular"),
                      1.0f, 1.0f, 1.0f);

          glUniform3f(program.get_uniform_location("point_lights[1].position"),
                      2.0f, 1.0f, -2.0f);
          glUniform1f(program.get_uniform_location("point_lights[1].constant"),
                      1.0f);
          glUniform1f(program.get_uniform_location("point_lights[1].linear"),
                      0.09f);
          glUniform1f(program.get_uniform_location("point_lights[1].quadratic"),
                      0.032f);
          glUniform3f(program.get_uniform_location("point_lights[1].ambient"),
                      0.05f, 0.05f, 0.05f);
          glUniform3f(program.get_uniform_location("point_lights[1].diffuse"),
                      0.8f, 0.8f, 0.8f);
          glUniform3f(program.get_uniform_location("point_lights[1].specular"),
                      1.0f, 1.0f, 1.0f);
          glUniform3f(program.get_uniform_location("material.specular"), 0.8,
                      0.9, 0.9);
          glUniform1f(program.get_uniform_location("material.shininess"), 5.0);

          glBindTexture(GL_TEXTURE_1D, colormap_tex);
          glDrawArrays(GL_POINTS, 0, solver.num_particles);
        }));
  };

  template <typename TF3, typename TQ, typename TF>
  void add_pile_shading_program(Pile<TF3, TQ, TF> const& pile) {
    // rigid mesh shader
    // https://github.com/opengl-tutorials/ogl
#include "alluvion/glsl/mesh_with_normal.frag"
#include "alluvion/glsl/mesh_with_normal.vert"
    display_->add_shading_program(new ShadingProgram(
        kMeshWithNormalVertexShaderStr, kMeshWithNormalFragmentShaderStr,
        {"MVP", "V", "M", "camera_worldspace", "material.diffuse",
         "material.specular", "material.shininess",
         "directional_light.direction", "directional_light.ambient",
         "directional_light.diffuse", "directional_light.specular",
         "point_lights[0].position", "point_lights[0].constant",
         "point_lights[0].linear", "point_lights[0].quadratic",
         "point_lights[0].ambient", "point_lights[0].diffuse",
         "point_lights[0].specular",
         //
         "point_lights[1].position", "point_lights[1].constant",
         "point_lights[1].linear", "point_lights[1].quadratic",
         "point_lights[1].ambient", "point_lights[1].diffuse",
         "point_lights[1].specular"},
        {std::make_tuple(0, 3, 0), std::make_tuple(0, 3, 0)},
        [&pile](ShadingProgram& program, Display& display) {
          glm::mat4 const& projection_matrix =
              display.camera_.getProjectionMatrix();
          glm::mat4 const& view_matrix = display.camera_.getViewMatrix();
          glUniformMatrix4fv(program.get_uniform_location("V"), 1, GL_FALSE,
                             glm::value_ptr(view_matrix));
          glm::vec3 const& camera_worldspace = display.camera_.getCenter();
          glUniform3f(program.get_uniform_location("camera_worldspace"),
                      camera_worldspace[0], camera_worldspace[1],
                      camera_worldspace[2]);

          glUniform3f(
              program.get_uniform_location("directional_light.direction"), 0.2f,
              1.0f, 0.3f);
          glUniform3f(program.get_uniform_location("directional_light.ambient"),
                      0.05f, 0.05f, 0.05f);
          glUniform3f(program.get_uniform_location("directional_light.diffuse"),
                      0.4f, 0.4f, 0.4f);
          glUniform3f(
              program.get_uniform_location("directional_light.specular"), 0.5f,
              0.5f, 0.5f);

          glUniform3f(program.get_uniform_location("point_lights[0].position"),
                      2.0f, 2.0f, 2.0f);
          glUniform1f(program.get_uniform_location("point_lights[0].constant"),
                      1.0f);
          glUniform1f(program.get_uniform_location("point_lights[0].linear"),
                      0.09f);
          glUniform1f(program.get_uniform_location("point_lights[0].quadratic"),
                      0.032f);
          glUniform3f(program.get_uniform_location("point_lights[0].ambient"),
                      0.05f, 0.05f, 0.05f);
          glUniform3f(program.get_uniform_location("point_lights[0].diffuse"),
                      0.8f, 0.8f, 0.8f);
          glUniform3f(program.get_uniform_location("point_lights[0].specular"),
                      1.0f, 1.0f, 1.0f);

          glUniform3f(program.get_uniform_location("point_lights[1].position"),
                      2.0f, 1.0f, -2.0f);
          glUniform1f(program.get_uniform_location("point_lights[1].constant"),
                      1.0f);
          glUniform1f(program.get_uniform_location("point_lights[1].linear"),
                      0.09f);
          glUniform1f(program.get_uniform_location("point_lights[1].quadratic"),
                      0.032f);
          glUniform3f(program.get_uniform_location("point_lights[1].ambient"),
                      0.05f, 0.05f, 0.05f);
          glUniform3f(program.get_uniform_location("point_lights[1].diffuse"),
                      0.8f, 0.8f, 0.8f);
          glUniform3f(program.get_uniform_location("point_lights[1].specular"),
                      1.0f, 1.0f, 1.0f);

          for (U i = 0; i < pile.get_size(); ++i) {
            MeshBuffer const& mesh_buffer = pile.mesh_buffer_list_[i];
            if (mesh_buffer.vertex != 0 && mesh_buffer.index != 0) {
              glm::mat4 model_matrix = pile.get_matrix(i);
              glm::mat4 mvp_matrix =
                  projection_matrix * view_matrix * model_matrix;

              glUniformMatrix4fv(program.get_uniform_location("M"), 1, GL_FALSE,
                                 glm::value_ptr(model_matrix));
              glUniformMatrix4fv(program.get_uniform_location("MVP"), 1,
                                 GL_FALSE, glm::value_ptr(mvp_matrix));
              glUniform3f(program.get_uniform_location("material.diffuse"), 0.2,
                          0.3, 0.87);
              glUniform3f(program.get_uniform_location("material.specular"),
                          0.8, 0.9, 0.9);
              glUniform1f(program.get_uniform_location("material.shininess"),
                          5.0);

              glBindBuffer(GL_ARRAY_BUFFER, mesh_buffer.vertex);
              glEnableVertexAttribArray(0);
              glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
              glBindBuffer(GL_ARRAY_BUFFER, mesh_buffer.normal);
              glEnableVertexAttribArray(1);
              glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
              glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_buffer.index);
              glDrawElements(GL_TRIANGLES, mesh_buffer.num_indices,
                             GL_UNSIGNED_INT, 0);
              glDisableVertexAttribArray(0);
              glDisableVertexAttribArray(1);
            }
          }
        }));
  }
  void run();
  void set_camera(float3 camera_pos, float3 center);

  Display* display_;
};
}  // namespace alluvion

#endif /*  ALLUVION_DISPLAY_PROXY_HPP */
