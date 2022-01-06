#ifndef ALLUVION_DISPLAY_PROXY_HPP
#define ALLUVION_DISPLAY_PROXY_HPP

#include <glm/gtc/type_ptr.hpp>
#include <regex>

#include "alluvion/colormaps.hpp"
#include "alluvion/display.hpp"
#include "alluvion/pile.hpp"
#include "alluvion/solver.hpp"
#include "alluvion/solver_df.hpp"

namespace alluvion {
// Motivations
// 1. cannot pass Display* to Python (because not copy-constructible?)
// 2. avoid circular dependencies: Pile -> Store -> Display
template <typename TF>
class DisplayProxy {
 public:
  typedef std::conditional_t<std::is_same_v<TF, float>, float3, double3> TF3;
  typedef std::conditional_t<std::is_same_v<TF, float>, float4, double4> TQ;
  DisplayProxy(Display* display) : display_(display) {}
  GLuint create_colormap_viridis() {
    return display_->create_colormap(kViridisData.data(), kViridisData.size());
  }
  void add_map_graphical_pointers(Store& store) {
    display_->add_shading_program(
        new ShadingProgram(nullptr, nullptr, {}, {},
                           [&store](ShadingProgram& program, Display& display) {
                             store.map_graphical_pointers();
                           }));
  }
  void add_unmap_graphical_pointers(Store& store) {
    display_->add_shading_program(
        new ShadingProgram(nullptr, nullptr, {}, {},
                           [&store](ShadingProgram& program, Display& display) {
                             store.unmap_graphical_pointers();
                           }));
  }
  template <typename TSolver>
  void add_step(TSolver& solver, U num_steps) {
    display_->add_shading_program(new ShadingProgram(
        nullptr, nullptr, {}, {},
        [&solver, num_steps](ShadingProgram& program, Display& display) {
          for (U i = 0; i < num_steps; ++i) {
            solver.template step<0>();
          }
        }));
  }
  template <typename TVariable>
  void add_normalize(Solver<TF>& solver, TVariable const* v,
                     Variable<1, TF>* particle_normalized_attr, TF lower_bound,
                     TF upper_bound) {
    display_->add_shading_program(new ShadingProgram(
        nullptr, nullptr, {}, {},
        [&solver, v, particle_normalized_attr, lower_bound, upper_bound](
            ShadingProgram& program, Display& display) {
          solver.normalize(v, particle_normalized_attr, lower_bound,
                           upper_bound);
        }));
  }
  void add_clear() {
    display_->add_shading_program(
        new ShadingProgram(nullptr, nullptr, {}, {},
                           [](ShadingProgram& program, Display& display) {
                             glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                           }));
  }
  void add_particle_shading_program(Variable<1, TF3> const& x,
                                    Variable<1, TF> const& attr,
                                    GLuint colormap_tex, float particle_radius,
                                    Solver<TF> const& solver) {
#include "alluvion/glsl/particle.frag"
#include "alluvion/glsl/particle.vert"
    std::string vector3_str = "vec3";
    std::string fp_str = "float";
    std::string arb_gpu_shader_fp64_str = "";
    std::string arb_vertex_attrib_64bit_str = "";
    std::string tf3_to_float3_str = "$1";
    std::string tf_to_float_str = "$1";
    GLenum attribute_type = GL_FLOAT;
    if constexpr (std::is_same_v<TF, double>) {
      vector3_str = "dvec3";
      fp_str = "double";
      arb_gpu_shader_fp64_str = "#extension GL_ARB_gpu_shader_fp64 : enable";
      arb_vertex_attrib_64bit_str =
          "#extension GL_ARB_vertex_attrib_64bit : enable";
      tf3_to_float3_str = "vec3($1)";
      tf_to_float_str = "float($1)";
      attribute_type = GL_DOUBLE;
    }
    std::string typed_vertex_shader = std::regex_replace(
        kParticleVertexShaderStr, std::regex("\\bTF3\\b"), vector3_str);
    typed_vertex_shader =
        std::regex_replace(typed_vertex_shader, std::regex("\\bTF\\b"), fp_str);
    typed_vertex_shader = std::regex_replace(typed_vertex_shader,
                                             std::regex("#ARB_GPU_SHADER_FP64"),
                                             arb_gpu_shader_fp64_str);
    typed_vertex_shader = std::regex_replace(
        typed_vertex_shader, std::regex("#ARB_VERTEX_ATTRIB_64BIT"),
        arb_vertex_attrib_64bit_str);
    typed_vertex_shader = std::regex_replace(
        typed_vertex_shader,
        std::regex("TF3_TO_FLOAT3\\(([a-zA-Z_][a-zA-Z0-9_]*)\\)"),
        tf3_to_float3_str);
    typed_vertex_shader = std::regex_replace(
        typed_vertex_shader,
        std::regex("TF_TO_FLOAT\\(([a-zA-Z_][a-zA-Z0-9_]*)\\)"),
        tf_to_float_str);
    display_->add_shading_program(new ShadingProgram(
        typed_vertex_shader.c_str(), kParticleFragmentShaderStr.c_str(),
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
        {std::make_tuple(
             reinterpret_cast<GraphicalVariable<1, TF3> const&>(x).vbo_, 3,
             attribute_type, 0),
         std::make_tuple(
             reinterpret_cast<GraphicalVariable<1, TF> const&>(attr).vbo_, 1,
             attribute_type, 0)},
        [colormap_tex, particle_radius, &solver](ShadingProgram& program,
                                                 Display& display) {
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
  }

  void add_pile_shading_program(Pile<TF> const& pile) {
    // rigid mesh shader
    // https://github.com/opengl-tutorials/ogl
#include "alluvion/glsl/mesh_with_normal.frag"
#include "alluvion/glsl/mesh_with_normal.vert"
    display_->add_shading_program(new ShadingProgram(
        kMeshWithNormalVertexShaderStr.c_str(),
        kMeshWithNormalFragmentShaderStr.c_str(),
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
        {std::make_tuple(0, 3, GL_FLOAT, 0),
         std::make_tuple(0, 3, GL_FLOAT, 0)},
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
  void run() { return display_->run(); }
  void draw() { return display_->draw(); }
  void set_camera(float3 camera_pos, float3 center) {
    display_->camera_.setEye(camera_pos.x, camera_pos.y, camera_pos.z);
    display_->camera_.setCenter(center.x, center.y, center.z);
    display_->camera_.update();
    display_->update_trackball_camera();
  }
  void set_clip_planes(float near, float far) {
    display_->camera_.setClipPlanes(near, far);
    display_->update_trackball_camera();
  }
  CompleteFramebuffer* create_framebuffer() {
    GLuint fbo = display_->create_framebuffer();
    return &(display_->get_framebuffer(fbo));
  }
  void bind_framebuffer(CompleteFramebuffer& buffer) {
    glBindFramebuffer(GL_FRAMEBUFFER, buffer.fbo_);
  }
  void add_bind_framebuffer_step(CompleteFramebuffer& buffer) {
    display_->add_shading_program(new ShadingProgram(
        nullptr, nullptr, {}, {},
        [&buffer](ShadingProgram& program, Display& display) {
          glBindFramebuffer(GL_FRAMEBUFFER, buffer.fbo_);
        }));
  }
  void add_show_framebuffer_shader(CompleteFramebuffer const& buffer) {
    constexpr float kScreenQuadXYTex[] = {
        // positions   // texCoords
        -1.0f, 1.0f,  0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f,  -1.0f, 1.0f, 0.0f, -1.0f, 1.0f,  0.0f, 1.0f,
        1.0f,  -1.0f, 1.0f, 0.0f, 1.0f,  1.0f,  1.0f, 1.0f};
    GLuint screen_quad =
        display_->create_dynamic_array_buffer<float4>(6, kScreenQuadXYTex);
#include "alluvion/glsl/screen.frag"
#include "alluvion/glsl/screen.vert"
    display_->add_shading_program(new ShadingProgram(
        kScreenVertexShaderStr.c_str(), kScreenFragmentShaderStr.c_str(), {},
        {std::make_tuple(screen_quad, 4, GL_FLOAT, 0)},
        [&](ShadingProgram& program, Display& display) {
          glBindFramebuffer(GL_FRAMEBUFFER, 0);
          glDisable(GL_DEPTH_TEST);
          glClear(GL_COLOR_BUFFER_BIT);

          glBindTexture(GL_TEXTURE_2D, buffer.color_tex_);
          glDrawArrays(GL_TRIANGLES, 0, 6);
          glEnable(GL_DEPTH_TEST);
        }));
  }
  void resize(int width, int height) {
    display_->resize_callback(display_->window_, width, height);
  }
  Display* display_;
};
}  // namespace alluvion

#endif /*  ALLUVION_DISPLAY_PROXY_HPP */
