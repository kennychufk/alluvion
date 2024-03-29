#ifndef ALLUVION_DISPLAY_HPP
#define ALLUVION_DISPLAY_HPP

#include <glad/glad.h>

#include <tuple>
#include <unordered_map>
#include <utility>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <deque>
#include <memory>

#include "alluvion/complete_framebuffer.hpp"
#include "alluvion/graphical_allocator.hpp"
#include "alluvion/mesh.hpp"
#include "alluvion/shading_program.hpp"
#include "alluvion/trackball.hpp"
#include "alluvion/unique_mesh_buffer.hpp"
#include "alluvion/unique_texture.hpp"
#include "alluvion/unique_vbo.hpp"
namespace alluvion {
class Display {
 private:
  Trackball trackball_;
  std::deque<std::unique_ptr<ShadingProgram>> programs_;
  std::unordered_map<GLuint, UniqueTexture> textures_;
  std::unordered_map<GLuint, UniqueVbo> vbos_;
  std::unordered_map<GLuint, UniqueMeshBuffer> mesh_dict_;
  std::unordered_map<GLuint, CompleteFramebuffer> framebuffers_;

 public:
  GLFWwindow *window_;
  Camera camera_;
  int width_;
  int height_;
  Display(int width, int height, const char *title, bool offscreen = false);
  virtual ~Display();

  static void error_callback(int error, const char *description);
  static void mouse_button_callback(GLFWwindow *window, int button, int action,
                                    int mods);
  static void key_callback(GLFWwindow *window, int key, int scancode,
                           int action, int mods);
  static void move_callback(GLFWwindow *window, double xpos, double ypos);
  static void scroll_callback(GLFWwindow *window, double xpos, double ypos);
  static void resize_callback(GLFWwindow *window, int width, int height);

  GLuint create_framebuffer();
  void run();
  void draw();
  GLuint create_colormap(std::array<GLfloat, 3> const *colormap_data,
                         GLsizei palette_size);
  GLuint create_monochrome_texture(unsigned char const *texture_data,
                                   GLsizei width, GLsizei height);
  template <typename M>
  GLuint create_dynamic_array_buffer(U num_elements, void const *src) {
    GLuint vbo =
        GraphicalAllocator::allocate_dynamic_array_buffer<M>(num_elements, src);
    vbos_.emplace(std::piecewise_construct, std::forward_as_tuple(vbo),
                  std::forward_as_tuple(vbo));
    return vbo;
  }
  MeshBuffer create_mesh_buffer(Mesh const &mesh);
  void remove_mesh_buffer(MeshBuffer const &mesh_buffer);
  void add_shading_program(ShadingProgram *program);
  CompleteFramebuffer &get_framebuffer(GLuint fbo);
  CompleteFramebuffer const &get_framebuffer(GLuint fbo) const;
  void update_trackball_camera();
};
}  // namespace alluvion

#endif /* ALLUVION_DISPLAY_HPP */

/*
    LICENSE BEGIN

    trackball - A 3D view interactor for C++ programs.
    Copyright (C) 2016  Remik Ziemlinski

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    LICENSE END
*/
