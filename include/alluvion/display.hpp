#ifndef ALLUVION_DISPLAY_HPP
#define ALLUVION_DISPLAY_HPP

#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <memory>
#include <vector>

#include "alluvion/shading_program.hpp"
#include "alluvion/trackball.hpp"
namespace alluvion {
class Display {
 private:
  GLFWwindow *window_;
  Trackball trackball_;
  std::vector<std::unique_ptr<ShadingProgram>> programs_;
  GLuint vao_;

 public:
  Camera camera_;
  int width_;
  int height_;
  Display(int width, int height, const char *title);
  virtual ~Display();

  static void error_callback(int error, const char *description);
  static void mouse_button_callback(GLFWwindow *window, int button, int action,
                                    int mods);
  static void key_callback(GLFWwindow *window, int key, int scancode,
                           int action, int mods);
  static void move_callback(GLFWwindow *window, double xpos, double ypos);
  static void scroll_callback(GLFWwindow *window, double xpos, double ypos);
  static void resize_callback(GLFWwindow *window, int width, int height);

  void run();
  void add_shading_program(ShadingProgram *program);
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
