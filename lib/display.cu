#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>

#include "alluvion/display.hpp"

namespace alluvion {
Display::Display(int width, int height, const char *title) : window_(nullptr) {
  glfwSetErrorCallback(&Display::error_callback);

  if (!glfwInit()) exit(EXIT_FAILURE);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window_ = glfwCreateWindow(width, height, title, NULL, NULL);
  if (!window_) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwMakeContextCurrent(window_);
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
  glfwSwapInterval(1);

  using namespace std::placeholders;

  glfwSetWindowUserPointer(window_, this);
  glfwSetCursorPosCallback(window_, &Display::move_callback);
  glfwSetKeyCallback(window_, &Display::key_callback);
  glfwSetMouseButtonCallback(window_, &Display::mouse_button_callback);
  glfwSetScrollCallback(window_, &Display::scroll_callback);
  glfwSetWindowSizeCallback(window_, &Display::resize_callback);

  update_trackball_camera();
  resize_callback(window_, width, height);

  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);  // necessary for initialization
  glBindVertexArray(0);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_POINT_SPRITE_ARB);
  glDepthMask(GL_TRUE);
  glDepthRange(0.0f, 1.0f);
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glClearColor(0.2f, 0.2f, 0.2f, 0.0f);
  glClearDepth(1.0f);
}
Display::~Display() {
  for (std::unique_ptr<ShadingProgram> &program : programs_) {
    program.reset();
  }
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void Display::error_callback(int error, const char *description) {
  std::cerr << description << std::endl;
}

void Display::mouse_button_callback(GLFWwindow *window, int button, int action,
                                    int mods) {
  Display *display = static_cast<Display *>(glfwGetWindowUserPointer(window));
  switch (action) {
    case GLFW_PRESS: {
      switch (button) {
        case GLFW_MOUSE_BUTTON_LEFT:
          display->trackball_.setLeftClicked(true);
          break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
          display->trackball_.setMiddleClicked(true);
          break;
        case GLFW_MOUSE_BUTTON_RIGHT:
          display->trackball_.setRightClicked(true);
          break;
      }

      double xpos, ypos;
      glfwGetCursorPos(window, &xpos, &ypos);
      display->trackball_.setClickPoint(xpos, ypos);
      break;
    }
    case GLFW_RELEASE: {
      switch (button) {
        case GLFW_MOUSE_BUTTON_LEFT:
          display->trackball_.setLeftClicked(false);
          break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
          display->trackball_.setMiddleClicked(false);
          break;
        case GLFW_MOUSE_BUTTON_RIGHT:
          display->trackball_.setRightClicked(false);
          break;
      }
      break;
    }
    default:
      break;
  }
}

void Display::key_callback(GLFWwindow *window, int key, int scancode,
                           int action, int mods) {
  Display *display = static_cast<Display *>(glfwGetWindowUserPointer(window));
  float length;

  switch (action) {
    case GLFW_PRESS:
      switch (key) {
        case GLFW_KEY_ESCAPE:
          // Exit app on ESC key.
          glfwSetWindowShouldClose(window, GL_TRUE);
          break;
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
          display->trackball_.setSpeed(5.f);
          break;
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
          display->trackball_.setSpeed(.1f);
          break;
        case GLFW_KEY_C:
          std::cout << "(" << display->camera_.getEye().x << ","
                    << display->camera_.getEye().y << ","
                    << display->camera_.getEye().z << ") "
                    << "(" << display->camera_.getCenter().x << ","
                    << display->camera_.getCenter().y << ","
                    << display->camera_.getCenter().z << ") "
                    << "(" << display->camera_.getUp().x << ","
                    << display->camera_.getUp().y << ","
                    << display->camera_.getUp().z << ")\n";
          break;
        case GLFW_KEY_R:
          // Reset the view.
          display->camera_.reset();
          display->update_trackball_camera();
          break;
        case GLFW_KEY_T:
          // Toogle motion type.
          if (display->trackball_.getMotionRightClick() ==
              Trackball::FIRSTPERSON) {
            display->trackball_.setMotionRightClick(Trackball::PAN);
          } else {
            display->trackball_.setMotionRightClick(Trackball::FIRSTPERSON);
          }
          break;
        case GLFW_KEY_X:
          // Snap view to axis.
          length = glm::length(display->camera_.getEye() -
                               display->camera_.getCenter());
          display->camera_.setEye(length, 0, 0);
          display->camera_.setUp(0, 1, 0);
          display->camera_.update();
          display->update_trackball_camera();
          break;
        case GLFW_KEY_Y:
          length = glm::length(display->camera_.getEye() -
                               display->camera_.getCenter());
          display->camera_.setEye(0, length, 0);
          display->camera_.setUp(1, 0, 0);
          display->camera_.update();
          display->update_trackball_camera();
          break;
        case GLFW_KEY_Z:
          length = glm::length(display->camera_.getEye() -
                               display->camera_.getCenter());
          display->camera_.setEye(0, 0, length);
          display->camera_.setUp(1, 0, 0);
          display->camera_.update();
          display->update_trackball_camera();
          break;
        default:
          break;
      }
      break;
    case GLFW_RELEASE:
      switch (key) {
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
        case GLFW_KEY_LEFT_SHIFT:
        case GLFW_KEY_RIGHT_SHIFT:
          display->trackball_.setSpeed(1.f);
          break;
      }
      break;
    default:
      break;
  }
}

void Display::move_callback(GLFWwindow *window, double xpos, double ypos) {
  Display *display = static_cast<Display *>(glfwGetWindowUserPointer(window));
  display->trackball_.setClickPoint(xpos, ypos);
}

void Display::scroll_callback(GLFWwindow *window, double xpos, double ypos) {
  Display *display = static_cast<Display *>(glfwGetWindowUserPointer(window));
  display->trackball_.setScrollDirection(xpos + ypos > 0 ? true : false);
}

void Display::resize_callback(GLFWwindow *window, int width, int height) {
  Display *display = static_cast<Display *>(glfwGetWindowUserPointer(window));
  display->trackball_.setScreenSize(width, height);
  display->camera_.setRenderSize(width, height);
  display->camera_.update();
  display->width_ = width;
  display->height_ = height;
  glViewport(0, 0, width, height);
}

void Display::run() {
  while (!glfwWindowShouldClose(window_)) {
    trackball_.update();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindVertexArray(vao_);

    for (std::unique_ptr<ShadingProgram> &program : programs_) {
      program->update(*this);
    }
    glfwSwapBuffers(window_);
    glfwPollEvents();
  }
}

void Display::add_shading_program(ShadingProgram *program) {
  programs_.emplace_back(program);
}

void Display::update_trackball_camera() { trackball_.setCamera(&camera_); }

}  // namespace alluvion

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
