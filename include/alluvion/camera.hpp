#ifndef RSMZ_CAMERA_H
#define RSMZ_CAMERA_H

#include <glm/glm.hpp>

namespace alluvion {

class Camera {
 public:
  Camera();
  ~Camera();

  const glm::mat4& getViewMatrix();
  const glm::mat4& getProjectionMatrix();
  const float* getViewMatrixFlat();
  const glm::vec3& getCenter();
  const glm::vec3& getEye();
  const glm::vec3& getUp();
  void reset();
  void setCenter(float x, float y, float z);
  void setCenter(const glm::vec3& c);
  void setEye(float x, float y, float z);
  void setEye(const glm::vec3& e);
  void setUp(float x, float y, float z);
  void setUp(const glm::vec3& u);
  void setRenderSize(float width, float height);
  void setFocalLength(float focal_length);
  void setClipPlanes(float near, float far);
  void setSensorWidth(float sensor_width);
  void update();

 private:
  glm::vec3 mCenter;
  glm::vec3 mEye;
  glm::vec3 mUp;

  float width_;
  float height_;
  float focal_length_;
  float sensor_width_;
  float clip_near_;
  float clip_far_;

  glm::mat4 view_matrix_;
  glm::mat4 projection_matrix_;
};

}  // namespace alluvion

#endif  // ALLUVION_CAMERA_HPP

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
