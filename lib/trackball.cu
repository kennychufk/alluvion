#include <glm/gtx/norm.hpp>  // length2

#include "alluvion/trackball.hpp"

namespace alluvion {

const glm::vec3 Trackball::X(1.f, 0.f, 0.f);
const glm::vec3 Trackball::Y(0.f, 1.f, 0.f);
const glm::vec3 Trackball::Z(0.f, 0.f, 1.f);

Trackball::Trackball()
    : mCameraMotionLeftClick(ARC),
      mCameraMotionMiddleClick(ROLL),
      mCameraMotionRightClick(FIRSTPERSON),
      mCameraMotionScroll(ZOOM),
      mHeight(1),
      mIsDragging(false),
      mIsLeftClick(false),
      mIsMiddleClick(false),
      mIsRightClick(false),
      mIsScrolling(false),
      mPanScale(.005f),
      mRollScale(.005f),
      mRollSum(0.f),
      mRotation(1.f, 0, 0, 0),
      mRotationSum(1.f, 0, 0, 0),
      mSpeed(1.f),
      mWidth(1),
      mZoomScale(.1f),
      mZoomSum(0.f) {}

Trackball::~Trackball() {}

char Trackball::clickQuadrant(float x, float y) {
  float halfw = .5 * mWidth;
  float halfh = .5 * mHeight;

  if (x > halfw) {
    // Opengl image coordinates origin is upperleft.
    if (y < halfh) {
      return 1;
    } else {
      return 4;
    }
  } else {
    if (y < halfh) {
      return 2;
    } else {
      return 3;
    }
  }
}

void Trackball::computeCameraEye(glm::vec3& eye) {
  glm::vec3 orientation = mRotationSum * Z;

  if (mZoomSum) {
    mTranslateLength += mZoomScale * mZoomSum;
    mZoomSum = 0;  // Freeze zooming after applying.
  }

  eye = mTranslateLength * orientation + mCamera->getCenter();
}

void Trackball::computeCameraUp(glm::vec3& up) {
  up = glm::normalize(mRotationSum * Y);
}

void Trackball::computePan(glm::vec3& pan) {
  glm::vec2 click = mClickPoint - mPrevClickPoint;
  glm::vec3 look = mCamera->getEye() - mCamera->getCenter();
  float length = glm::length(look);
  glm::vec3 right = glm::normalize(mRotationSum * X);

  pan = (mCamera->getUp() * -click.y + right * click.x) * mPanScale * mSpeed *
        length;
}

void Trackball::computePointOnSphere(const glm::vec2& point,
                                     glm::vec3& result) {
  // https://www.opengl.org/wiki/Object_Mouse_Trackball
  float x = (2.f * point.x - mWidth) / mWidth;
  float y = (mHeight - 2.f * point.y) / mHeight;

  float length2 = x * x + y * y;

  if (length2 <= .5) {
    result.z = sqrt(1.0 - length2);
  } else {
    result.z = 0.5 / sqrt(length2);
  }

  float norm = 1.0 / sqrt(length2 + result.z * result.z);

  result.x = x * norm;
  result.y = y * norm;
  result.z *= norm;
}

void Trackball::computeRotationBetweenVectors(const glm::vec3& u,
                                              const glm::vec3& v,
                                              glm::quat& result) {
  float cosTheta = glm::dot(u, v);
  glm::vec3 rotationAxis(0.0f, 0.0f, 0.0f);
  static const float EPSILON = 1.0e-5f;

  if (cosTheta < -1.0f + EPSILON) {
    // Parallel and opposite directions.
    rotationAxis = glm::cross(glm::vec3(0.f, 0.f, 1.f), u);

    if (glm::length2(rotationAxis) < 0.01) {
      // Still parallel, retry.
      rotationAxis = glm::cross(glm::vec3(1.f, 0.f, 0.f), u);
    }

    rotationAxis = glm::normalize(rotationAxis);
    result = glm::angleAxis(180.0f, rotationAxis);
  } else if (cosTheta > 1.0f - EPSILON) {
    // Parallel and same direction.
    result = glm::quat(1, 0, 0, 0);
    return;
  } else {
    float theta = acos(cosTheta);
    rotationAxis = glm::cross(u, v);

    rotationAxis = glm::normalize(rotationAxis);
    result = glm::angleAxis(theta * mSpeed, rotationAxis);
  }
}

void Trackball::drag() {
  if (mPrevClickPoint == mClickPoint) {
    // Not moving during drag state, so skip unnecessary processing.
    return;
  }

  computePointOnSphere(mClickPoint, mStopVector);
  computeRotationBetweenVectors(mStartVector, mStopVector, mRotation);
  // Reverse so scene moves with cursor and not away due to camera model.
  mRotation = glm::inverse(mRotation);

  drag(mIsLeftClick, mCameraMotionLeftClick);
  drag(mIsMiddleClick, mCameraMotionMiddleClick);
  drag(mIsRightClick, mCameraMotionRightClick);

  // After applying drag, reset relative start state.
  mPrevClickPoint = mClickPoint;
  mStartVector = mStopVector;
}

void Trackball::drag(bool isClicked, CameraMotionType motion) {
  if (!isClicked) {
    return;
  }

  switch (motion) {
    case ARC:
      dragArc();
      break;
    case FIRSTPERSON:
      dragFirstPerson();
      break;
    case PAN:
      dragPan();
      break;
    case ROLL:
      rollCamera();
      break;
    case ZOOM:
      dragZoom();
      break;
    default:
      break;
  }
}

void Trackball::dragArc() {
  mRotationSum *= mRotation;  // Accumulate quaternions.

  updateCameraEyeUp(true, true);
}

void Trackball::dragFirstPerson() {
  glm::vec3 pan(0.0f, 0.0f, 0.0f);
  computePan(pan);
  mCamera->setCenter(pan + mCamera->getCenter());
  mCamera->update();
  freezeTransform();
}

void Trackball::dragPan() {
  glm::vec3 pan(0.0f, 0.0f, 0.0f);
  computePan(pan);
  mCamera->setCenter(pan + mCamera->getCenter());
  mCamera->setEye(pan + mCamera->getEye());
  mCamera->update();
  freezeTransform();
}

void Trackball::dragZoom() {
  glm::vec2 dir = mClickPoint - mPrevClickPoint;
  float ax = fabs(dir.x);
  float ay = fabs(dir.y);

  if (ay >= ax) {
    setScrollDirection(dir.y <= 0);
  } else {
    setScrollDirection(dir.x <= 0);
  }

  updateCameraEyeUp(true, false);
}

void Trackball::freezeTransform() {
  if (mCamera) {
    // Opengl is ZYX order.
    // Flip orientation to rotate scene with sticky cursor.
    mRotationSum = glm::inverse(glm::quat(mCamera->getViewMatrix()));
    mTranslateLength = glm::length(mCamera->getEye() - mCamera->getCenter());
  }
}

Camera* Trackball::getCamera() { return mCamera; }

Trackball::CameraMotionType Trackball::getMotionLeftClick() {
  return mCameraMotionLeftClick;
}

Trackball::CameraMotionType Trackball::getMotionMiddleClick() {
  return mCameraMotionMiddleClick;
}

Trackball::CameraMotionType Trackball::getMotionRightClick() {
  return mCameraMotionRightClick;
}

Trackball::CameraMotionType Trackball::getMotionScroll() {
  return mCameraMotionScroll;
}

void Trackball::rollCamera() {
  glm::vec2 delta = mClickPoint - mPrevClickPoint;
  char quad = clickQuadrant(mClickPoint.x, mClickPoint.y);
  switch (quad) {
    case 1:
      delta.y = -delta.y;
      delta.x = -delta.x;
      break;
    case 2:
      delta.x = -delta.x;
      break;
    case 3:
      break;
    case 4:
      delta.y = -delta.y;
    default:
      break;
  }

  glm::vec3 axis = glm::normalize(mCamera->getCenter() - mCamera->getEye());
  float angle = mRollScale * mSpeed * (delta.x + delta.y + mRollSum);
  glm::quat rot = glm::angleAxis(angle, axis);
  mCamera->setUp(rot * mCamera->getUp());
  mCamera->update();
  freezeTransform();
  mRollSum = 0;
}

void Trackball::scroll() {
  switch (mCameraMotionScroll) {
    case ROLL:
      rollCamera();
      break;
    case ZOOM:
      updateCameraEyeUp(true, false);
      break;
    default:
      break;
  }
}

void Trackball::setCamera(Camera* c) {
  mCamera = c;
  freezeTransform();
}

void Trackball::setClickPoint(double x, double y) {
  mPrevClickPoint = mClickPoint;
  mClickPoint.x = x;
  mClickPoint.y = y;
}

void Trackball::setLeftClicked(bool value) { mIsLeftClick = value; }

void Trackball::setMiddleClicked(bool value) { mIsMiddleClick = value; }

void Trackball::setMotionLeftClick(CameraMotionType motion) {
  mCameraMotionLeftClick = motion;
}

void Trackball::setMotionMiddleClick(CameraMotionType motion) {
  mCameraMotionMiddleClick = motion;
}

void Trackball::setMotionRightClick(CameraMotionType motion) {
  mCameraMotionRightClick = motion;
}

void Trackball::setMotionScroll(CameraMotionType motion) {
  mCameraMotionScroll = motion;
}

void Trackball::setRightClicked(bool value) { mIsRightClick = value; }

void Trackball::setScreenSize(float width, float height) {
  if (width > 1 && height > 1) {
    mWidth = width;
    mHeight = height;
  }
}

void Trackball::setScrollDirection(bool up) {
  mIsScrolling = true;
  float inc = mSpeed * (up ? -1.f : 1.f);
  mZoomSum += inc;
  mRollSum += inc;
}

void Trackball::setSpeed(float s) { mSpeed = s; }

void Trackball::update() {
  const bool isClick = mIsLeftClick || mIsMiddleClick || mIsRightClick;

  if (!mIsDragging) {
    if (isClick) {
      mIsDragging = true;
      computePointOnSphere(mClickPoint, mStartVector);
    } else if (mIsScrolling) {
      scroll();
      mIsScrolling = false;
    }
  } else {
    if (isClick) {
      drag();
    } else {
      mIsDragging = false;
    }
  }
}

void Trackball::updateCameraEyeUp(bool eye, bool up) {
  if (eye) {
    glm::vec3 eye(0.0f, 0.0f, 0.0f);
    computeCameraEye(eye);
    mCamera->setEye(eye);
  }
  if (up) {
    glm::vec3 up(0.0f, 0.0f, 0.0f);
    computeCameraUp(up);
    mCamera->setUp(up);
  }
  mCamera->update();
}

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
