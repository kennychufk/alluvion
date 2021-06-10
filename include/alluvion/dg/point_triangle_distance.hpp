#ifndef ALLUVION_DG_POINT_TRIANGLE_DISTANCE_HPP
#define ALLUVION_DG_POINT_TRIANGLE_DISTANCE_HPP

#include <Eigen/Core>
#include <array>

#include "alluvion/dg/common.hpp"

namespace alluvion {
namespace dg {

enum class NearestEntity { VN0, VN1, VN2, EN0, EN1, EN2, FN };

template <typename TF>
TF point_triangle_sqdistance(Vector3r<TF> const& point,
                             std::array<Vector3r<TF> const*, 3> const& triangle,
                             Vector3r<TF>* nearest_point = nullptr,
                             NearestEntity* ne = nullptr) {
  Vector3r<TF> diff = *triangle[0] - point;
  Vector3r<TF> edge0 = *triangle[1] - *triangle[0];
  Vector3r<TF> edge1 = *triangle[2] - *triangle[0];
  TF a00 = edge0.dot(edge0);
  TF a01 = edge0.dot(edge1);
  TF a11 = edge1.dot(edge1);
  TF b0 = diff.dot(edge0);
  TF b1 = diff.dot(edge1);
  TF c = diff.dot(diff);
  TF det = std::abs(a00 * a11 - a01 * a01);
  TF s = a01 * b1 - a11 * b0;
  TF t = a01 * b0 - a00 * b1;

  TF d2 = -1.0;

  if (s + t <= det) {
    if (s < 0) {
      if (t < 0)  // region 4
      {
        if (b0 < 0) {
          t = 0;
          if (-b0 >= a00) {  // VN1
            if (ne) *ne = NearestEntity::VN1;
            s = 1;
            d2 = a00 + (2) * b0 + c;
          } else {
            // EN0
            if (ne) *ne = NearestEntity::EN0;
            s = -b0 / a00;
            d2 = b0 * s + c;
          }
        } else {
          s = 0;
          if (b1 >= 0) {  // VN0
            if (ne) *ne = NearestEntity::VN0;
            t = 0;
            d2 = c;
          } else if (-b1 >= a11) {
            // VN2
            if (ne) *ne = NearestEntity::VN2;
            t = 1;
            d2 = a11 + (2) * b1 + c;
          } else {
            // EN2
            if (ne) *ne = NearestEntity::EN2;
            t = -b1 / a11;
            d2 = b1 * t + c;
          }
        }
      } else  // region 3
      {
        s = 0;
        if (b1 >= 0) {  // VN0
          if (ne) *ne = NearestEntity::VN0;
          t = 0;
          d2 = c;
        } else if (-b1 >= a11) {  // VN2
          if (ne) *ne = NearestEntity::VN2;
          t = 1;
          d2 = a11 + (2) * b1 + c;
        } else {  // EN2
          if (ne) *ne = NearestEntity::EN2;
          t = -b1 / a11;
          d2 = b1 * t + c;
        }
      }
    } else if (t < 0)  // region 5
    {
      t = 0;
      if (b0 >= 0) {  // VN0
        if (ne) *ne = NearestEntity::VN0;
        s = 0;
        d2 = c;
      } else if (-b0 >= a00) {  // VN1
        if (ne) *ne = NearestEntity::VN1;
        s = 1;
        d2 = a00 + (2) * b0 + c;
      } else {  // EN0
        if (ne) *ne = NearestEntity::EN0;
        s = -b0 / a00;
        d2 = b0 * s + c;
      }
    } else  // region 0
    {       // FN
      if (ne) *ne = NearestEntity::FN;
      // minimum at interior point
      TF invDet = (1) / det;
      s *= invDet;
      t *= invDet;
      d2 = s * (a00 * s + a01 * t + (2) * b0) +
           t * (a01 * s + a11 * t + (2) * b1) + c;
    }
  } else {
    TF tmp0, tmp1, numer, denom;

    if (s < 0)  // region 2
    {
      tmp0 = a01 + b0;
      tmp1 = a11 + b1;
      if (tmp1 > tmp0) {
        numer = tmp1 - tmp0;
        denom = a00 - (2) * a01 + a11;
        if (numer >= denom) {  // VN1
          if (ne) *ne = NearestEntity::VN1;
          s = 1;
          t = 0;
          d2 = a00 + (2) * b0 + c;
        } else {
          // EN1
          if (ne) *ne = NearestEntity::EN1;
          s = numer / denom;
          t = 1 - s;
          d2 = s * (a00 * s + a01 * t + (2) * b0) +
               t * (a01 * s + a11 * t + (2) * b1) + c;
        }
      } else {
        s = 0;
        if (tmp1 <= 0) {  // VN2
          if (ne) *ne = NearestEntity::VN2;
          t = 1;
          d2 = a11 + (2) * b1 + c;
        } else if (b1 >= 0) {  // VN0
          if (ne) *ne = NearestEntity::VN0;
          t = 0;
          d2 = c;
        } else {
          // EN2
          if (ne) *ne = NearestEntity::EN2;
          t = -b1 / a11;
          d2 = b1 * t + c;
        }
      }
    } else if (t < 0)  // region 6
    {
      tmp0 = a01 + b1;
      tmp1 = a00 + b0;
      if (tmp1 > tmp0) {
        numer = tmp1 - tmp0;
        denom = a00 - (2) * a01 + a11;
        if (numer >= denom) {  // VN2
          if (ne) *ne = NearestEntity::VN2;
          t = 1;
          s = 0;
          d2 = a11 + (2) * b1 + c;
        } else {
          // EN1
          if (ne) *ne = NearestEntity::EN1;
          t = numer / denom;
          s = 1 - t;
          d2 = s * (a00 * s + a01 * t + (2) * b0) +
               t * (a01 * s + a11 * t + (2) * b1) + c;
        }
      } else {
        t = 0;
        if (tmp1 <= 0) {  // VN1
          if (ne) *ne = NearestEntity::VN1;
          s = 1;
          d2 = a00 + (2) * b0 + c;
        } else if (b0 >= 0) {  // VN0
          if (ne) *ne = NearestEntity::VN0;
          s = 0;
          d2 = c;
        } else {
          // EN0
          if (ne) *ne = NearestEntity::EN0;
          s = -b0 / a00;
          d2 = b0 * s + c;
        }
      }
    } else  // region 1
    {
      numer = a11 + b1 - a01 - b0;
      if (numer <= 0) {  // VN2
        if (ne) *ne = NearestEntity::VN2;
        s = 0;
        t = 1;
        d2 = a11 + (2) * b1 + c;
      } else {
        denom = a00 - (2) * a01 + a11;
        if (numer >= denom) {  // VN1
          if (ne) *ne = NearestEntity::VN1;
          s = 1;
          t = 0;
          d2 = a00 + (2) * b0 + c;
        } else {  // EN1
          if (ne) *ne = NearestEntity::EN1;
          s = numer / denom;
          t = 1 - s;
          d2 = s * (a00 * s + a01 * t + (2) * b0) +
               t * (a01 * s + a11 * t + (2) * b1) + c;
        }
      }
    }
  }

  // Account for numerical round-off error.
  if (d2 < 0) {
    d2 = 0;
  }

  if (nearest_point) *nearest_point = *triangle[0] + s * edge0 + t * edge1;

  return d2;

  // result.distance = sqrt(d2);
  // result.triangleClosestPoint = triangle.v[0] + s*edge0 + t*edge1;
  // result.triangleParameter[1] = s;
  // result.triangleParameter[2] = t;
  // result.triangleParameter[0] = 1 - s - t;
  // return result;
}

}  // namespace dg
}  // namespace alluvion
#endif
