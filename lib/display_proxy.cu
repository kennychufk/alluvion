
#include "alluvion/colormaps.hpp"
#include "alluvion/display_proxy.hpp"

namespace alluvion {
DisplayProxy::DisplayProxy(Display *display) : display_(display) {}

GLuint DisplayProxy::create_colormap_viridis() {
  return display_->create_colormap(kViridisData.data(), kViridisData.size());
}

void DisplayProxy::run() { return display_->run(); }
void DisplayProxy::set_camera(float3 camera_pos, float3 center) {
  display_->camera_.setEye(camera_pos.x, camera_pos.y, camera_pos.z);
  display_->camera_.setCenter(center.x, center.y, center.z);
  display_->camera_.update();
  display_->update_trackball_camera();
}
}  // namespace alluvion
