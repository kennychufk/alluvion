const char* kParticleVertexShaderStr = R"CODE(
#version 330 core
layout(location = 0) in vec3 particle_position_worldspace;

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;
uniform vec2 screen_dimension;
uniform float particle_radius;

out vec3 particle_center_worldspace;

void main() {
  vec4 position_cameraspace = V * M * vec4(particle_position_worldspace, 1.0);
  particle_center_worldspace = particle_position_worldspace;

  gl_Position = P * position_cameraspace;
  gl_PointSize = particle_radius * P[0][0] * screen_dimension.x / -position_cameraspace.z;
}
)CODE";
