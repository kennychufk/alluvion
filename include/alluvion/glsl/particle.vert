const std::string kParticleVertexShaderStr = R"CODE(
#version 330 core
#ARB_GPU_SHADER_FP64
#ARB_VERTEX_ATTRIB_64BIT
layout(location = 0) in TF3 particle_position_worldspace;
layout(location = 1) in TF particle_normalized_attr;

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;
uniform vec2 screen_dimension;
uniform float particle_radius;

out vec3 particle_center_worldspace;
out float particle_normalized_attr_pass;

void main() {
  vec4 position_cameraspace = V * M * vec4(TF3_TO_FLOAT3(particle_position_worldspace), 1.0);
  particle_center_worldspace = TF3_TO_FLOAT3(particle_position_worldspace);

  gl_Position = P * position_cameraspace;
  gl_PointSize = particle_radius * P[0][0] * screen_dimension.x / -position_cameraspace.z;

  particle_normalized_attr_pass = TF_TO_FLOAT(particle_normalized_attr);
}
)CODE";
