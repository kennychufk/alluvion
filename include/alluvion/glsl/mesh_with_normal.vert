const std::string kMeshWithNormalVertexShaderStr = R"CODE(
#version 330 core
layout(location = 0) in vec3 vertex_position_modelspace;
layout(location = 1) in vec3 vertex_normal_modelspace;

uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;

out vec3 position_worldspace;
out vec3 normal_cameraspace;

void main() {
  // Output position of the vertex, in clip space : MVP * position
  gl_Position =  MVP * vec4(vertex_position_modelspace,1);

  // Position of the vertex, in worldspace : M * position
  position_worldspace = (M * vec4(vertex_position_modelspace,1)).xyz;

  // Vector that goes from the vertex to the camera, in camera space.
  // In camera space, the camera is at the origin (0,0,0).
  vec3 vertexPosition_cameraspace = ( V * M * vec4(vertex_position_modelspace,1)).xyz;

  // Normal of the the vertex, in camera space
  normal_cameraspace = ( V * M * vec4(vertex_normal_modelspace,0)).xyz; // Only correct if ModelMatrix does not scale the model ! Use its inverse transpose if not.
}

)CODE";
