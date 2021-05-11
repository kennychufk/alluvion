const char* kParticleFragmentShaderStr = R"CODE(
#version 330 core

in vec3 particle_center_worldspace;

struct PointLight {
  vec3 position;

  float constant;
  float linear;
  float quadratic;

  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

struct DirectionalLight {
  vec3 direction;

  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

struct Material {
  vec3 diffuse;
  vec3 specular;
  float shininess;
};

#define NUM_POINT_LIGHTS 2

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3 CalcDirLight(DirectionalLight light, vec3 normal, vec3 viewDir);

uniform float particle_radius;
uniform mat4 V;
uniform mat4 M;
uniform vec3 camera_worldspace;
uniform Material material;
uniform DirectionalLight directional_light;
uniform PointLight point_lights[NUM_POINT_LIGHTS];

out vec4 color;

void main() {
  vec3 normal_modelspace;
  normal_modelspace.xy = gl_PointCoord * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
  float normal_xy_length2 = dot(normal_modelspace.xy, normal_modelspace.xy);
  if (normal_xy_length2 > 1.0) discard;
  normal_modelspace.z = sqrt(1.0 - normal_xy_length2);

  vec3 position_modelspace = normal_modelspace * particle_radius;
  vec3 position_worldspace = (M * vec4(position_modelspace + particle_center_worldspace, 1.0)).xyz;

  vec3 n = normalize(normal_modelspace); // normal_modelspace and normal_cameraspace is almost the same for sphere?
  // Eye vector (towards the camera)
  vec3 E = normalize(camera_worldspace - position_worldspace);

  vec3 accumulate_color = CalcDirLight(directional_light, n, E);
  for (int i = 0; i < NUM_POINT_LIGHTS ; i++)
    accumulate_color += CalcPointLight(point_lights[i], n, position_worldspace, E);

  color = vec4(accumulate_color, 1.0);


}

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
  vec3 lightDir = normalize(light.position - fragPos);
  // diffuse shading
  float diff = max(dot(normal, lightDir), 0.0);
  // specular shading
  vec3 reflectDir = reflect(-lightDir, normal);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
  // attenuation
  float distance = length(light.position - fragPos);
  float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
  // combine results
  vec3 ambient = light.ambient * material.diffuse;
  vec3 diffuse = light.diffuse * diff * material.diffuse;
  vec3 specular = light.specular * spec * material.specular;
  ambient *= attenuation;
  diffuse *= attenuation;
  specular *= attenuation;
  return (ambient + diffuse + specular);
}

vec3 CalcDirLight(DirectionalLight light, vec3 normal, vec3 viewDir)
{
  vec3 lightDir = normalize(-light.direction);
  // diffuse shading
  float diff = max(dot(normal, lightDir), 0.0);
  // specular shading
  vec3 reflectDir = reflect(-lightDir, normal);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
  // combine results
  vec3 ambient = light.ambient * material.diffuse;
  vec3 diffuse = light.diffuse * diff * material.diffuse;
  vec3 specular = light.specular * spec * material.specular;
  return (ambient + diffuse + specular);
}
)CODE";
