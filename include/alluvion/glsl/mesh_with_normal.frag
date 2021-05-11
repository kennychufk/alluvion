const char* kMeshWithNormalFragmentShaderStr = R"CODE(
#version 330 core
in vec3 position_worldspace;
in vec3 normal_cameraspace;

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

uniform vec3 camera_worldspace;
uniform Material material;
uniform DirectionalLight directional_light;
uniform PointLight point_lights[NUM_POINT_LIGHTS];

out vec4 color;

void main() {
  // Normal of the computed fragment, in camera space
  vec3 n = normalize( normal_cameraspace );
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
