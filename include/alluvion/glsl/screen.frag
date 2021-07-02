const std::string kScreenFragmentShaderStr = R"CODE(
#version 330 core
out vec4 color;

in vec2 texcoord;

uniform sampler2D screen_tex;

void main()
{
    color = vec4(texture(screen_tex, texcoord).rgb, 1.0);
}
)CODE";
