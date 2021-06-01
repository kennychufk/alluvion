const char* kScreenVertexShaderStr = R"CODE(
#version 330 core
layout (location = 0) in vec4 clip_xy_tex;
out vec2 texcoord;

void main()
{
    texcoord = clip_xy_tex.zw;
    gl_Position = vec4(clip_xy_tex.x, clip_xy_tex.y, 0.0, 1.0); 
}
)CODE";
