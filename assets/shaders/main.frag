#version 330 core

in vec2 v_uv;
flat in float v_tex_index;

out vec4 out_color;
uniform sampler2DArray textures;

void main() {
    out_color = texture(textures, vec3(v_uv, v_tex_index));
}
