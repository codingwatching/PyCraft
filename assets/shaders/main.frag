#version 330 core

in vec2 v_uv;
flat in float v_tex_index;
in float v_fog_distance;

out vec4 out_color;

uniform sampler2DArray textures;
uniform vec3 fog_color = vec3(0.5, 0.69, 1.0);

void main() {
    vec4 tex_color = texture(textures, vec3(v_uv, v_tex_index));
    float fog_factor = exp(-v_fog_distance * 0.042);
    out_color = mix(vec4(fog_color, tex_color.a), tex_color, fog_factor);
}
