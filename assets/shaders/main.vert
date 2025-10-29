#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in float orientation;
layout(location = 2) in float tex_id;
layout(location = 3) in int scale_multiplier;

out vec3 v_pos;
flat out float v_tex_index;
out vec2 v_uv;
out float v_fog_distance;

uniform mat4 view;
uniform mat4 projection;

vec3 cubeVertex(int idx) {
    vec3 verts[36] = vec3[](
        // FRONT (+Z)
        vec3(-0.5,-0.5,0.5), vec3(0.5,-0.5,0.5), vec3(0.5, 0.5,0.5),
        vec3(0.5, 0.5,0.5), vec3(-0.5, 0.5,0.5), vec3(-0.5,-0.5,0.5),

        // BACK (-Z)
        vec3(0.5,-0.5,-0.5), vec3(-0.5,-0.5,-0.5), vec3(-0.5, 0.5,-0.5),
        vec3(-0.5, 0.5,-0.5), vec3(0.5, 0.5,-0.5), vec3(0.5,-0.5,-0.5),

        // LEFT (-X)
        vec3(-0.5,-0.5,-0.5), vec3(-0.5,-0.5, 0.5), vec3(-0.5, 0.5, 0.5),
        vec3(-0.5, 0.5, 0.5), vec3(-0.5, 0.5,-0.5), vec3(-0.5,-0.5,-0.5),

        // RIGHT (+X)
        vec3(0.5,-0.5, 0.5), vec3(0.5,-0.5,-0.5), vec3(0.5, 0.5,-0.5),
        vec3(0.5, 0.5,-0.5), vec3(0.5, 0.5, 0.5), vec3(0.5,-0.5, 0.5),

        // TOP (+Y)
        vec3(-0.5,0.5, 0.5), vec3(0.5,0.5, 0.5), vec3(0.5,0.5,-0.5),
        vec3(0.5,0.5,-0.5), vec3(-0.5,0.5,-0.5), vec3(-0.5,0.5, 0.5),

        // BOTTOM (-Y)
        vec3(-0.5,-0.5,-0.5), vec3(0.5,-0.5,-0.5), vec3(0.5,-0.5, 0.5),
        vec3(0.5,-0.5, 0.5), vec3(-0.5,-0.5, 0.5), vec3(-0.5,-0.5,-0.5)
    );
    return verts[idx];
}

vec2 cubeUV(int idx) {
    vec2 uvs[36] = vec2[](
        // FRONT (+Z)
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // BACK (-Z)
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // LEFT (-X)
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // RIGHT (+X)
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // TOP (+Y)
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // BOTTOM (-Y)
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0)
    );
    return uvs[idx];
}

void main() {
    int face = int(orientation);   // which face (0–5)
    int vert_id = gl_VertexID % 6; // vertex within face
    int idx = face * 6 + vert_id;  // final vertex index

    vec3 local_pos = cubeVertex(idx) * scale_multiplier;  // apply scale
    vec3 world_pos = position + local_pos;

    gl_Position = projection * view * vec4(world_pos, 1.0);

    v_pos = local_pos;
    v_tex_index = tex_id;
    v_uv = cubeUV(idx) * scale_multiplier;

    // fog distance (camera space z)
    v_fog_distance = length((view * vec4(world_pos, 1.0)).xyz);
}

