#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in float orientation;
layout(location = 2) in float tex_id;
layout(location = 3) in float width;
layout(location = 4) in float height;

out vec3 v_pos;
flat out float v_tex_index;
out vec2 v_uv;
out float v_fog_distance;
flat out float v_orientation;

uniform mat4 view;
uniform mat4 projection;

vec3 cubeVertex(int idx) {
    vec3 verts[36] = vec3[](
        // FRONT (+Z)
        vec3(0,0,1), vec3(1,0,1), vec3(1,1,1),
        vec3(1,1,1), vec3(0,1,1), vec3(0,0,1),

        // BACK (-Z)
        vec3(1,0,0), vec3(0,0,0), vec3(0,1,0),
        vec3(0,1,0), vec3(1,1,0), vec3(1,0,0),

        // LEFT (-X)
        vec3(0,0,0), vec3(0,0,1), vec3(0,1,1),
        vec3(0,1,1), vec3(0,1,0), vec3(0,0,0),

        // RIGHT (+X)
        vec3(1,0,1), vec3(1,0,0), vec3(1,1,0),
        vec3(1,1,0), vec3(1,1,1), vec3(1,0,1),

        // TOP (+Y)
        vec3(0,1,1), vec3(1,1,1), vec3(1,1,0),
        vec3(1,1,0), vec3(0,1,0), vec3(0,1,1),

        // BOTTOM (-Y)
        vec3(0,0,0), vec3(1,0,0), vec3(1,0,1),
        vec3(1,0,1), vec3(0,0,1), vec3(0,0,0)
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

    vec3 v = cubeVertex(idx);
    vec3 local_pos;

    if (face == 0) {          // FRONT (+Z)
        local_pos = vec3(v.x * width, v.y * height, 1.0);
    }
    else if (face == 1) {     // BACK (-Z)
        local_pos = vec3((1.0 - v.x) * width, v.y * height, 0.0);
    }
    else if (face == 2) {     // LEFT (-X)
        local_pos = vec3(0.0, v.y * height, v.z * width);
    }
    else if (face == 3) {     // RIGHT (+X)
        local_pos = vec3(1.0, v.y * height, (1.0 - v.z) * width);
    }
    else if (face == 4) {     // TOP (+Y)
        local_pos = vec3(v.x * width, 1.0, v.z * height);
    }
    else {                   // BOTTOM (-Y)
        local_pos = vec3(v.x * width, 0.0, (1.0 - v.z) * height);
    }

    vec3 world_pos = position + local_pos;

    gl_Position = projection * view * vec4(world_pos, 1.0);

    v_pos = local_pos;
    v_tex_index = tex_id;
    vec2 scaled_uv = cubeUV(idx);
    scaled_uv.x *= width;
    scaled_uv.y *= height;
    v_uv = scaled_uv;

    // fog distance (camera space z)
    v_fog_distance = length((view * vec4(world_pos, 1.0)).xyz);
    v_orientation = orientation;
}

