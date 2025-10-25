# version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in float tex_id;

out vec3 v_pos;
flat out float v_tex_index;
out vec2 v_uv;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 camera;

// procedural cube vertices (6 faces, 2 triangles per face)
vec3 cubeVertex(int idx) {
    vec3 verts[36] = vec3[](
        // Back face
        vec3(-0.5,-0.5,-0.5), vec3(0.5,-0.5,-0.5), vec3(0.5,0.5,-0.5),
        vec3(0.5,0.5,-0.5), vec3(-0.5,0.5,-0.5), vec3(-0.5,-0.5,-0.5),

        // Front face
        vec3(-0.5,-0.5,0.5), vec3(0.5,-0.5,0.5), vec3(0.5,0.5,0.5),
        vec3(0.5,0.5,0.5), vec3(-0.5,0.5,0.5), vec3(-0.5,-0.5,0.5),

        // Top face
        vec3(-0.5,0.5,-0.5), vec3(0.5,0.5,-0.5), vec3(0.5,0.5,0.5),
        vec3(0.5,0.5,0.5), vec3(-0.5,0.5,0.5), vec3(-0.5,0.5,-0.5),

        // Bottom face
        vec3(-0.5,-0.5,-0.5), vec3(0.5,-0.5,-0.5), vec3(0.5,-0.5,0.5),
        vec3(0.5,-0.5,0.5), vec3(-0.5,-0.5,0.5), vec3(-0.5,-0.5,-0.5),

        // Left face
        vec3(-0.5,-0.5,-0.5), vec3(-0.5,0.5,-0.5), vec3(-0.5,0.5,0.5),
        vec3(-0.5,0.5,0.5), vec3(-0.5,-0.5,0.5), vec3(-0.5,-0.5,-0.5),

        // Right face
        vec3(0.5,-0.5,-0.5), vec3(0.5,0.5,-0.5), vec3(0.5,0.5,0.5),
        vec3(0.5,0.5,0.5), vec3(0.5,-0.5,0.5), vec3(0.5,-0.5,-0.5)
    );
    return verts[idx];
}

// precomputed UVs for each vertex
vec2 cubeUV(int idx) {
    vec2 uvs[36] = vec2[](
        // Back face
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // Front face
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // Top face
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // Bottom face
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // Left face
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0),

        // Right face
        vec2(0,0), vec2(1,0), vec2(1,1),
        vec2(1,1), vec2(0,1), vec2(0,0)
    );
    return uvs[idx];
}

void main() {
    int vert_id = gl_VertexID % 36;
    vec3 local_pos = cubeVertex(vert_id);
    vec3 world_pos = position + local_pos;
    gl_Position = projection * view * vec4(world_pos,1.0);
    v_pos = local_pos;
    v_tex_index = tex_id;
    v_uv = cubeUV(vert_id);
}
