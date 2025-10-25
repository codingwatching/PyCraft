# Random example from chatgpt, just for reference
# CHATGPT HASNT WRITTEN THE ACTUAL CODE IN ANY OTHER FILES!
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from PIL import Image
import glm
import time

# ----------------- Shader Sources -----------------
vertex_src = """
#version 330 core

layout(location = 0) in vec4 instances;

out vec3 v_pos;
flat out float v_tex_index;
out vec2 v_uv;

uniform mat4 projection;
uniform mat4 view;

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
    vec3 world_pos = instances.xyz + local_pos;

    gl_Position = projection * view * vec4(world_pos,1.0);
    v_pos = local_pos;
    v_tex_index = instances.w;
    v_uv = cubeUV(vert_id);
}
"""

fragment_src = """
#version 330 core

in vec2 v_uv;
flat in float v_tex_index;

out vec4 out_color;
uniform sampler2DArray textures;

void main() {
    out_color = texture(textures, vec3(v_uv, v_tex_index));
}
"""

# ----------------- Initialize GLFW -----------------
if not glfw.init():
    raise Exception("glfw init failed")
window = glfw.create_window(800, 600, "Procedural Instanced Cubes", None, None)
glfw.make_context_current(window)
glEnable(GL_DEPTH_TEST)
glfw.swap_interval(0)

# ----------------- Compile Shader -----------------
shader = compileProgram(
    compileShader(vertex_src, GL_VERTEX_SHADER),
    compileShader(fragment_src, GL_FRAGMENT_SHADER)
)

# ----------------- Instance Data -----------------
from random import randint
n = 1024 * 32
x = 42
things = []
tex = []
for i in range(n):
    things.append([
        randint(-x, x),
        randint(-2*x, 0) - 2,
        randint(-x, x),
        randint(0, 2)
    ])
instances = np.array(things, dtype=np.float32)

# VBOs for instance attributes
instance_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
glBufferData(GL_ARRAY_BUFFER, instances.nbytes, instances, GL_STATIC_DRAW)

# VAO setup
vao = glGenVertexArrays(1)
glBindVertexArray(vao)

# instances
glBindBuffer(GL_ARRAY_BUFFER, instance_vbo)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,None)
glVertexAttribDivisor(0,1)

# ----------------- Texture Array -----------------
tex_files = ["cobblestone.jpg", "nothing.jpg"]
tex_id = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D_ARRAY, tex_id)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)

images = [Image.open(f).convert("RGBA") for f in tex_files]
w,h = images[0].size
layer_data = np.stack([np.array(im) for im in images], axis=0)
glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, w, h, len(images), 0, GL_RGBA, GL_UNSIGNED_BYTE, layer_data)

# ----------------- Projection -----------------
projection = glm.perspective(glm.radians(45.0), 800/600, 0.1, 100.0)
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")
glUseProgram(shader)
glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(projection))

# ----------------- Main Loop -----------------
start_time = time.time()
while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # rotate camera around Y-axis
    t = time.time() - start_time
    radius = 8
    cam_pos = glm.vec3(np.sin(t)*radius, 4, np.cos(t)*radius)
    view = glm.lookAt(cam_pos, glm.vec3(0,0,0), glm.vec3(0,1,0))
    glUniformMatrix4fv(view_loc,1,GL_FALSE,glm.value_ptr(view))

    glBindVertexArray(vao)
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex_id)
    glDrawArraysInstanced(GL_TRIANGLES, 0, 36, len(instances))  # 36 verts per cube, instanced

    glfw.swap_buffers(window)

glfw.terminate()

