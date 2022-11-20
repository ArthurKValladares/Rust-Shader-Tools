
#version 460

struct Vertex
{
	vec4 pos;
    vec4 color;    
    vec2 uv;
    vec2 pad;
};

layout(set = 0, binding = 1) readonly buffer Vertices
{
	Vertex vertices[];
};

void main() {
    Vertex v = vertices[gl_VertexIndex];

    gl_Position =  v.pos;
}