#version 330

in vec4 vertex;

uniform mat4 mvp_matrix;
out vec3 texcoord;

void main(void)
{
  // pass on the texture coordinates
  texcoord = vertex.xyz;
  texcoord.z = -texcoord.z;

  // transform the geometry to screen space
  gl_Position = mvp_matrix * vertex;
}
