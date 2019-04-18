#version 330

uniform samplerCube cubeMap;

in vec3 texcoord;

void main(void)
{
  gl_FragColor = texture(cubeMap, texcoord);
}
