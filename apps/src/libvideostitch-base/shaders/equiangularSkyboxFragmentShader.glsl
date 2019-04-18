#version 330

uniform samplerCube cubeMap;

const float PI = 3.1415926535897932384626433832795;

in vec3 texcoord;

void main(void)
{
  gl_FragColor = texture(cubeMap, 4. / PI * atan(texcoord));
}
