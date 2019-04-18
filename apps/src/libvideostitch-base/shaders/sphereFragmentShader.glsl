#ifdef GL_ES
// Set default precision to medium
precision mediump int;
precision mediump float;
#endif

uniform sampler2D sampler;
uniform mediump vec4 color;

uniform int preferTexture;

varying vec2 v_texcoord;

//! [0]
void main()
{
  // Set fragment color from texture
  gl_FragColor = texture2D(sampler, v_texcoord);
}
