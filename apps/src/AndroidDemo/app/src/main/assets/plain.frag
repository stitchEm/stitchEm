precision mediump float;

uniform sampler2D uSourceTex;

varying vec2 vTexCoord;

void main(void)
{
    gl_FragColor = texture2D(uSourceTex, vTexCoord);
}
