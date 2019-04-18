attribute vec2 aPosition;
attribute vec2 aTexCoord;

varying vec2 vTexCoord;

void main(void)
{
	vTexCoord = aTexCoord;	
  	gl_Position = vec4(aPosition.x, aPosition.y, 0.0, 1.0);
}
