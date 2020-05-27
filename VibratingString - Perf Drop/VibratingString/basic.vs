#version 330 core

in vec3 position;
out vec4 pos;

uniform mat4 u_model;


void main(){
	pos = vec4(position, 1.0);
	gl_Position = u_model * vec4(position, 1.0f);
}