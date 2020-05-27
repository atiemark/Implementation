#version 330 core

out vec4 color;
in vec4 pos;

void main(){

	color = vec4(-pos.z*4, pos.z*6, 1 + pos.z*6, 1.0f);
}