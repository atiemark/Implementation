#include <iostream>
#define GLEW_STATIC
#include <GL/glew.h>
#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <math.h>
#include <chrono>
#include <ctime>
#include <thread> 
#include "main.h"
#include "shader.h"
#include "glm/glm.hpp"
#include "glm/ext/matrix_transform.hpp"

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"



#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "glew32s.lib")
#pragma comment(lib, "opengl32.lib")


using namespace std;
using namespace Eigen;

struct MVertex {
	float x;
	float y;
	float z;
};

float cosTable[2048];

int windowSizeX = 1000;
int windowSizeY = 1000;
SDL_Window* window;
SDL_GLContext glContext;
SDL_AudioDeviceID audio_device;

int numMasses = 10;
int sliderMasses = numMasses;
float massWeight = 1;
float springStiffness = 30;
float dampingCoeff = 1;

vector<Vector2f> massPositionsString;
MatrixXf modeShapesString;
VectorXf modeFrequenciesString;
VectorXf modeDampingTermsString;
VectorXf modeGainsString;

float** modeShapesStringArray;
float* modeFrequenciesStringArray;
float* modeDampingTermsStringArray;
float* modeGainsStringArray;

vector<float> massPositionsMembrane;
MatrixXf modeShapesMembrane;
VectorXf modeFrequenciesMembrane;
VectorXf modeDampingTermsMembrane;
VectorXf modeGainsMembrane;


float lineColor[3] = { 1.0f, 1.0f, 1.0f };
float pointColor[3] = { 1.0f, 1.0f, 1.0f };
int lineWidth = 1;
int pointSize = 6;
float mouseDistThreshGlSpace = 1.f / (float)(numMasses + 1);

bool run = false;
bool displayModeShape = false;
int displayModeShapeNum = 1;

chrono::time_point<chrono::high_resolution_clock> start, now;
const char* simulation_modes[] = { "String", "Membrane", "3D Model" };
static const char* current_simulation_mode = simulation_modes[0];

const char* sim_methods[] = { "Cos", "Complex" };
static const char* current_sim_method = sim_methods[0];

float x_rot = 0;
float y_rot = 0;
bool cameraMode = false;
glm::mat4 model = glm::mat4(1.0f);

const Uint32 sampleRate = 48000;
Uint32 floatStreamLength = 1024;
float audioVolume = 0.5;

float audioBuffer[sampleRate * 600];
int bufferWriteIndex = 0;
int bufferReadIndex = 0;

int writtenSamples = 0;
int readSamples = 0;

bool close = false;
int mouseX = 0;
int mouseY = 0;
int deltaMouseX = 0;
int deltaMouseY = 0;

glm::mat4 x_rotMat = glm::mat4(1.0f);
glm::mat4 y_rotMat = glm::mat4(1.0f);
glm::mat4 panMat = glm::mat4(1.0f);
glm::mat4 zoom = glm::mat4(1.0f);


float curSample = 0;
int mSize = massPositionsString.size();
int mfSize = modeFrequenciesString.size();
float* yPositions = new float[mSize];
float t = 0;

float tIncr = 1.0f / sampleRate;
complex<float>* omegaIp;
complex<float>* omegaIm;
complex<float>* omegaIpExp;
complex<float>* omegaImExp;
float* modeGainsTimesModeShapes;
complex<float>* wpiExp;
complex<float>* wmiExp;


float* modeDampingTermsExp;
float* damping;
float* modeFrequenciesArr;

int subWriteIndex = 0;


void storeCosTable() {
	for (int i = 0; i < 2048; i++) {
		cosTable[i] = cos(2 * M_PI * (float)i / 2048.0f);
	}

}

void initMassesString() {
	massPositionsString.clear();
	float spacing = 2.0f / (float)(numMasses + 1);
	massPositionsString.reserve(numMasses);

	for (int i = 0; i < numMasses; i++) {
		massPositionsString.push_back(Vector2f(-1.f + (i + 1) * spacing, 0));
	}

	mouseDistThreshGlSpace = 1.f / (float)(numMasses + 1);
}

float getStiffnessCoeff2d(int i) {
	return springStiffness;
}


void initMassesMembrane() {
	massPositionsMembrane.resize((numMasses + 1) * (numMasses + 1));

	for (int i = 0; i < (numMasses + 1) * (numMasses + 1); i++) {
		massPositionsMembrane[i] = 0.0f;
	}
}

VectorXf getMasssDisplacementVectorString() {
	VectorXf mP(massPositionsString.size());
	for (int i = 0; i < massPositionsString.size(); i++) {
		mP(i) = massPositionsString[i](1);
	}
	return mP;
}

VectorXf getMasssDisplacementVectorMembrane() {
	VectorXf mP(massPositionsMembrane.size());
	for (int n = 0; n < massPositionsMembrane.size(); n++) {
		mP(n) = massPositionsMembrane[n];
	}
	return mP;
}

void initModesString() {

	//construct Stiffness Matrix

	MatrixXf K(numMasses, numMasses);
	K = MatrixXf::Zero(numMasses, numMasses);
	MatrixXf I = MatrixXf::Identity(numMasses, numMasses);

	for (int i = 0; i < numMasses; i++) {
		if (i > 0) {
			K(i, i - 1) = -getStiffnessCoeff2d(i - 1);
		}

		K(i, i) = getStiffnessCoeff2d(i) + getStiffnessCoeff2d(i + 1);

		if (i < numMasses - 1) {
			K(i, i + 1) = -getStiffnessCoeff2d(i + 1);
		}

	}

	GeneralizedSelfAdjointEigenSolver<MatrixXf> es(K, I);

	modeShapesString = es.eigenvectors();
	modeFrequenciesString = es.eigenvalues();
	modeDampingTermsString = es.eigenvalues();

	for (int i = 0; i < modeFrequenciesString.size(); i++) {
		modeFrequenciesString(i) = sqrt(abs(4 * modeFrequenciesString(i) - pow(modeFrequenciesString(i), 2)) / 2.0f);
		modeDampingTermsString(i) = -(dampingCoeff / 1000.0f * es.eigenvalues()(i)) / 2.0;
	}

	modeShapesStringArray = new float* [massPositionsString.size()];

	for (int i = 0; i < massPositionsString.size(); i++) {
		modeShapesStringArray[i] = new float[modeFrequenciesString.size()];
	}

	modeFrequenciesStringArray = new float[modeFrequenciesString.size()];
	modeDampingTermsStringArray = new float[modeDampingTermsString.size()];
	modeGainsStringArray = new float[modeGainsString.size()];

	for (int i = 0; i < massPositionsString.size(); i++) {
		for (int m = 0; m < modeFrequenciesString.size(); m++) {
			modeShapesStringArray[i][m] = modeShapesString(i, m);
		}
	}

	for (int i = 0; i < modeFrequenciesString.size(); i++) {
		modeFrequenciesStringArray[i] = modeFrequenciesString(i);
	}

	for (int i = 0; i < modeDampingTermsString.size(); i++) {
		modeDampingTermsStringArray[i] = modeDampingTermsString(i);
	}

	for (int i = 0; i < modeGainsString.size(); i++) {
		modeGainsStringArray[i] = modeGainsString(i);
	}

	if (numMasses < 4) {
		cout << "K = " << endl << K << endl;
		cout << "The eigenvalues are:" << endl << modeFrequenciesString << endl;
		cout << "The matrix of eigenvectors, V, is:" << endl << modeShapesString << endl << endl;
	}


}

void initModesMembrane() {
	MatrixXf K((numMasses + 1) * (numMasses + 1), (numMasses + 1) * (numMasses + 1));
	K = MatrixXf::Zero((numMasses + 1) * (numMasses + 1), (numMasses + 1) * (numMasses + 1));
	MatrixXf I = MatrixXf::Identity((numMasses + 1) * (numMasses + 1), (numMasses + 1) * (numMasses + 1));

	for (int row = 0; row < (numMasses + 1); row++) {
		for (int col = 0; col < (numMasses + 1); col++) {
			int k = 0;
			//determine number of neighbours for coefficients
			if (row - 1 >= 0) {
				k++;
				K((row - 1) * (numMasses + 1) + col, row * (numMasses + 1) + col) = -1;
			}
			if (row + 1 < numMasses + 1) {
				k++;
				K((row + 1) * (numMasses + 1) + col, row * (numMasses + 1) + col) = -1;
			}
			if (col - 1 >= 0) {
				k++;
				K(row * (numMasses + 1) + col - 1, row * (numMasses + 1) + col) = -1;
			}
			if (col + 1 < numMasses + 1) {
				k++;
				K(row * (numMasses + 1) + col + 1, row * (numMasses + 1) + col) = -1;
			}
			K(row * (numMasses + 1) + col, row * (numMasses + 1) + col) = k;

		}
	}

	GeneralizedSelfAdjointEigenSolver<MatrixXf> es(K, I);

	modeShapesMembrane = es.eigenvectors();
	modeFrequenciesMembrane = es.eigenvalues();
	modeDampingTermsMembrane = es.eigenvalues();

	for (int i = 0; i < modeFrequenciesMembrane.size(); i++) {
		modeFrequenciesMembrane(i) = sqrt(abs(4 * modeFrequenciesMembrane(i) - pow(modeFrequenciesMembrane(i), 2)) / 2.0f);
		modeDampingTermsMembrane(i) = -(dampingCoeff / 1000.0f * es.eigenvalues()(i)) / 2.0;
	}

	if (numMasses < 4) {
		cout << "K = " << endl << K << endl;
		cout << "The eigenvalues (frequencies) are:" << endl << modeFrequenciesMembrane << endl;
		cout << "The matrix of eigenvectors (shapes), V, is:" << endl << modeShapesMembrane << endl << endl;
		cout << "The damping terms are:" << endl << modeDampingTermsMembrane << endl << endl;
	}
}


void drawLine2d(float x1, float y1, float x2, float y2) {
	glLineWidth(lineWidth);
	glColor3f(lineColor[0], lineColor[1], lineColor[2]);
	glBegin(GL_LINES);
	glVertex2f(x1, y1);
	glVertex2f(x2, y2);
	glEnd();
}

void drawDot2d(float x, float y) {
	glPointSize(pointSize);
	glColor3f(pointColor[0], pointColor[1], pointColor[2]);
	glBegin(GL_POINTS);
	glVertex2f(x, y);
	glEnd();
}


void drawString() {
	drawLine2d(-1, 0, massPositionsString[0](0), massPositionsString[0](1));

	for (int i = 0; i < numMasses; i++) {
		drawDot2d(massPositionsString[i](0), massPositionsString[i](1));
		if (i > 0) {
			drawLine2d(massPositionsString[i - 1](0), massPositionsString[i - 1](1), massPositionsString[i](0), massPositionsString[i](1));
		}
	}
	drawLine2d(massPositionsString[numMasses - 1](0), massPositionsString[numMasses - 1](1), 1, 0);
}

void drawMembrane() {
	float spacing = 1.0f / (float)(numMasses + 1);

	for (int i = 0; i < numMasses; i++) {
		for (int u = 0; u < numMasses; u++) {
			glBegin(GL_TRIANGLES);
			glVertex3f(i * spacing - 0.5, u * spacing - 0.5, massPositionsMembrane[i * (numMasses + 1) + u]);
			glVertex3f((i + 1) * spacing - 0.5, u * spacing - 0.5, massPositionsMembrane[(i + 1) * (numMasses + 1) + u]);
			glVertex3f(i * spacing - 0.5, (u + 1) * spacing - 0.5, massPositionsMembrane[i * (numMasses + 1) + (u + 1)]);
			glEnd();

			glBegin(GL_TRIANGLES);
			glVertex3f((i + 1) * spacing - 0.5, (u + 1) * spacing - 0.5, massPositionsMembrane[(i + 1) * (numMasses + 1) + u + 1]);
			glVertex3f(i * spacing - 0.5, (u + 1) * spacing - 0.5, massPositionsMembrane[i * (numMasses + 1) + u + 1]);
			glVertex3f((i + 1) * spacing - 0.5, u * spacing - 0.5, massPositionsMembrane[(i + 1) * (numMasses + 1) + u]);
			glEnd();
		}
	}

}

void draw3D() {

}

float distance2d(Vector2f v1, Vector2f v2) {
	return sqrt(pow(v1(0) - v2(0), 2) + pow(v1(1) - v2(1), 2));
}

float distance2d(Vector2f v1, float x, float y) {
	return distance2d(v1, Vector2f(x, y));
}

float distance3d(glm::vec3 v1, glm::vec3 v2) {
	return sqrt(pow(v1[0] - v2[0], 2) + pow(v1[1] - v2[1], 2) + pow(v1[2] - v2[2], 2));
}

float pointLineDistance3d(glm::vec3 line1, glm::vec3 line2, glm::vec3 point) {
	return glm::length(glm::cross(point - line1, point - line2)) / glm::length(line1 - line2);
}

Vector2f screenSpaceToGlSpace(int x, int y) {
	return Vector2f((((float)x / (float)windowSizeX) - 0.5f) * 2.f, -(((float)y / (float)windowSizeY) - 0.5f) * 2.f);
}

glm::vec3 screenSpaceToGLSpace3D(int x, int y) {
	return model * glm::vec4(((float)x / (float)windowSizeX - 0.5f) * 2.0f, -((float)y / (float)windowSizeY - 0.5f) * 2.f, 0, 0);

}



void audioStringSimCos() {

	while (run) {
		curSample = 0;

		//iterate over all modes
		for (int i = 0; i < mSize; i++) {

			damping[i] *= modeDampingTermsExp[i];
			float cosT = damping[i] * cos(t * modeFrequenciesArr[i]);

			//iterate over all mass points
			for (int m = 0; m < mfSize; m++) {
				curSample += modeGainsTimesModeShapes[m * mSize + i] * cosT;

			}
		}


		t += tIncr;
		audioBuffer[bufferWriteIndex] = curSample * audioVolume;
		bufferWriteIndex++;
		//writtenSamples++;

	}

}

void audioStringSimCmplx() {

	while (run) {
		curSample = 0;

		for (int i = 0; i < mSize; i++) {
			wpiExp[i] *= omegaIpExp[i];
			wmiExp[i] *= omegaImExp[i];
			float wi = (wpiExp[i] + wmiExp[i]).real();

			for (int m = 0; m < mfSize; m++) {
				curSample += modeGainsTimesModeShapes[m * mSize + i] * wi;

			}
		}


		// t += tIncr;
		audioBuffer[bufferWriteIndex] = curSample * audioVolume;
		bufferWriteIndex++;
		//writtenSamples++;

	}

}


void audioCallback(void* unused, Uint8* byteStream, int byteStreamLength) {
	float* floatStream = (float*)byteStream;

	/*
	std::chrono::duration<float> dur = std::chrono::high_resolution_clock::now() - audio;
	audio = std::chrono::high_resolution_clock::now();

	cout << dur.count() << endl;*/

	Uint32 i;
	for (i = 0; i < floatStreamLength; i++) {


		if (bufferReadIndex == bufferWriteIndex) {
			//cout << "Underflow! " << readSamples << endl;
			readSamples++;
		}
		else {
			floatStream[i] = audioBuffer[bufferReadIndex];
			//cout << "read: " << audioBuffer[bufferWriteIndex] << " @ " << bufferReadIndex << endl;
			bufferReadIndex++;
		}
	}
}


void startStringSim() {
	run = true;
	initModesString();
	modeGainsString = modeShapesString.inverse() * getMasssDisplacementVectorString();

	modeGainsStringArray = new float[modeGainsString.size()];
	for (int i = 0; i < modeGainsString.size(); i++) {
		modeGainsStringArray[i] = modeGainsString(i);
	}

	cout << "Mode Gains: " << endl;
	for (int i = 0; i < modeGainsString.size(); i++) {
		cout << "Mode " << i << " : " << modeGainsString(i) << endl;
	}


	mSize = massPositionsString.size();
	mfSize = modeFrequenciesString.size();

	omegaIp = new complex<float>[mfSize];
	omegaIm = new complex<float>[mfSize];
	omegaIpExp = new complex<float>[mfSize];
	omegaImExp = new complex<float>[mfSize];
	modeGainsTimesModeShapes = new float[mSize * mfSize];
	wpiExp = new complex<float>[mSize];
	wmiExp = new complex<float>[mSize];


	for (int m = 0; m < mfSize; m++) {
		*(omegaIp + m) = complex<float>(modeDampingTermsString(m), modeFrequenciesString(m) * 2 * M_PI);
	}


	for (int m = 0; m < mfSize; m++) {
		*(omegaIm + m) = complex<float>(modeDampingTermsString(m), -modeFrequenciesString(m) * 2 * M_PI);
	}


	for (int m = 0; m < mfSize; m++) {
		*(omegaIpExp + m) = exp(*(omegaIp + m) * tIncr);
	}


	for (int m = 0; m < mfSize; m++) {
		*(omegaImExp + m) = exp(*(omegaIm + m) * tIncr);
	}


	for (int i = 0; i < mSize; ++i) {
		for (int m = 0; m < mfSize; m++) {
			*(modeGainsTimesModeShapes + i * mSize + m) = (modeGainsString(m) * modeShapesString(i, m)) / 2;
		}
	}

	for (int m = 0; m < mSize; m++) {
		*(wpiExp + m) = complex<float>(1, 0);
	}


	for (int m = 0; m < mSize; m++) {
		*(wmiExp + m) = complex<float>(1, 0);
	}


	modeDampingTermsExp = new float[mfSize];
	damping = new float[mfSize];
	modeFrequenciesArr = new float[mfSize];


	for (int m = 0; m < mfSize; m++) {
		*(modeDampingTermsExp + m) = exp(modeDampingTermsString(m) * tIncr);
	}


	for (int m = 0; m < mfSize; m++) {
		*(damping + m) = 1;
	}



	for (int m = 0; m < mfSize; m++) {
		*(modeFrequenciesArr + m) = modeFrequenciesString(m) * 2 * M_PI;
	}


	cout << "Run String Simulation = " << run << endl;
	start = std::chrono::high_resolution_clock::now();
	if (current_sim_method == sim_methods[0]) {
		std::thread t1(audioStringSimCos);
		t1.detach();
	}
	if (current_sim_method == sim_methods[1]) {
		std::thread t1(audioStringSimCmplx);
		t1.detach();
	}
	
	bufferReadIndex = 0;
	bufferWriteIndex = 0;
	SDL_PauseAudioDevice(audio_device, 0);

	now = std::chrono::high_resolution_clock::now();
	subWriteIndex = 0;
}

vector<long> dampingTermExecutionTimes;
vector<long> modeGainTermExecutionTimes;
vector<long> cosTermExecutionTimes;
vector<long> modeShapeTermExecutionTimes;
vector<long> massPositionMultiplicationExecutionTimes;


void stopStringSim() {
	run = false;
	initModesString();
	SDL_PauseAudioDevice(audio_device, 1);
	readSamples = 0;
	writtenSamples = 0;

	/*
	if (dampingTermExecutionTimes.size() != 0) {
		long dampingTermExecutionTimesAvg = 0;
		for (unsigned int i = 0; i < dampingTermExecutionTimes.size(); i++) {
			dampingTermExecutionTimesAvg += dampingTermExecutionTimes[i];
		}
		dampingTermExecutionTimesAvg = dampingTermExecutionTimesAvg / dampingTermExecutionTimes.size();

		long modeGainTermExecutionTimesAvg = 0;
		for (unsigned int i = 0; i < modeGainTermExecutionTimes.size(); i++) {
			modeGainTermExecutionTimesAvg += modeGainTermExecutionTimes[i];
		}
		modeGainTermExecutionTimesAvg = modeGainTermExecutionTimesAvg / modeGainTermExecutionTimes.size();

		long cosTermExecutionTimesAvg = 0;
		for (unsigned int i = 0; i < cosTermExecutionTimes.size(); i++) {
			cosTermExecutionTimesAvg += cosTermExecutionTimes[i];
		}
		cosTermExecutionTimesAvg = cosTermExecutionTimesAvg / cosTermExecutionTimes.size();

		long modeShapeTermExecutionTimesAvg = 0;
		for (unsigned int i = 0; i < modeShapeTermExecutionTimes.size(); i++) {
			modeShapeTermExecutionTimesAvg += modeShapeTermExecutionTimes[i];
		}
		modeShapeTermExecutionTimesAvg = modeShapeTermExecutionTimesAvg / modeShapeTermExecutionTimes.size();

		long massPositionMultiplicationExecutionTimesAvg = 0;
		for (unsigned int i = 0; i < massPositionMultiplicationExecutionTimes.size(); i++) {
			massPositionMultiplicationExecutionTimesAvg += massPositionMultiplicationExecutionTimes[i];
		}
		massPositionMultiplicationExecutionTimesAvg = massPositionMultiplicationExecutionTimesAvg / massPositionMultiplicationExecutionTimes.size();


		cout << "Execution Times:" << endl;
		cout << "dampingTermExecutionTime avg (microseconds) " << dampingTermExecutionTimesAvg << endl;
		cout << "modeGainTermExecutionTime avg (microseconds) " << modeGainTermExecutionTimesAvg << endl;
		cout << "cosTermExecutionTimes avg (microseconds) " << cosTermExecutionTimesAvg << endl;
		cout << "modeShapeTermExecutionTime avg (microseconds) " << modeShapeTermExecutionTimesAvg << endl;
		cout << "massPositionMultiplicationExecutionTime avg (microseconds) " << massPositionMultiplicationExecutionTimesAvg << endl;
	}*/

}

void startMembraneSim() {
	run = true;
	initModesMembrane();
	modeGainsMembrane = modeShapesMembrane.inverse() * getMasssDisplacementVectorMembrane();
	cout << "Mode Gains: " << endl;
	for (int i = 0; i < modeGainsMembrane.size(); i++) {
		cout << "Mode " << i << " : " << modeGainsMembrane(i) << endl;
	}
	cout << "Run Membrane Simulation = " << run << endl;
	start = std::chrono::high_resolution_clock::now();
}

void stopMembraneSim() {
	run = false;
	initModesMembrane();
}


void stringUpdate() {
	now = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> dur = now - start;
	float t = dur.count();
	//float sample = 0;
	//cout << "hallo" << endl;

	for (int i = 0; i < massPositionsString.size(); i++) {
		massPositionsString[i](1) = 0;
		for (int m = 0; m < modeFrequenciesString.size(); m++) {
			//auto start = std::chrono::high_resolution_clock::now();
			float damping = exp(modeDampingTermsStringArray[m] * t);
			//auto elapsed = std::chrono::high_resolution_clock::now() - start;
			//dampingTermExecutionTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

			//start = std::chrono::high_resolution_clock::now();
			float modeGain = modeGainsStringArray[m];
			//elapsed = std::chrono::high_resolution_clock::now() - start;
			//modeGainTermExecutionTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

			//start = std::chrono::high_resolution_clock::now();
			float cosTerm = cos(t * modeFrequenciesStringArray[m]);
			//elapsed = std::chrono::high_resolution_clock::now() - start;
			//cosTermExecutionTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

			//start = std::chrono::high_resolution_clock::now();
			float modeShape = modeShapesStringArray[i][m];
			//elapsed = std::chrono::high_resolution_clock::now() - start;
			//modeShapeTermExecutionTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

			//start = std::chrono::high_resolution_clock::now();
			massPositionsString[i](1) += damping * modeGain * modeShape * cosTerm;
			//cout << massPositionsString[i](1) << endl;
			//elapsed = std::chrono::high_resolution_clock::now() - start;
			//massPositionMultiplicationExecutionTimes.push_back(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
		}
	}
}

void membraneUpdate() {
	//now = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> dur = now - start;
	float t = dur.count();
	//float sample = 0;
	for (int n = 0; n < massPositionsMembrane.size(); n++) {
		massPositionsMembrane[n] = 0;
		for (int m = 0; m < modeFrequenciesMembrane.size(); m++) {
			massPositionsMembrane[n] += exp(modeDampingTermsMembrane(m) * t) * modeGainsMembrane(m) * cos(t * modeFrequenciesMembrane(m)) * modeShapesMembrane(n, m);
		}
	}
}

void showModeShapeString() {
	for (int i = 0; i < massPositionsString.size(); i++) {
		massPositionsString[i](1) = modeShapesString(i, displayModeShapeNum - 1);
	}
}

void showModeShapeMembrane() {
	for (int n = 0; n < massPositionsMembrane.size(); n++) {
		massPositionsMembrane[n] = modeShapesMembrane(n, displayModeShapeNum - 1);
	}
}

void initSDL() {
	SDL_Init(SDL_INIT_EVERYTHING);

	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	window = SDL_CreateWindow("Vibrating String App v0.1", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, windowSizeX, windowSizeY, SDL_WINDOW_OPENGL);
	glContext = SDL_GL_CreateContext(window);
	SDL_GL_MakeCurrent(window, glContext);
}


int initGlew() {
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		cout << glewGetErrorString(err) << endl;
		cin.get();
		return -1;
	}

	std::cout << glGetString(GL_VERSION) << endl;
}

void initIMGUI() {
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplSDL2_InitForOpenGL(window, glContext);
	ImGui_ImplOpenGL3_Init(NULL);
}

void drawGUI() {
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame(window);

	ImGui::SetNextWindowPos(ImVec2(0, 0));

	ImGui::NewFrame();

	// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.


	ImGui::Begin("Settings");                          // Create a window called "Hello, world!" and append into it.

	//ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
	//ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state

	if (ImGui::BeginCombo("##combo", current_simulation_mode)) // The second parameter is the label previewed before opening the combo.
	{
		for (int n = 0; n < IM_ARRAYSIZE(simulation_modes); n++)
		{
			bool is_selected = (current_simulation_mode == simulation_modes[n]); // You can store your selection however you want, outside or inside your objects
			if (ImGui::Selectable(simulation_modes[n], is_selected)) {
				current_simulation_mode = simulation_modes[n];
				if (is_selected)
					ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
			}
		}
		ImGui::EndCombo();
	}

	if (ImGui::BeginCombo("##sim method", current_sim_method)) // The second parameter is the label previewed before opening the combo.
	{
		for (int n = 0; n < IM_ARRAYSIZE(sim_methods); n++)
		{
			bool is_selected = (current_sim_method == sim_methods[n]); // You can store your selection however you want, outside or inside your objects
			if (ImGui::Selectable(sim_methods[n], is_selected)) {
				current_sim_method = sim_methods[n];
				if (is_selected)
					ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
			}
		}
		ImGui::EndCombo();
	}

	ImGui::SliderInt("# masses", &sliderMasses, 2, 100);
	ImGui::SliderFloat("Stiffness", &springStiffness, 0.1, 60000);
	ImGui::SliderFloat("Damping", &dampingCoeff, 0.01, 10);
	ImGui::SliderFloat("Audio Volume", &audioVolume, 0, 1);

	ImGui::Checkbox("Display Mode Shape", &displayModeShape);
	if (displayModeShape) {
		if (current_simulation_mode == simulation_modes[0]) {
			ImGui::SliderInt("Mode Shape #", &displayModeShapeNum, 1, numMasses);
		}
		else if (current_simulation_mode == simulation_modes[1]) {
			ImGui::SliderInt("Mode Shape #", &displayModeShapeNum, 1, (numMasses + 1) * (numMasses + 1));
		}
	}

	//ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	//ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

	if (!displayModeShape) {
		if (run) {
			if (ImGui::Button("Stop")) {
				if (current_simulation_mode == simulation_modes[0]) {
					stopStringSim();
				}
				else if (current_simulation_mode == simulation_modes[1]) {
					stopMembraneSim();
				}
			}
		}
		else {
			if (ImGui::Button("Start")) {
				if (current_simulation_mode == simulation_modes[0]) {
					startStringSim();
				}
				else if (current_simulation_mode == simulation_modes[1]) {
					startMembraneSim();
				}

			}
		}

		ImGui::SameLine();


		if (ImGui::Button("Reset")) {
			run = false;
			if (current_simulation_mode == simulation_modes[0]) {
				stopStringSim();
				initMassesString();
				initModesString();
			}
			else if (current_simulation_mode == simulation_modes[1]) {
				stopMembraneSim();
				initMassesMembrane();
				initModesMembrane();
			}
		}

		ImGui::SameLine();
	}


	if (!displayModeShape && current_simulation_mode == simulation_modes[1]) {
		if (cameraMode) {
			if (ImGui::Button("Edit Mode")) {
				cameraMode = false;
			}
		}
		else {
			if (ImGui::Button("Camera Mode")) {
				cameraMode = true;
			}
		}
		ImGui::SameLine();
	}
	else {
		cameraMode = true;
	}



	//ImGui::Text("counter = %d", counter);

	ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

	ImGui::End();

	// Rendering
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void drawClearFrame() {
	glViewport(0, 0, windowSizeX, windowSizeY);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

void handleEvents() {
	//Event handling (mouse clicks/drags)
	SDL_Event event;
	while (SDL_PollEvent(&event)) {

		ImGui_ImplSDL2_ProcessEvent(&event);

		if (event.type == SDL_QUIT) {
			close = true;
		}

		//start string sim with right mouse button
		if (current_simulation_mode == simulation_modes[0] && !displayModeShape && event.button.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_RIGHT) {

			if (!run) {
				startStringSim();
			}
			else {
				stopStringSim();
			}

		}

		//update number of masses dynmically
		if (!displayModeShape && event.button.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
			if (numMasses != sliderMasses) {
				numMasses = sliderMasses;
				initMassesString();
				stopStringSim();
				//initMassesMembrane();
				//stopMembraneSim();

			}
		}

		//membrane camera pan
		if (cameraMode && event.button.type == SDL_MOUSEMOTION && event.button.button == SDL_BUTTON(SDL_BUTTON_RIGHT)) {
			glm::vec3 trans = glm::vec3(panMat[3]);
			if (!(trans.x < -1.0 && event.motion.xrel < 0 || trans.x > 1.0 && event.motion.xrel > 0)) {
				panMat = glm::translate(panMat, glm::vec3(event.motion.xrel / 350.0f, 0, 0));
			}
			if (!(trans.y > 1.0 && event.motion.yrel < 0 || trans.y < -1.0 && event.motion.yrel > 0)) {
				panMat = glm::translate(panMat, glm::vec3(0, -event.motion.yrel / 350.0f, 0));
			}
		}

		//membrane mesh displace
		if (!displayModeShape && event.button.type == SDL_MOUSEMOTION && event.button.button == SDL_BUTTON_LEFT) {
			if (!cameraMode && current_simulation_mode == simulation_modes[1]) {
				Vector2f rayStart = screenSpaceToGlSpace(event.motion.x, event.motion.y);

				float spacing = 1.0f / (float)(numMasses + 1);
				for (int u = 0; u < numMasses + 1; u++) {
					for (int v = 0; v < numMasses + 1; v++) {
						glm::vec3 transformedMassPos = model * glm::vec4(spacing * u - 0.5, spacing * v - 0.5, massPositionsMembrane[u * (numMasses + 1) + v], 0);
						if (pointLineDistance3d(glm::vec3(rayStart(0), rayStart(1), 0), glm::vec3(rayStart(0), rayStart(1), 1), transformedMassPos) < spacing) {
							massPositionsMembrane[u * (numMasses + 1) + v] += -0.005;
						}
					}
				}

			}
		}

		if (ImGui::GetIO().WantCaptureMouse != 1 && !run && event.type == SDL_MOUSEMOTION && event.button.button == SDL_BUTTON_LEFT)
		{
			mouseX = event.motion.x;
			mouseY = event.motion.y;
			x_rot = event.motion.yrel / 500.0f;
			y_rot = event.motion.xrel / 500.0f;

			//cout << event.motion.xrel << " " << event.motion.yrel << endl;

			//string mesh displace
			if (!displayModeShape && current_simulation_mode == simulation_modes[0]) {

				for (int i = 0; i < massPositionsString.size(); i++) {
					Vector2f mouseGl = screenSpaceToGlSpace(mouseX, mouseY);
					if (abs(massPositionsString[i](0) - mouseGl(0)) < mouseDistThreshGlSpace && abs(massPositionsString[i](1) - mouseGl(1) < 0.5)) {
						massPositionsString[i](1) = -(((float)mouseY / (float)windowSizeY) - 0.5f) * 2.f;
					}
				}
			}//membrane camera rotate
			else if (cameraMode && current_simulation_mode == simulation_modes[1]) {
				//x_rot += (float) (mouseX - deltaMouseX)/5000.0f;
				//y_rot += (float) (mouseY - deltaMouseY)/5000.0f;

				x_rotMat = glm::rotate(x_rotMat, x_rot, glm::vec3(1, 0, 0));
				y_rotMat = glm::rotate(y_rotMat, y_rot, glm::vec3(0, 0, 1));
			}

			//cout << "Mouse = (" << screenSpaceToGlSpace(mouseX, mouseY)(0) << ", " << screenSpaceToGlSpace(mouseX, mouseY)(1) << ")" << endl;
		}

		//zoom
		if (cameraMode && event.type == SDL_MOUSEWHEEL)
		{
			cout << event.wheel.y << endl;
			float scale = 1 + event.wheel.y / 60.0f;
			zoom = glm::scale(zoom, glm::vec3(scale, scale, scale));
		}

	}
}

void initAudio() {
	// opening an audio device:
	SDL_AudioSpec want;
	SDL_AudioSpec audioSpec;
	SDL_zero(want);
	want.freq = sampleRate;
	want.format = AUDIO_F32SYS;
	want.channels = 1;
	want.samples = 1024;
	want.callback = audioCallback;

	audio_device = SDL_OpenAudioDevice(NULL, 0, &want, &audioSpec, SDL_AUDIO_ALLOW_FORMAT_CHANGE);

	if (audioSpec.format != want.format) {
		printf("\nCouldn't get Float32 audio format.\n");
	}

	SDL_PauseAudioDevice(audio_device, 1);
}

int main(int argc, char** argv) {

	initSDL();
	initGlew();
	initIMGUI();
	storeCosTable();

	initMassesString();
	initModesString();
	initMassesMembrane();
	initModesMembrane();
	initAudio();

	Shader shader("basic.vs", "basic.fs");
	shader.bind();
	int ModelMatrixUniformLocation = glGetUniformLocation(shader.getShaderId(), "u_model");

	

	while (!close) {

		drawClearFrame();
		drawGUI();


		//String mode

		if (current_simulation_mode == simulation_modes[0]) {
			model = glm::mat4(1.0f);
			glUniformMatrix4fv(ModelMatrixUniformLocation, 1, GL_FALSE, &model[0][0]);
			if (displayModeShape) {
				showModeShapeString();
			}
			drawString();
		}

		//Membrane Mode
		else if (current_simulation_mode == simulation_modes[1]) {
			model = zoom * panMat * x_rotMat * y_rotMat;
			glUniformMatrix4fv(ModelMatrixUniformLocation, 1, GL_FALSE, &model[0][0]);
			if (displayModeShape) {
				showModeShapeMembrane();
			}
			drawMembrane();
		}


		//3D Mode
		else if (current_simulation_mode == simulation_modes[2]) {
			draw3D();
		}


		if (run) {
			if (current_simulation_mode == simulation_modes[0]) {
				stringUpdate();
			}
			else if (current_simulation_mode == simulation_modes[1]) {
				membraneUpdate();
			}
			/*
			auto elapsed = std::chrono::high_resolution_clock::now() - now;
			if ((elapsed / std::chrono::milliseconds(1)) > 1000) {
				now = std::chrono::high_resolution_clock::now();
				int idx = bufferWriteIndex;
				cout << idx - subWriteIndex << endl;
				subWriteIndex = idx;
			}*/


			//cout << "underflow = " << readSamples - writtenSamples << endl;
		}


		SDL_GL_SwapWindow(window);
		handleEvents();
	}

	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	SDL_GL_DeleteContext(glContext);
	SDL_CloseAudioDevice(audio_device);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}
