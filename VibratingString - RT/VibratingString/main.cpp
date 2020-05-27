#include <iostream>

#define SDL_MAIN_HANDLED
#include <SDL.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <math.h>
#include <chrono>
#include <ctime>
#include <thread> 
#include <complex>
#include <cmath>

#pragma comment(lib, "SDL2.lib")


using namespace std;
using namespace Eigen;

struct MVertex {
	float x;
	float y;
	float z;
};

SDL_AudioDeviceID audio_device;

int numMasses = 10;
int sliderMasses = numMasses;
float massWeight = 1;
float springStiffness = 60000;
float dampingCoeff = 1;

vector<Vector2f> massPositionsString;
MatrixXf modeShapesString;
VectorXf modeFrequenciesString;
VectorXf modeDampingTermsString;
VectorXf modeGainsString;


chrono::time_point<chrono::high_resolution_clock> start, now;

const Uint32 sampleRate = 48000;
Uint32 floatStreamLength = 1024;
int audioVolume = 1;

float audioBuffer[sampleRate * 60];
int bufferWriteIndex = 0;
int bufferReadIndex = 0;

int writtenSamples = 0;
int readSamples = 0;

bool close = false;





float curSample = 0;
float curSampleC = 0;
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


float* modeDampingTermsExp = new float[mfSize];
float* damping = new float[mfSize];
float* modeFrequenciesArr = new float[mfSize];




void initMassesString() {
	massPositionsString.clear();
	float spacing = 2.0f / (float)(numMasses + 1);
	massPositionsString.reserve(numMasses);

	for (int i = 0; i < numMasses; i++) {
		massPositionsString.push_back(Vector2f(-1.f + (i + 1) * spacing, 0));
	}
}

float getStiffnessCoeff2d(int i) {
	return springStiffness;
}


VectorXf getMasssDisplacementVectorString() {
	VectorXf mP(massPositionsString.size());
	for (int i = 0; i < massPositionsString.size(); i++) {
		mP(i) = massPositionsString[i](1);
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


	if (numMasses < 4) {
		cout << "K = " << endl << K << endl;
		cout << "The eigenvalues are:" << endl << modeFrequenciesString << endl;
		cout << "The matrix of eigenvectors, V, is:" << endl << modeShapesString << endl << endl;
	}


}



float distance2d(Vector2f v1, Vector2f v2) {
	return sqrt(pow(v1(0) - v2(0), 2) + pow(v1(1) - v2(1), 2));
}

float distance2d(Vector2f v1, float x, float y) {
	return distance2d(v1, Vector2f(x, y));
}



void audioStringSimCos() {

	while (true) {
		curSample = 0;

		for (int i = 0; i < mSize; i++) {

			damping[i] *= modeDampingTermsExp[i];

			if (std::fpclassify(damping[i]) == FP_SUBNORMAL) {
				damping[i] = 0; // Treat denormals as 0.
			}

			float cosT = 2 *damping[i] * cos(t * modeFrequenciesArr[i]);

			for (int m = 0; m < mfSize; m++) {
				curSample += modeGainsTimesModeShapes[i * mSize + m] * cosT;

			}
		}


		t += tIncr;
		audioBuffer[bufferWriteIndex /*% (sampleRate * 100)*/] = curSample;
		bufferWriteIndex++;
		//writtenSamples++;

		/*
		83123
		113368
		147225
		159761
		195857
		151634
		332146
		149452
		155595
		340800
		366915
		*/

	}

}

void audioStringSimCmplx() {

	while (true) {
		curSample = 0;

		for (int i = 0; i < mSize; i++) {
			wpiExp[i] *= omegaIpExp[i];
			wmiExp[i] *= omegaImExp[i];
			float wi = (wpiExp[i] + wmiExp[i]).real();

			for (int m = 0; m < mfSize/2; m++) {
				curSample += modeGainsTimesModeShapes[m * mSize + i] * wi;

			}
		}


		//t += tIncr;
		audioBuffer[bufferWriteIndex /*% (sampleRate * 100)*/] = curSample;
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

		floatStream[i] = audioBuffer[bufferReadIndex];
		//cout << "read: " << audioBuffer[bufferWriteIndex] << " @ " << bufferReadIndex << endl;
		bufferReadIndex++;

		/*
		if (bufferReadIndex == bufferWriteIndex) {
			cout << "Underflow! " << readSamples << endl;
			readSamples++;
		}
		else {
			floatStream[i] = audioBuffer[bufferReadIndex];
			//cout << "read: " << audioBuffer[bufferWriteIndex] << " @ " << bufferReadIndex << endl;
			bufferReadIndex++;
		}*/
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

	numMasses = 30;
	int initialStringModeShape = 0;

	initSDL();
	initAudio();

	initMassesString();
	initModesString();

	for (int i = 0; i < massPositionsString.size(); i++) {
		massPositionsString[i](1) = modeShapesString(i, initialStringModeShape);
	}
	
	modeGainsString = modeShapesString.inverse() * getMasssDisplacementVectorString();

	/*
	cout << "Mode Gains: " << endl;
	for (int i = 0; i < modeGainsString.size(); i++) {
		cout << "Mode " << i << " : " << modeGainsString(i) << endl;
	}*/

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


	bufferReadIndex = 0;
	bufferWriteIndex = 0;
	auto now = std::chrono::high_resolution_clock::now();
	std::thread t1(audioStringSimCos);
	t1.detach();
	float subt = 0;
	int subWriteIndex = 0;
	SDL_PauseAudioDevice(audio_device, 0);

	while (true) {
		auto elapsed = std::chrono::high_resolution_clock::now() - now;
		if ((elapsed/ std::chrono::milliseconds(1)) > 1000) {
			now = std::chrono::high_resolution_clock::now();
			int idx = bufferWriteIndex;
			cout << idx - subWriteIndex << endl;
			subWriteIndex = idx;
		}
		//stringUpdate();
	}

	return 0;
}
