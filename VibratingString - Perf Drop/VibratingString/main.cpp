#include <iostream>
#include <chrono>

using namespace std;

int bufferWriteIndex = 0;
float curSample = 0;

float damping[5] = { 1, 1, 1, 1, 1 };

float modeDampingTermsExp[5] = { 0.447604, 0.0497871, 0.00247875, 0.00012341, 1.37263e-05 };
float modeDampingTermsExp2[5] = { -0.803847, -3, -6, -9, -11.1962 };


int main(int argc, char** argv) {
	
	float subt = 0;
	int subWriteIndex = 0;
	auto now = std::chrono::high_resolution_clock::now();
	

	while (true) {
		
		curSample = 0;

		for (int i = 0; i < 5; i++) {

			//Slow version
			damping[i] = damping[i] * modeDampingTermsExp[i];

			if (std::fpclassify(damping[i]) == FP_SUBNORMAL) {
				damping[i] = 0; // Treat denormals as 0.
			}

			//Fast version
			//damping[i] = damping[i] * modeDampingTermsExp[i];
			float cosT = 2 * damping[i];


			for (int m = 0; m < 5; m++) {
				curSample += cosT;

			}
		}

		//t += tIncr;
		bufferWriteIndex++;


		//measure calculations per second
		auto elapsed = std::chrono::high_resolution_clock::now() - now;
		if ((elapsed / std::chrono::milliseconds(1)) > 1000) {
			now = std::chrono::high_resolution_clock::now();
			int idx = bufferWriteIndex;
			cout << idx - subWriteIndex << endl;
			subWriteIndex = idx;
		}

	}
}
