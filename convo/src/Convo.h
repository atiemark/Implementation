#pragma once

#include "sdl/SDL.h"
#include <atomic>
#include "Convolve.h"

class Convo
{
public:
    Convo();
    bool Init();
    bool InitAudio();
    ~Convo();

    void Run();

private:
    void HandleEvents();

    static void AudioCallback( void *userData, Uint8 *stream, int length );

private:
    SDL_Window *window = nullptr;
    SDL_AudioStream *outStream = nullptr;
    SDL_AudioStream *sampleStream = nullptr;
    bool running = false;
    bool sdl = false;
    SDL_AudioDeviceID device;
    SDL_AudioSpec targetSpec;
    SDL_AudioSpec sourceSpec;

    std::atomic_bool play[5] = { false };
    Uint8 *buffer[5];
    Uint32 length[5];
    Uint8 *filterBuffer;
    Uint32 filterLength;
    float *filterBlock;

    Convolve convolver;
};
