#include "Convo.h"
#include <iostream>
#include <cassert>

#define SAMPLES 8192

Convo::Convo()
#if _DEBUG_VCV
    : convolver( SAMPLES )
#endif
{

}

bool Convo::Init()
{
    if ( SDL_Init( SDL_INIT_EVERYTHING ) ) {
        std::cerr << "SDL_Init: " << SDL_GetError() << '\n';
        return false;
    }

    window = SDL_CreateWindow( "IFFT",
                               SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                               1280, 720,
                               SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN );

    if ( window == nullptr ) {
        std::cerr << "SDL_CreateWindow: " << SDL_GetError() << '\n';
        return false;
    }


    return true;
}

bool Convo::InitAudio()
{
    sourceSpec.freq = 44100;
    sourceSpec.format = AUDIO_F32;
    sourceSpec.channels = 1;
    sourceSpec.samples = SAMPLES;
    sourceSpec.userdata = this;
    sourceSpec.callback = AudioCallback;

    device = SDL_OpenAudioDevice( nullptr,
                                  0,
                                  &sourceSpec,
                                  &targetSpec,
                                  SDL_AUDIO_ALLOW_ANY_CHANGE );

    std::cout << "\nRequested audio spec:\n"
        << "  Sample frequency: " << sourceSpec.freq << '\n'
        << "  Format          : " << sourceSpec.format << '\n'
        << "  Channels        : " << static_cast<int>(sourceSpec.channels) << '\n'
        << "  Frame size      : " << sourceSpec.samples << '\n';

    std::cout << "\nObtained audio spec:\n"
        << "  Sample frequency: " << targetSpec.freq << '\n'
        << "  Format          : " << targetSpec.format << '\n'
        << "  Channels        : " << static_cast<int>(targetSpec.channels) << '\n'
        << "  Frame size      : " << targetSpec.samples << '\n';


    outStream = SDL_NewAudioStream(
        sourceSpec.format, sourceSpec.channels, sourceSpec.freq,
        targetSpec.format, targetSpec.channels, targetSpec.freq
    );

    SDL_AudioSpec spec;

    SDL_LoadWAV( "res/cricket.wav", &spec, &buffer[0], &length[0] );
    SDL_LoadWAV( "res/dripping.wav", &spec, &buffer[1], &length[1] );
    SDL_LoadWAV( "res/drumset.wav", &spec, &buffer[2], &length[2] );
    SDL_LoadWAV( "res/footsteps on concrete.wav", &spec, &buffer[3], &length[3] );
    SDL_LoadWAV( "res/Gunshot.wav", &spec, &buffer[4], &length[4] );

    sampleStream = SDL_NewAudioStream(
        spec.format, spec.channels, spec.freq,
        AUDIO_F32, 1, 44100
    );

    SDL_AudioSpec filterSpec {};
    SDL_LoadWAV( "res/garage.wav", &filterSpec, &filterBuffer, &filterLength );
    SDL_AudioStream *filterStream = SDL_NewAudioStream(
        filterSpec.format, filterSpec.channels, filterSpec.freq,
        AUDIO_F32, 1, 44100
    );

    SDL_AudioStreamPut( filterStream, filterBuffer, filterLength );
    SDL_AudioStreamFlush( filterStream );
    int available = SDL_AudioStreamAvailable( filterStream );
    filterBlock = new float[available / sizeof( float )];
    SDL_AudioStreamGet( filterStream, filterBlock, available );
    SDL_FreeWAV( filterBuffer );
    SDL_FreeAudioStream( filterStream );

    convolver.setKernel( filterBlock, available / sizeof( float ) );
    delete[] filterBlock;

    SDL_PauseAudioDevice( device, 0 );

    return true;
}

Convo::~Convo()
{
    if ( sdl ) {
        if ( window ) {
            SDL_DestroyWindow( window );
        }

        SDL_CloseAudioDevice( device );

        SDL_FreeWAV( buffer[0] );
        SDL_FreeWAV( buffer[1] );
        SDL_FreeWAV( buffer[2] );
        SDL_FreeWAV( buffer[3] );
        SDL_FreeWAV( buffer[4] );
        SDL_FreeAudioStream( outStream );
        SDL_FreeAudioStream( sampleStream );

        SDL_Quit();
    }
}

void Convo::Run()
{
    if ( !InitAudio() ) return;

    running = true;
    while ( running ) {
        HandleEvents();

        SDL_Delay( 1 );
    }
}

void Convo::HandleEvents()
{
    SDL_Event event;

    while ( SDL_PollEvent( &event ) ) {
        switch ( event.type ) {
        case SDL_QUIT:
            running = false;
            break;

        case SDL_KEYDOWN:
            switch ( event.key.keysym.scancode ) {
            case SDL_SCANCODE_ESCAPE:
                running = false;
                break;

            case SDL_SCANCODE_1:
                play[0] = event.key.repeat == 0; break;

            case SDL_SCANCODE_2:
                play[1] = event.key.repeat == 0; break;

            case SDL_SCANCODE_3:
                play[2] = event.key.repeat == 0; break;

            case SDL_SCANCODE_4:
                play[3] = event.key.repeat == 0; break;

            case SDL_SCANCODE_5:
                play[4] = event.key.repeat == 0; break;
            }
            break;

        case SDL_KEYUP:
            switch ( event.key.keysym.scancode ) {
            case SDL_SCANCODE_1:
                play[0] = false; break;

            case SDL_SCANCODE_2:
                play[1] = false; break;

            case SDL_SCANCODE_3:
                play[2] = false; break;

            case SDL_SCANCODE_4:
                play[3] = false; break;

            case SDL_SCANCODE_5:
                play[4] = false; break;
            }
            break;
        }
    }
}

void Convo::AudioCallback( void * userData, Uint8 * stream, int length )
{
    size_t inptr = 0;
    static float inbuf[SAMPLES];
    static float outbuf[SAMPLES];
    static float phi = 0.0f;
    static float phi2 = 0.0f;

    int const flength = length / sizeof( float ) / 2;
    assert( flength == SAMPLES );

    Convo *self = static_cast<Convo *>(userData);

    for ( int i = 0; i < 5; i++ ) {
        if ( self->play[i] ) {
            self->play[i].store( false );
            SDL_AudioStreamPut( self->sampleStream, self->buffer[i], self->length[i] );
            SDL_AudioStreamFlush( self->sampleStream );
        }
    }

    int available = SDL_AudioStreamAvailable( self->sampleStream ) / sizeof( float );

    for ( int i = 0; i < flength; ++i ) {
        float sample;
        if ( available ) {
            SDL_AudioStreamGet( self->sampleStream, &sample, sizeof( float ) );
            --available;
        } else {


            phi2 += 2 * M_PI * (1.0f / 44100.0f) * 0.05f;
            if ( phi2 >= (2 * M_PI) ) {
                phi2 -= 2 * M_PI;
            }

            phi += 2 * M_PI * (1.0f / 44100.0f) * ((9000+ std::sin( phi2 ) * 9000) + 30);

            if ( phi >= (2 * M_PI) ) {
                phi -= 2 * M_PI;
            }

            sample = std::sin( phi );
            //sample = 0.0f;
        }
        inbuf[inptr++] = sample;
        //SDL_AudioStreamPut( self->outStream, &sample, sizeof( float ) );
    }

    SDL_AudioStreamGet( self->outStream, inbuf, SAMPLES * sizeof( float ) );
    self->convolver.processBlock( inbuf, outbuf );

    SDL_AudioStreamPut( self->outStream, outbuf, SAMPLES * sizeof( float ) );

    int ava = SDL_AudioStreamAvailable( self->outStream );
    int rc = SDL_AudioStreamGet( self->outStream, stream, length );
}
