#include "Convo.h"
#include <iostream>
#include "sdl\sdl.h"
#undef main // what were you thinking???


int main()
{
    Convo convo;

    if ( !convo.Init() ) {
        return EXIT_FAILURE;
    }

    convo.Run();

    return 0;
}
