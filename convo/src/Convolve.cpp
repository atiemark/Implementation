#include "Convolve.h"
#include "pffft/pffft.h"
#include <cstring>
#include <fstream>


Convolve::~Convolve()
{
    if ( setup ) {
        pffft_destroy_setup( setup );
    }

    pffft_aligned_free( filterBlocks );
    pffft_aligned_free( inBlocks );
    pffft_aligned_free( workBlock );
    pffft_aligned_free( tail );
}

void Convolve::setKernel( float *kernel, size_t length )
{   
    if ( !setup ) {
        setup = pffft_new_setup( blockSize * 2, PFFFT_REAL );
    }

    blockCount = (length - 1) / blockSize + 1;

    filterBlocks = allocate( blockCount * blockSize * sizeof( float ) * 2 );
    inBlocks = allocate( blockCount * blockSize * sizeof( float ) * 2 );
    workBlock = allocate( blockSize * sizeof( float ) * 2 );
    tail = allocate( blockSize * sizeof( float ) );

    for ( size_t block = 0; block < blockCount; block++ ) {
        std::memset( workBlock, 0, blockSize * sizeof( float ) * 2 );

        for ( size_t sample = 0; sample < blockSize && (block * blockSize + sample) < length; sample++ ) {
            //workBlock[sample] = filterBuffer->getSample( block * blockSize + sample );
            workBlock[sample] = kernel[block * blockSize + sample];
        }

        pffft_transform( setup, workBlock, &filterBlocks[blockIndex( block )], nullptr, PFFFT_FORWARD );
    }

    outSample = inSample = inBlock = 0;
    std::memset( workBlock, 0, blockSize * sizeof( float ) * 2 );
}

void Convolve::processBlock( const float* input, float* output )
{
    assert( setup );
    assert( filterBlocks );
    assert( inBlocks );
    assert( workBlock );

    //inBlocks[blockIndex( inBlock ) + inSample] = in->pull();
    //++inSample;

    std::memset( workBlock, 0, sizeof( float ) * blockSize * 2 );
    std::memcpy( workBlock, input, blockSize * sizeof( float ) );

    //if ( inSample >= blockSize - 1 ) {
    pffft_transform( setup, workBlock, &inBlocks[blockIndex( inBlock )], nullptr, PFFFT_FORWARD );
    std::memset( workBlock, 0, blockSize * sizeof( float ) * 2 );

    for ( size_t convInBlock = 0, convFilterBlock = inBlock; convInBlock < blockCount; ++convInBlock, convFilterBlock = (convFilterBlock + blockCount - 1) % blockCount ) {
        //size_t inpos = (inBlock - 1 + blockCount) % blockCount;
        pffft_zconvolve_accumulate( setup, &inBlocks[blockIndex( convInBlock )], &filterBlocks[blockIndex( convFilterBlock )], workBlock, 1.0f );
    }
    pffft_transform( setup, workBlock, workBlock, nullptr, PFFFT_BACKWARD );

    float const scale = 1.0f / (blockSize * 2.0f);
    for ( size_t i = 0; i < blockSize; ++i ) {
        workBlock[i] += tail[i];
        workBlock[i] *= scale;
    }

    std::memcpy( tail, &workBlock[blockSize], blockSize * sizeof( float ) );

    outSample = inSample = 0;
    ++inBlock;
    if ( inBlock >= blockCount ) {
        inBlock = 0;
    }

    std::memcpy( output, workBlock, blockSize * sizeof( float ) );
//}

//return workBlock[outSample++];
}
