#pragma once

#include <cassert>
#include <cstring>
#include "pffft/pffft.h"



// Performs blockwise convolution using a partitioned overlap-save algorithm.
// see http://pcfarina.eng.unipr.it/Public/Papers/164-Mohonk2001.PDF
class Convolve
{
public:
    Convolve() = default;
    ~Convolve();
    void setKernel( float *kernel, size_t length );
    void processBlock( const float *in, float *out );

private:
    inline static size_t blockIndex( size_t i )
    {
        return blockSize * 2 * i;
    }

    inline static float *allocate( size_t size )
    {
        float *mem = static_cast<float *>(pffft_aligned_malloc( size ));
        assert( mem );
        std::memset( mem, 0, size );
        return mem;
    }

private:
    // Considerations for selecting block size:
    //   - Latency:
    //     Since DFT is performed blockwise, the node needs to buffer this many samples
    //     before they can be DFT'd, convolved and output. This implies that the output
    //     from this node is delayed by the number of samples specified,
    //     i.e. blockSize / sampleRate seconds.
    //   - Frequency resolution:
    //     The DFT outputs as many frequency bins as there are samples in time domain
    //     with each bin spanning sampleRate / blockSize Hertz. Generally, a higher
    //     resolution is preferred.
    //   - Performance:
    //     Transformation and convolution of very small buffers (fewer than about 64 
    //     samples) is inefficient to a degree that time-domain convolution might well 
    //     be faster. Big buffers (more than about 64K samples) cause the DFT to take 
    //     longer. The sweet spot seems to be somewhere between 1024 and 8192.
    //   - PFFFT requirement:
    //     The library currently used to compute forward and inverse DFT requires a block
    //     size of at least 32 samples.
    //   - Needs to be a power of 2.
    static constexpr size_t blockSize = 8192; //1 << 15;
    static_assert((blockSize & (blockSize - 1)) == 0, "blockSize needs to be a power of 2");
    static_assert(blockSize >= 32, "PFFT requires a block size of >= 32 samples");

    // Frequency-domain blocks of the filter.
    // The blocks are pointers to arrays that are dynamically allocated by PFFFT in order
    // to fulfill alignment requirements imposed by SIMD instructions.
    //std::vector<float *> filterBlocks;

    //float *inputBuffer;     // source buffer for DFT
    //float *inputFft;        // destination buffer for DFT
    //float *outputBuffer;    // buffer for convolved and inverse-transformed samples

    float *filterBlocks;    // contains the DFT'd filter
    float *inBlocks;        // contains input samlpes
    float *workBlock;       // general-purpose block used for DFT
    float *tail;

    size_t inBlock = 0;
    size_t inSample = 0;
    size_t outSample = 0;
    size_t blockCount;
    PFFFT_Setup *setup = nullptr;
};
