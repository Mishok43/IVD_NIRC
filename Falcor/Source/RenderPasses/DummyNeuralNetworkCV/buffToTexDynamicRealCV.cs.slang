/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

#include "Utils/NVAPI.slangh" // We need this for 64-bit atomics.

StructuredBuffer<float3> gPredictedColor;
StructuredBuffer<uint> gPredBufferPixelID;
StructuredBuffer<float3> gPredBufferThroughput;

RWTexture2D<float4> gPredictedRadianceMerged;

#define RAD_SCALE 1.0f
#define RAD_SCALE_INV 1.0/RAD_SCALE
#define SQRT 0

bool IsNaN(float x)
{
    return !(x < 0.f || x > 0.f || x == 0.f);
}

cbuffer PerFrameCB
{
    uint gElems;            // Number of elements in input intermediate buffer.
};

[numthreads(1024, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 texDim;
    uint levels;
    gPredictedRadianceMerged.GetDimensions(texDim.x, texDim.y);
    uint id = dispatchThreadID.x;
    if(id >= gElems)
        return;

    float3 pred = gPredictedColor[id];
    
    const float3 throughtput = gPredBufferThroughput[id];

    uint pixelIDFlat = gPredBufferPixelID[id];
    uint2 pixelID;
    pixelID.y = pixelIDFlat/texDim.x;
    pixelID.x = pixelIDFlat-pixelID.y*texDim.x;



    #if SQRT
    pred*=pred;
    #endif

    
    float4 c;
    c.xyz = pred*(throughtput*RAD_SCALE_INV);
    c.w = 0.0f;

    //c = 0.1;
    
    if(true){
        #if 1
        NvInterlockedAddFp16x4(gPredictedRadianceMerged, pixelID, c);
        #else
        gPredictedRadianceMerged[pixelID] = c;
        #endif
    }
    
}
