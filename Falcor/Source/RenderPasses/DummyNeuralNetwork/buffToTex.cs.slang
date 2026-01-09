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


StructuredBuffer<float3> gPredictedColor;

Texture2D<float3> gEstimatedRadiance;
Texture2D<float3> gPredictionThp;

RWTexture2D<float4> colorOutput;

#define RAD_SCALE 1.0f
#define RAD_SCALE_INV 1.0/RAD_SCALE
#define SQRT 0

bool IsNaN(float x)
{
    return !(x < 0.f || x > 0.f || x == 0.f);
}

[numthreads(32, 32, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 texDim;
    uint levels;
    colorOutput.GetDimensions(texDim.x, texDim.y);
    uint id = dispatchThreadID.y*texDim.x+dispatchThreadID.x;
    float3 pred = gPredictedColor[id];
    #if SQRT
    pred*=pred;
    #endif

    float3 c = gEstimatedRadiance[dispatchThreadID.xy];
    if(!IsNaN(pred.x))
        c += gPredictionThp[dispatchThreadID.xy]*(pred*RAD_SCALE_INV);
    
    colorOutput[dispatchThreadID.xy] = float4(c.r, c.g, c.b, 1.0);
}
