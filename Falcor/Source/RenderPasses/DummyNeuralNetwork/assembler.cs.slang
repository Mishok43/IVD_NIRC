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

import RenderPasses.Shared.NeuralNetwork.NeuralStructures;


cbuffer PerFrameCB : register(b0)
{
    uint32_t predictDataOffset;
    float outScale;
};


StructuredBuffer<float3> gPredictedRadianceSTYBuffer;
StructuredBuffer<LThp> gLThpSelfTrainingBuffer;
StructuredBuffer<TrainingAdditionalData> gXTrainAdditionalBuffer;

RWStructuredBuffer<YData> gYTrainBuffer;

#ifndef DIRECT_CACHE
#define DIRECT_CACHE 0
#endif

#define RAD_SCALE 1.0f
#define RAD_SCALE_INV 1.0/RAD_SCALE
[numthreads(128, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint launchIndex = dispatchThreadID.x;

    TrainingAdditionalData xadd = gXTrainAdditionalBuffer[launchIndex];
    
    LThp xlthp = gLThpSelfTrainingBuffer[xadd.lightPathID];
    
    float3 surfaceRad = clamp(xlthp.L- xadd.att, 0.0, 1000000);
    float3 pathRadiance = (surfaceRad)/xadd.thp;
    
    YData ydata;
    ydata.radiance =  clamp(pathRadiance, 0.0, 1000000)*outScale;
    ydata.weight_thp = xadd.thp;
    ydata.weight_add = xadd.att;
    gYTrainBuffer[launchIndex] = ydata;
}
