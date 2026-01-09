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

/** Path tracing pass.

    This file contains the entry points for all ray tracing programs.
    We import the path tracer utility functions defined in PathTracer.slang.

    The host sets the compile-time constants in StaticParams.slang.
    It also sets the following defines for optional I/O buffers:

    is_valid_<name> is 1 if buffer with this name is bound, 0 otherwise.
*/
import PathTracer;
import Utils.Timing.GpuTimer;
import RenderPasses.Shared.PathTracer.LoadShadingData;
import RenderPasses.Shared.NeuralNetwork.NeuralStructuresCV;

#define ONLY_POS 1


#ifndef MLOD
#define MLOD 1
#endif

cbuffer GeneralData : register(b4)
{
    float3 bbStart;
    float3 bbEnd;
    uint lodSize0;
    uint lodSize1;
    uint lodSize2;
    uint gd_hash_grid_res;
    uint gd_hash_grid_size;
    uint max_bounce_with_cv;
};

ParameterBlock<PathTracerData> gData;

// Outputs (optional)
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputAlbedo;
RWTexture2D<uint> gOutputTime;




RWStructuredBuffer<XInferenceData> gPredBuffer;
RWStructuredBuffer<uint> gPredBufferPixelID;
RWStructuredBuffer<float3> gPredBufferThroughput;


RWStructuredBuffer<OutputTransformData> gPredBufferOutActAddData;

RWStructuredBuffer<LThp> gLThpSelfTrainingBuffer;

RWStructuredBuffer<XInferenceData> gXTrainBuffer;

RWStructuredBuffer<TrainingAdditionalData> gXTrainAdditionalBuffer;

RWStructuredBuffer<int> gNNMapPredBufferKeys;
RWStructuredBuffer<int> gNNMapPredBufferValues;


RWStructuredBuffer<int> gNNMapTrainBufferKeys;
RWStructuredBuffer<int> gNNMapTrainBufferValues;




// RWStructuredBuffer<float3> gYTrainBuffer;

// Static configuration based on which buffers are bound.
#define is_valid(name) (is_valid_##name != 0)

/** ********************* Ray index 0: Scatter ray ************************ */

[shader("miss")] void scatterMiss(inout ScatterRayData rayData
                                  : SV_RayPayload) {
}

    [shader("anyhit")] void scatterAnyHit(inout ScatterRayData rayData
                                          : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs
                                          : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
        IgnoreHit();
#endif
}

[shader("closesthit")] void scatterClosestHit(inout ScatterRayData rayData
                                              : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs
                                              : SV_IntersectionAttributes)
{
    // Store hit information. Note we don't access the materials here.
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    rayData.packedHitInfo = HitInfo(triangleHit).pack();
}

    /************************** Ray index 1: Shadow ray ************************ */

[shader("miss")] 
void shadowMiss(inout ShadowRayData rayData : SV_RayPayload)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")] 
void shadowAnyHit(inout ShadowRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
        IgnoreHit();
#endif
}

/** ******************************** RayGen ******************************** */

/** This is the entry point for the path tracer.

    We generate N paths (= #spp) per pixel, which are traced into the scene.
    The path tracer is written as a for-loop over path segments, where each
    iteration traces a shadow ray for direct illumination and a scatter ray.

    The hit shader for the scatter ray currently generates ray parameters for
    the shadow ray to evaluate direct illumination and generates ray parameters
    for the next scatter ray, which are both returned the raygen shader to be
    traced. This is more efficient than tracing from the hit shader. The max
    recusion depth = 1.
*/

static const float PI = 3.14159265f;

XInferenceData constructPredictData(ShadingData sd, SampleGenerator sg, bool permute = false)
{
    XInferenceData surfaceInfo;
#if 1
    //surfaceInfo.Position = ((sd.posW - bbStart) / (bbEnd - bbStart) * 2 - 1.0);
    surfaceInfo.Position = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
#else
    surfaceInfo.Position = clamp((sd.posW - gScene.camera.data.posW) / 65.0, -1.0, 1.0);

    
#endif
#if !ONLY_POS
    surfaceInfo.ScatteredDir = sd.V;
    surfaceInfo.SurfaceNormal = sd.N;
    surfaceInfo.SurfaceRoughness = 1.0 - exp(-sd.linearRoughness);
    surfaceInfo.DiffuseReflectance = sd.diffuse;
    surfaceInfo.SpecularRefelctance = sd.specular;
#endif

#if 0
    if(permute){
    surfaceInfo.Position += (sampleNext3D(sg)*2-1.0)*0.005;
    surfaceInfo.ScatteredDir += (sampleNext2D(sg)*2-1.0)*0.005;
    surfaceInfo.SurfaceRoughness += (sampleNext1D(sg)*2-1.0)*0.005;
    surfaceInfo.DiffuseReflectance += (sampleNext3D(sg)*2-1.0)*0.005;
    surfaceInfo.SpecularRefelctance += (sampleNext3D(sg)*2-1.0)*0.005;
    }
#endif
    return surfaceInfo;
}


#define EMISSIVE_HACK 0
#define DUMMY_MAPPING 0
#define SMALL_TRAIN_BUFF 1
#define USE_METRIC 0
#define ENABLE_NRC 1

#define RENDER_ONLY_CV 0

#define RENDER_SPEC 0
#define RENDER_DIFF 1
#define RENDER_ONLY_INDIRECT 1
#define TRAIN_DIRECT 0


OutputTransformData constructOutputTransformData(ShadingData sd){

    OutputTransformData res;
    res.normal = sd.N;
    res.viewDir = sd.V;
    res.roughness = sd.linearRoughness;
    res.diffuseAlbedoInvPi = sd.diffuse/PI;
    res.specAlbedo = sd.specular;
    
    
    res.specFactor = 0.0f;
    res.only_cv = 0.0f;
    res.specFactor = 1.0f;

    #if !RENDER_DIFF
    res.diffuseAlbedoInvPi = 0.0;
    #endif


    #if !RENDER_SPEC
    res.specFactor = 0.0f;
    #endif

    #if RENDER_ONLY_CV
    res.only_cv = 1.0f;
    #endif
    
    return res;
}



#define HASH_GRID 0
#ifndef GRADIENT_DENOISING
#define GRADIENT_DENOISING 0
#endif

#define GRADIENT_DENOISING 0

uint32_t fast_hash(const uint3 pos_grid) {
	// While 1 is technically not a good prime for hashing (or a prime at all), it helps memory coherence
	// and is sufficient for our use case of obtaining a uniformly colliding index from high-dimensional
	// coordinates.
	constexpr uint32_t primes[7] = { 25165843, 19349663, 83492791, 25165843, 6291469, 12582917, 3145739 };

	uint32_t result = 0;
    result ^= pos_grid.x * primes[0];
    result ^= pos_grid.y * primes[1];
    result ^= pos_grid.z * primes[2];
	return result;
}

uint mapToHashGrid(float3 pos, uint res, uint size){
    pos *= (res-1);
    int3 gridIndex = int3(floor(pos));
    uint index = fast_hash(gridIndex);
    return index % size;
}

uint getIDNN(float3 pos, uint lodSIZE){
#if HASH_GRID
    mapToHashGrid(pos, lodSIZE*8, lodSIZE*lodSIZE*lodSIZE);
#else
    pos *= lodSIZE;
    int3 gridIndex = int3(floor(pos));
    gridIndex = clamp(gridIndex, 0, lodSIZE-1);
    return gridIndex.x + gridIndex.y * lodSIZE + gridIndex.z * lodSIZE* lodSIZE;
#endif
}

[shader("raygeneration")] 
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    const uint2 launchIndexDef = launchIndex;
    uint2 launchDim = DispatchRaysDimensions().xy;
    bool relyOnNRCForLastVertex;
    uint2 predictDim = launchDim;
    uint2 trainDim = launchDim/uint2(8, 4);

    
    float3 trainOffset = 0;
#if CONTINUE_RAY
    const uint pathIDTraining = launchIndex.y * launchDim.x + launchIndex.x;
    launchDim *= uint2(8, 4);

    predictDim = launchDim;
    trainDim = launchDim/uint2(8, 4);

    uint frameSeed2 = gData.params.useFixedSeed ? 0 : gData.params.frameCount;
    SampleGenerator sg0 = SampleGenerator.create(launchIndex, frameSeed2 * 234);
    // FIX IT!!! MULT * 0
    launchIndex.xy = launchIndex.xy * uint2(8, 4) + sampleNext2D(sg0) * uint2(7, 3);

    float offsetSize = 1.0f/lodSize1*0.2;
    
#if HASH_GRID
    offsetSize = 1.0f/(lodSize1*2)*0.2;    
#endif

    trainOffset = sampleNext3D(sg0)*(2*offsetSize)-offsetSize;
    relyOnNRCForLastVertex = sampleNext1D(sg0) > 0.03;
#if !ENABLE_NRC
    relyOnNRCForLastVertex = false;
#endif
#endif
    const uint predictLOD0Offset = predictDim.x*predictDim.y;
    const uint predictLOD0WithNRCOffset = predictDim.x*predictDim.y+trainDim.x*trainDim.y;

    relyOnNRCForLastVertex = false;

//    trainOffset = 0;
   // relyOnNRCForLastVertex = false;
    
    logSetPixel(launchIndex);
    printSetPixel(launchIndex);

    GpuTimer timer;
    if (is_valid(gOutputTime))
        timer.start();

    float3 outColor = float3(0, 0, 0);
    float3 outAlbedo = float3(0, 0, 0);
    float outAlpha = 0.f;

    // NRC
    float4 outPredictionL = float4(0);
    float4 outPredictionThp = float4(0);

    bool set_pred = false;
    XInferenceData prediction = {};

    uint depth = 0;

    Ray camRay = gScene.camera.computeRayPinhole(launchIndex, launchDim);

    HitInfo hit;
    ShadingData sd;
    
    OutputTransformData outputTransformData;
    outputTransformData.normal = -1;
    

    uint32_t useNeuralNumIter = 1;
    uint32_t GTNumIter = gData.params.maxBounces;
    
#if !ENABLE_NRC
    useNeuralNumIter = GTNumIter;
#endif
    PathData path = {};
    bool bIsSurfaceShadingDataLoaded = false;
    if (loadShadingData(launchIndex, launchDim, gScene.camera, sd, hit))
    {
        bIsSurfaceShadingDataLoaded = true;
        // const uint2 q_size = uint2(8, 4);
        // const uint2 q_indices = uint2(float2(launchDim) / q_size + uint2(1.0));
        // const uint2 quadrant = launchIndex * q_size / launchDim;
        
        // 
        
        // SampleGenerator sg = SampleGenerator.create(quadrant, frameSeed);

        // const uint2 chosenIdx = q_indices * quadrant + uint2(q_indices * sampleNext2D(sg));
        // print(chosenIdx);
        // //if ((launchIndex.x == chosenIdx.x) && (launchIndex.y == chosenIdx.y))
        // //    continue_ray = true;
        uint frameSeed = gData.params.useFixedSeed ? 0 : gData.params.frameCount;
        //SampleGenerator sg2 = SampleGenerator.create(launchIndex/uint2(8, 4), frameSeed);
        //if(all(uint2(sampleNext2D(sg2)*(int2(8, 4)-1)) == launchIndex%uint2(8,4)))
        //    continue_ray = true;   

        // Pixel represents a primary hit. Compute its contribution.

        // Compute ray origin for new rays spawned from the G-buffer.
        float3 rayOrigin = sd.computeNewRayOrigin();

#if USE_METRIC
        const float3 divVec = rayOrigin - path.origin;
        const float rLen_sq = dot(divVec, divVec);
        const float cosTheta = getCos(-camRay.dir, sd.N);
        const float a0 = rLen_sq / (4.0 * 3.14159265 * cosTheta);
        float aN = 0.0;
#else
        const float a0 = 0.0;
        float aN = 0.0;
#endif
        
        // Loop over samples in pixel.
        // Setup path data.
        
        path.origin = rayOrigin;
        path.thp = float3(1.f);
        path.L = 0.0;
        path.hit = hit;

        // Create sample generator.  
        path.sg = SampleGenerator.create(launchIndex, frameSeed * kSamplesPerPixel);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/res    tore the state from memory if it becomes a problem.
        for (uint i = 0; i < gData.params.prngDimension; i++)
            sampleNext1D(path.sg);

        // TODO: Use (kRayFootprintMode != TexLODMode::Mip0) when slang is fixed.
        if (!(kRayFootprintMode == TexLODMode::Mip0))
        {
            // Create the ray footprint data for TexLOD.
            path.rayFootprint = RayFootprint.create(hit.getTriangleHit(), launchIndex, launchDim, rayOrigin, gScene.camera.getPosition(), sd.faceN, sd.N, gData.params.screenSpacePixelSpreadAngle, sd.linearRoughness, path.isSpecular());
        }

#if USE_METRIC
        const float C = 0.1;
        bool refueled = false;
#endif

        bool bKeepTracing = true;


        uint l = 0;
        

        #if RENDER_SPEC
        l |= (uint)LobeType::Specular;
        #endif 

        #if RENDER_DIFF
        l |= (uint)LobeType::Diffuse;
        #endif


        sd.setActiveLobes(l);

        // Trace the path.
#if CONTINUE_RAY
        do
        {
            //sd.setActiveLobes((uint)LobeType::Diffuse);
            //sd.linearRoughness = 1.0f;

            //sd.linearRoughness = 0.6f;
            sd.setActiveLobes(l);

            outputTransformData = constructOutputTransformData(sd);


            prediction = constructPredictData(sd, path.sg, true);
            outPredictionThp = float4(path.thp, 0);
            float3 pathLPrev = path.L;
            float3 pathThpPrev = path.thp;
            bKeepTracing = tracePathSimple(gData, sd, path, depth, aN, rayOrigin, !TRAIN_DIRECT && true);

            outputTransformData.incidentDir = path.dir;
            outputTransformData.pdf = path.pdf;
            
            outPredictionL = float4(path.L, 1.0);
            
            

            if(depth <3){
                float addU = sampleNext1D(path.sg);
                bool add_train = (SMALL_TRAIN_BUFF) ? (addU < 1.0) : true;
                
                if (add_train)
                {
                    uint tid = gXTrainBuffer.IncrementCounter();
                    
                    bool keepAddingToTrain = true;
                    #if 0
                    if(tid != 0){
                        gXTrainBuffer.DecrementCounter();
                        keepAddingToTrain = false;
                    }
                    #endif

                    if(keepAddingToTrain){
                    gXTrainBuffer[tid] = prediction;

                    // Add index of NN to the buffer 
                    int pow = 1;

                    int indexOffset = 0;
                    float3 bboxPos = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
                    

                    #if GRADIENT_DENOISING
                    uint hashid = mapToHashGrid(bboxPos, gd_hash_grid_res, gd_hash_grid_size);
                    gDenoisingMapValue[tid] = tid;
                    gDenoisingMapKey[tid] = hashid;
                    gDenoisingPositions[tid].x = bboxPos.x;
                    gDenoisingPositions[tid].y = bboxPos.y;
                    gDenoisingPositions[tid].z = bboxPos.z;
                    #endif


                    uint lodSizes[3] = {lodSize0, lodSize1, lodSize2};
                    
                    for (int LOD = 0; LOD < MLOD; LOD++)
                    {
                        pow = lodSizes[LOD];
                        float3 scaled = (bboxPos+trainOffset*float(LOD))*pow;
                        
                        int3 gridIndex = int3(floor(scaled));
                        gridIndex = clamp(gridIndex, 0, pow-1);
                        int idNN = indexOffset + getIDNN(saturate(scaled/pow), pow);
                        if(MLOD != 0){
                            if(LOD == 0){
                                gNNMapTrainBufferKeys[tid * MLOD + LOD] = 0; 
                            }
                            else{
                                #if DUMMY_MAPPING
                                gNNMapTrainBufferKeys[tid * MLOD + LOD] = 1+8+64; 
                                #else
                                gNNMapTrainBufferKeys[tid * MLOD + LOD] = idNN; 
                                #endif
                            }
                        }
                        gNNMapTrainBufferValues[tid * MLOD + LOD] = tid; 
                        
                        
                        indexOffset += pow * pow * pow;
                        pow = pow * 2;
                    }

                    TrainingAdditionalData d;
                    d.lightPathID = pathIDTraining;
                    if(TRAIN_DIRECT)
                    {
                        d.dirL = 0;    //clamp((path.L-pathLPrev)/pathThpPrev, 0, 1000);
                    }
                    else
                    {
                        d.dirL = 0;
                    }

                    d.att = clamp(path.L, 0, 100000); // substract direct lighting
                    
                    if(TRAIN_DIRECT)
                    {
                        d.att = clamp(pathLPrev, 0, 1000);
                    }
                    
                    d.thp = path.thp; 
                    d.incidentDir = path.dir;
                    d.pdf = path.pdf;
                    gXTrainAdditionalBuffer[tid] = d;
                    }
                }
            }

            if (!relyOnNRCForLastVertex && depth == GTNumIter-1){
                break;
            }

            if (depth == useNeuralNumIter && relyOnNRCForLastVertex)
            {
                set_pred = true;
                break;
            }

            depth += 1;
        } while (bKeepTracing);
#else


        sd.setActiveLobes(l);

        float4 prevOutPredictionL = outPredictionL;
        do
        {
            //sd.setActiveLobes((uint)LobeType::Diffuse);
            //sd.linearRoughness = 0.6f;  
                        
            prediction = constructPredictData(sd, path.sg);
            outputTransformData = constructOutputTransformData(sd);
            outPredictionThp = float4(path.thp, 0);
            bKeepTracing = tracePathSimple(gData, sd, path, depth, aN, rayOrigin,!TRAIN_DIRECT && RENDER_ONLY_INDIRECT);

            outputTransformData.incidentDir = path.dir;
            outputTransformData.pdf = path.pdf;

            #if RENDER_ONLY_INDIRECT
            if(depth == 0){
                path.L = 0;
            }
            #endif
            outPredictionL = float4(path.L, 1.0);    
            
            bool bRequestedCV = false;
            if (ENABLE_NRC && depth < GTNumIter-1 && depth < max_bounce_with_cv)
            {   
                bRequestedCV = true;
                uint tid = gPredBuffer.IncrementCounter();
                print(tid);
                gPredBuffer[tid] = prediction;
                gPredBufferOutActAddData[tid] = outputTransformData;
                gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;
                gPredBufferThroughput[tid] = outPredictionThp.xyz;
                
#if 0
                // Add index of NN to the buffer 
                int pow = 2;
                int indexOffset = 1;
                //float3 bboxPos = (sd.posW - bbStart) / (bbEnd - bbStart);
                //float3 bboxPos = (prediction.Position + 1) / 2 / (1.0f + 0.0001f);
                float3 bboxPos = prediction.Position;
                
                int zeroLODoffset = predictLOD0WithNRCOffset;
                print(bboxPos);
                uint lodSizes[3] = {lodSize0, lodSize1, lodSize2};

                for (int LOD = 1 ; LOD < MLOD; LOD++)
                {
                    pow = lodSizes[LOD];
                    float3 scaled = bboxPos*pow;
                    int3 gridIndex = int3(floor(scaled));
                    gridIndex = clamp(gridIndex, 0, pow-1);
                    //int idNN = indexOffset + gridIndex.x + gridIndex.y * pow + gridIndex.z * pow * pow;
                    int idNN = indexOffset + getIDNN(saturate(scaled/pow), pow);

                    #if DUMMY_MAPPING
                    gNNMapPredBufferKeys[pathIDGlobal * (MLOD-1) + LOD - 1] = 1+8+64;
                    #else
                    gNNMapPredBufferKeys[pathIDGlobal * (MLOD-1) + LOD - 1] = idNN;
                    #endif
                    if(LOD == 1)
                    print(idNN);
                    gNNMapPredBufferValues[pathIDGlobal * (MLOD-1) + LOD - 1] = pathIDGlobal; 
                    indexOffset += pow * pow * pow;
                    pow = pow * 2;
                }
#endif
            }

            #if RENDER_ONLY_CV
            #if TRAIN_DIRECT
            outPredictionL = 0;
            #endif
            // if(depth == 1)
            break;
            
            #endif
            
            
            if(depth == GTNumIter-1){
                break;
            }

            depth++;
            prevOutPredictionL = outPredictionL;
            
        } while (bKeepTracing);
#endif

        logPathLength(path.length);

        // Accumulate after clamping.
        // Note the comparison is written so that NaNs propagate (unless the compiler rewrites it).
        // TODO: Check the generated code that this is the case.
        outColor += gData.params.clampSamples && path.L > gData.params.clampThreshold ? gData.params.clampThreshold : path.L;

        // We're done accumulating over all samples.
        const float invSpp = 1.f / kSamplesPerPixel;
        outColor *= invSpp;
        outAlbedo = sd.diffuse + sd.specular;
        outAlpha = 1.f;
    }
    else
    {   
        outPredictionL.xyz = evalBackground(-sd.V);

        outPredictionL = 1.0;
        outPredictionL.w = 1.0f;
        //outPredictionL = 1000.0f;
        // Background pixel.
        outPredictionL = 10000;
        outColor = evalBackground(-sd.V);
        outColor = 1000.0;


        outAlbedo = outColor.rgb;
        outAlpha = kForceAlphaOne ? 1.f : 0.f;
    }


    assert(!any(isnan(outColor)));

    // Write outputs.
    // These are all optional so using compile-time checks to decide which ones to write.
    
    if (is_valid(gOutputAlbedo))
        gOutputAlbedo[launchIndex] = float4(outAlbedo, 1);

    // Write time.
    if (is_valid(gOutputTime))
        gOutputTime[launchIndex] = timer.getElapsed();

    const uint pathIDGlobal = launchIndex.y * launchDim.x + launchIndex.x;

#if !ENABLE_NRC
    outPredictionThp = 0;
#endif
    

    XInferenceData dummyPred;
    dummyPred.Position = -1;
    
    #if !ONLY_POS
    dummyPred.ScatteredDir = -2;
    dummyPred.SurfaceNormal = -3;
    dummyPred.SurfaceRoughness = -4;
    dummyPred.DiffuseReflectance = -5;
    dummyPred.SpecularRefelctance = -6;
    #endif

    // NRC output
#if CONTINUE_RAY
    
    LThp data;
    data.L = clamp(outPredictionL.xyz, 0, 100000);
    data.thp = outPredictionThp.xyz;

    if (!relyOnNRCForLastVertex || !set_pred || !bIsSurfaceShadingDataLoaded)
    {
        data.thp = 0;
    }
    
    const uint pathIDTrainingAfterPredict = pathIDTraining+predictLOD0Offset;
    gLThpSelfTrainingBuffer[pathIDTraining] = data;

    launchDim /= (8, 4);
    if (relyOnNRCForLastVertex && set_pred){

        
        
        gPredBuffer[pathIDTrainingAfterPredict] = prediction;
        gPredBufferOutActAddData[pathIDTrainingAfterPredict] = outputTransformData;

        int pow = 2;
        int indexOffset = 1;
        float3 bboxPos = prediction.Position;
        uint lodSizes[3] = {lodSize0, lodSize1, lodSize2};
        int zeroLODoffset = predictLOD0WithNRCOffset;
        for (int LOD = 1 ; LOD < MLOD; LOD++)
        {
            pow = lodSizes[LOD];
            float3 scaled = bboxPos*pow;
            int3 gridIndex = int3(floor(scaled));
            gridIndex = clamp(gridIndex, 0, pow-1);

            float3 innerOffset = scaled - gridIndex;
            int idNN = indexOffset + getIDNN(saturate(scaled/pow), pow);

            gNNMapPredBufferValues[pathIDTrainingAfterPredict * (MLOD-1) + LOD - 1] = pathIDTrainingAfterPredict; 

            #if DUMMY_MAPPING
            gNNMapPredBufferKeys[pathIDTrainingAfterPredict * (MLOD-1) + LOD - 1] = 1+8+64;                         
            #else
            gNNMapPredBufferKeys[pathIDTrainingAfterPredict * (MLOD-1) + LOD - 1] = idNN; 
            #endif
            indexOffset += pow * pow * pow;
            pow = pow * 2;
        }
    }
    else{
        // gPredBufferOutActAddData[pathIDTrainingAfterPredict] = outputTransformData;
        // gPredBuffer[pathIDTrainingAfterPredict] = dummyPred;
        // int zeroLODoffset = predictLOD0WithNRCOffset;
        // for (int LOD = 1 ; LOD < MLOD; LOD++)
        // {
        //     gNNMapPredBufferKeys[pathIDTrainingAfterPredict * (MLOD-1) + LOD - 1] = 1+8+64;
        // }
    }

#else
    if (is_valid(gOutputColor)){
        gOutputColor[launchIndex] = outPredictionL;
    }
#endif
    //gOutputColor[launchIndex] = 0;
}
