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
import RenderPasses.Shared.NeuralNetwork.NeuralStructures;
import Utils.Math.HalfUtils;

#define ONLY_POS 1


#ifndef MLOD
#define MLOD 1
#endif


#define NUM_GUIDING_VERTICES 1
#define EMISSIVE_HACK 0
#define DUMMY_MAPPING 0
#define SMALL_TRAIN_BUFF 1
#define USE_METRIC 0
#define OUTPUT_TRANSFORM 0  
#define DISABLE_NORMALS 0
#define RENDER_ONLY_CV 0

#define NORMALS_HACK 1

#define RENDER_SPEC 1
#define RENDER_DIFF 1


#ifndef TRAIN_DIRECT 
#define TRAIN_DIRECT 0
#endif


#ifndef USE_GUIDANCE_OF_ENV_MAP
#define USE_GUIDANCE_OF_ENV_MAP 0
#endif


#ifndef SUPPORT_NEE
#define SUPPORT_NEE 1
#endif

#ifndef DEBUG_RENDER
#define DEBUG_RENDER 0
#endif

#ifndef RESIDUAL_ERROR_ESTIMATION
#define RESIDUAL_ERROR_ESTIMATION 0
#endif

#ifndef DEBUG_VIEW
#define DEBUG_VIEW 1
#endif

#ifndef BIASED
#define BIASED 0
#endif

#ifndef ENABLE_NRC
#define ENABLE_NRC 1
#endif

#ifndef NUM_NRC_SAMPLES
#define NUM_NRC_SAMPLES 3
#endif

#ifndef RENDER_ONLY_INDIRECT
#define RENDER_ONLY_INDIRECT 1
#endif


#ifndef INCIDENT_NRC
#define INCIDENT_NRC 0
#endif


#define SPHERICAL_HASH_GRID 0

#ifndef Q_LEARNING
#define Q_LEARNING 0
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
    int2 debug_pixel_id;
    int2 original_dim;

        uint expected_value_num_samples;
    uint num_training_id;
};

ParameterBlock<PathTracerData> gData;


// Outputs (optional)
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputAlbedo;
RWTexture2D<uint> gOutputTime;


#ifndef VARIANCE_LOSS_MODE
#define VARIANCE_LOSS_MODE 0
#endif

#ifndef NUM_VARIANCE_LOSS_SAMPLES
#define NUM_VARIANCE_LOSS_SAMPLES 1
#endif


#ifndef ESTIMATOR_NETWORK
#define ESTIMATOR_NETWORK 0
#endif


RWTexture2D<float4> gExpectedValueTmp;
RWStructuredBuffer<uint> gLossVarianceScatterID;
RWStructuredBuffer<float3> gLossVarianceThroughput;


RWStructuredBuffer<uint> gPredBuffer0;
RWStructuredBuffer<float3> gPredBuffer1;
RWStructuredBuffer<uint> gPredBuffer2;


RWStructuredBuffer<uint> gXTrainResErrorID;
RWStructuredBuffer<ResidualErrorData> gResErrorID;
RWStructuredBuffer<uint> gPredBufferMapping1;

RWStructuredBuffer<uint> gXTrainBuffer0;
RWStructuredBuffer<float3> gXTrainBuffer1;
RWStructuredBuffer<uint> gXTrainBuffer2;

RWStructuredBuffer<uint> gEstimatorNNXTrainBuffer0;
RWStructuredBuffer<float3> gEstimatorNNXTrainBuffer1;
RWStructuredBuffer<uint> gEstimatorNNXTrainBuffer2;

RWStructuredBuffer<TrainingAdditionalData> gEstimatorNNXTrainAdditionalBuffer;

RWStructuredBuffer<uint> gPredBufferPixelID;
RWStructuredBuffer<float3> gPredBufferThroughput;
RWStructuredBuffer<OutputTransformData> gPredBufferOutActAddData;
RWStructuredBuffer<LThp> gLThpSelfTrainingBuffer;
RWStructuredBuffer<TrainingAdditionalData> gXTrainAdditionalBuffer;

RWStructuredBuffer<int> gNNMapTrainBufferKeys;
RWStructuredBuffer<int> gNNMapTrainBufferValues;

RWStructuredBuffer<int> gNNMapPredBufferKeys;
RWStructuredBuffer<int> gNNMapPredBufferValues;


uint addToPredBufferWithoutMapping(XInferenceData data){
    uint id = gPredBuffer0.IncrementCounter();
    gPredBuffer0[id] = (f32tof16(data.Direction.x) << 16) | f32tof16(data.Direction.y);
    gPredBuffer1[id] = data.Position;
    gPredBuffer2[id] = (f32tof16(data.SurfaceNormal.x) << 16) | f32tof16(data.SurfaceNormal.y);

    return id;
}

uint addToPredBuffer(XInferenceData data, uint posID){
    uint id = gPredBuffer0.IncrementCounter();
    gPredBuffer0[id] = (f32tof16(data.Direction.x) << 16) | f32tof16(data.Direction.y);
    //gPredBuffer2[id] = data.SurfaceNormal;
    gPredBuffer2[id] = (f32tof16(data.SurfaceNormal.x) << 16) | f32tof16(data.SurfaceNormal.y);
    gPredBufferMapping1[id] = posID;

    return id;
}

uint addToXTrainBuffer(XInferenceData data){
    uint id = gXTrainBuffer0.IncrementCounter();
    gXTrainBuffer0[id] = (f32tof16(data.Direction.x) << 16) | f32tof16(data.Direction.y);
    gXTrainBuffer1[id] = data.Position;
    gXTrainBuffer2[id] = (f32tof16(data.SurfaceNormal.x) << 16) | f32tof16(data.SurfaceNormal.y);
    return id;
}

uint addToEstimatorNNXTrainBuffer(XInferenceData data){
    uint id = gEstimatorNNXTrainBuffer0.IncrementCounter();
    gEstimatorNNXTrainBuffer0[id] = (f32tof16(data.Direction.x) << 16) | f32tof16(data.Direction.y);
    gEstimatorNNXTrainBuffer1[id] = data.Position;
    gEstimatorNNXTrainBuffer2[id] = (f32tof16(data.SurfaceNormal.x) << 16) | f32tof16(data.SurfaceNormal.y);
    return id;
}



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

XInferenceData constructPredictData(ShadingData sd, bool permute = false)
{
    XInferenceData surfaceInfo;

    surfaceInfo.Position = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
    surfaceInfo.SurfaceNormal =  ndir_to_oct_unorm(sd.N);
    surfaceInfo.Direction = ndir_to_oct_snorm(sd.V);

#if !ONLY_POS
    surfaceInfo.ScatteredDir = sd.V;
    surfaceInfo.SurfaceRoughness = 1.0 - exp(-sd.linearRoughness);
    surfaceInfo.DiffuseReflectance = sd.diffuse;
    surfaceInfo.SpecularRefelctance = sd.specular;
#endif

    return surfaceInfo;
}

XInferenceData encodeData(ShadingData sd, float3 pathDir, inout float4 thp){
    XInferenceData prediction;
    #if INCIDENT_NRC                
    prediction.Position = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
                            
    #if SPHERICAL_HASH_GRID
    prediction.Direction.xy = saturate(ndir_to_oct_unorm(pathDir));
    prediction.Direction.z = 1.0;
    #else
    prediction.Direction = ndir_to_oct_snorm(pathDir);
    #endif

    prediction.SurfaceNormal = ndir_to_oct_unorm(sd.N);
    #else
    prediction = constructPredictData(sd);
    thp.xyz *= sd.diffuse+sd.specular;
    #endif

    return prediction;
}


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

#ifndef HEMISPHERICAL_RENDER
#define HEMISPHERICAL_RENDER 0
#endif

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
    SampleGenerator sg0 = SampleGenerator.create(launchIndex, frameSeed2 * 234+23423*CONTINUE_RAY);
    // FIX IT!!! MULT * 0
    launchIndex.xy = launchIndex.xy * uint2(8, 4) + sampleNext2D(sg0) * uint2(7, 3);

    float offsetSize = 1.0f/lodSize1*0.2;
    
#if HASH_GRID
    offsetSize = 1.0f/(lodSize1*2)*0.2;    
#endif

    //trainOffset = sampleNext3D(sg0)*(2*offsetSize)-offsetSize;
    
    relyOnNRCForLastVertex = sampleNext1D(sg0) > 0.03;
#if !ENABLE_NRC
    relyOnNRCForLastVertex = false;
#endif
#endif
    const uint predictLOD0Offset = predictDim.x*predictDim.y;
    const uint predictLOD0WithNRCOffset = predictDim.x*predictDim.y+trainDim.x*trainDim.y;

#if !Q_LEARNING
    relyOnNRCForLastVertex = false;
#endif
    
//    trainOffset = 0;
   // relyOnNRCForLastVertex = false;
    printSetPixel(launchIndex);
    logSetPixel(launchIndex);
   // printSetPixel(launchIndex);

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
    
    #if OUTPUT_TRANSFORM
    OutputTransformData outputTransformData;
    outputTransformData.normal = -1;
    #endif
    int residualErrorID = -1;
    
    uint32_t useNeuralNumIter = 1;
    uint32_t GTNumIter = gData.params.maxBounces;
    GTNumIter = max_bounce_with_cv;
    if(Q_LEARNING && ENABLE_NRC && relyOnNRCForLastVertex){
        GTNumIter -= 1;
    }
    
#if !ENABLE_NRC
    useNeuralNumIter = GTNumIter;
#endif
    PathData path = {};
    bool bIsSurfaceShadingDataLoaded = false;

    uint2 surfacePixelID = launchIndex;

#if HEMISPHERICAL_RENDER
    surfacePixelID = debug_pixel_id;
#endif

    if (loadShadingData(surfacePixelID, original_dim, gScene.camera, sd, hit))
    {
        // sd.linearRoughness *= 1.0;
        // sd.diffuse = 0.025;
        // sd.specular = 1.0;
        sd.linearRoughness = 0.35;
        sd.diffuse = 0.1;
        sd.specular = 1.0;
        bIsSurfaceShadingDataLoaded = true;
        uint frameSeed = gData.params.useFixedSeed ? 0 : gData.params.frameCount;
        
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
        path.sg = SampleGenerator.create(launchIndex, frameSeed * 12);

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

        #if !RENDER_SPEC
            if(sd.metallic > 0.5f){
                sd.diffuse = 0.1;
                sd.metallic = 0.0f;
            }
        #endif

        bool RussianRoullete = kUseRussianRoulette;

        bool bExecuteSelfLearn = false;

        ShadingData sdLastSurface;

        ShadingData sdLastSurfaceForVariance;
        PathData pathLastSurace;
        PathData pathLastSuraceForVariance;

        // Trace the path.



#if CONTINUE_RAY
        gXTrainResErrorID[pathIDTraining] = launchIndex.y * launchDim.x + launchIndex.x;

        do
        {
            sd.setActiveLobes(l);
            #if !RENDER_SPEC
                    if(sd.metallic > 0.5f){
                    sd.diffuse = 0.1;
                    sd.metallic = 0.0f;
                }
            if(all(sd.diffuse) < 0.05){
                sd.diffuse = 0.2;
            }
            #endif

            

#if DISABLE_NORMALS
            sd.N = sd.faceN;
#endif
            //sd.linearRoughness = 1.0f;

            //sd.linearRoughness = 0.6f;
            
            #if OUTPUT_TRANSFORM
            outputTransformData = constructOutputTransformData(sd);
            #endif

            #if !INCIDENT_NRC                
            prediction = constructPredictData(sd);              
            #endif

            
            float3 posPrev = sd.posW;
            float3 faceNprev = sd.N;

            float3 predAlbedo = sd.diffuse+sd.specular;
            outPredictionThp = float4(path.thp, 0);
            float3 pathLPrev = path.L;
            float3 pathThpPrev = path.thp;
            float3 emissive = sd.emissive;

            ShadingData sdPrev = sd;
            #if !INCIDENT_NRC
            prediction = constructPredictData(sd);              
            #endif

            bool successSample = false;

            sdLastSurfaceForVariance = sd;
            pathLastSuraceForVariance = path;

            rayOrigin = sd.computeNewRayOrigin();

            float3 prevFaceN = sd.faceN;

            ShadowRay EnvShadowRay;
            bKeepTracing = tracePathSimple(gData, sd, path, depth, aN, rayOrigin, EnvShadowRay, successSample, false);
            
            #if VARIANCE_LOSS_MODE && ESTIMATOR_NETWORK
            if(depth == 0){
                TrainingAdditionalData d;
                uint tid = addToEstimatorNNXTrainBuffer(constructPredictData(sdLastSurfaceForVariance));

                d.lightPathID = pathIDTraining;
                 #if TRAIN_DIRECT 
                    d.att = pathLPrev;
                #else
                    d.att = path.L;
                #endif
                d.thp = pathThpPrev*predAlbedo; 
                
                d.pdf = path.pdf;
                gEstimatorNNXTrainAdditionalBuffer[tid] = d;
            }
            #endif

            #if NORMALS_HACK
            successSample = successSample && (dot(prevFaceN, path.dir) > 0.0);
            #endif
            

            if(bExecuteSelfLearn){
                break;
            }
            
            outPredictionL = float4(path.L, 1.0);

            if(!successSample){
                break;
            }

            #if INCIDENT_NRC                
            prediction.Position = saturate((posPrev - bbStart) / (bbEnd - bbStart));
            prediction.Direction = ndir_to_oct_snorm(path.dir);
            prediction.SurfaceNormal = ndir_to_oct_unorm(faceNprev);
            #endif


            #if OUTPUT_TRANSFORM
            outputTransformData.incidentDir = path.dir;
            outputTransformData.pdf = path.pdf;
            #endif
            


            if(depth < GTNumIter && ((INCIDENT_NRC && depth < NUM_GUIDING_VERTICES) || (!INCIDENT_NRC && depth != 0))){

                float addU = sampleNext1D(path.sg);

                float addProb = 1.0f;
                if(depth == 1){ 
                    addProb = 1.0;
                }
                if(depth == 2){
                    addProb = 0.15;
                }
                if(depth == 3){
                    addProb = 0.05;
                }

                bool add_train = addU <= addProb;
                uint trainingID = 0;

                TrainingAdditionalData d;
                if (add_train)
                {
                    
                    #if VARIANCE_LOSS_MODE 
                    #if ESTIMATOR_NETWORK
                        XInferenceData predictionNetworkEstimater = constructPredictData(sdLastSurfaceForVariance);
                        trainingID =  addToPredBufferWithoutMapping(predictionNetworkEstimater);
                        gPredBufferThroughput[trainingID] = -pathThpPrev*predAlbedo;
                    #else
                        trainingID = gLossVarianceScatterID.IncrementCounter();
                    #endif
                    #endif

                    #if INCIDENT_NRC && TRAIN_DIRECT && SUPPORT_NEE && 0
                        {   
                            d.lightPathID = pathIDTraining;                
                            d.att = pathLPrev;




                            // check if light sampling generated a correct sample
                            if(EnvShadowRay.rayParams.w == 1.0f){
                                // Light Sampling Strategy
                                float3 toLightDir = EnvShadowRay.rayParams.xyz;
                                float4 toLightThp = float4(EnvShadowRay.Lr /gData.envMapSampler.eval(toLightDir), 1.0f);
                                XInferenceData predData = encodeData(sdPrev, toLightDir, toLightThp);

                                #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                                toLightThp.xyz *= gScene.envMap.eval(toLightDir);
                                #endif
                                    
                                uint tid = addToXTrainBuffer(predData);
                                d.thp = toLightThp.xyz; 

                                print(toLightThp.xyz);
                                print(path.thp);


                                #if VARIANCE_LOSS_MODE
                                    d.thp *= -1.0f;
                                #endif
                                d.pdf = path.pdf;

                                gXTrainAdditionalBuffer[tid] = d;

                                #if VARIANCE_LOSS_MODE
                                    gLossVarianceScatterID[tid] = trainingID;
                                    gLossVarianceThroughput[tid] = d.thp;
                                #endif
                            }
                        }
                        {
                            // BRDF Sampling Strategy

                            // We came here through BRDF sampling. The other sampling strategy is
                            // env map sampling. Evaluate it's probability for the current ray dir.
                            float lightPdf = gData.envMapSampler.evalPdf(path.dir) * getEnvLightSelectionPdf();

                            // Compute MIS weighted contribution from the environment map.
                            float misWeight = evalMIS(gData.params, 1, path.pdf, kLightSamplesPerVertex, lightPdf);          
                            float4 toLightThp = float4(path.thp, 0.0f);        
                            toLightThp *= misWeight;

                            print(toLightThp);

                            XInferenceData predData = encodeData(sdPrev, path.dir, toLightThp);

                            #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                            toLightThp.xyz *= gScene.envMap.eval(toLightDir);
                            #endif
                                
                            uint tid = addToXTrainBuffer(predData);

                            
                            d.thp = toLightThp.xyz; 

                            #if VARIANCE_LOSS_MODE
                                d.thp *= -1.0f;
                            #endif
                            d.pdf = path.pdf;

                            gXTrainAdditionalBuffer[tid] = d;

                            #if VARIANCE_LOSS_MODE
                                gLossVarianceScatterID[tid] = trainingID;
                                gLossVarianceThroughput[tid] = d.thp;
                            #endif
                        }

                    #else
                            
                            uint tid = addToXTrainBuffer(prediction);

                            d.lightPathID = pathIDTraining;
                                            
                            #if INCIDENT_NRC
                                #if TRAIN_DIRECT
                                    d.att = pathLPrev;
                                #else
                                    d.att = path.L;
                                #endif
                                d.thp = path.thp; 
                            #else
                                d.att = pathLPrev;
                                d.thp = pathThpPrev*predAlbedo; 
                            #endif

                            #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                                d.thp *= gScene.envMap.eval(path.dir);
                            #endif


                            
                            #if VARIANCE_LOSS_MODE
                                d.thp *= -1.0f;
                            #endif
                            d.pdf = path.pdf;

                            gXTrainAdditionalBuffer[tid] = d;

                            #if VARIANCE_LOSS_MODE
                                gLossVarianceScatterID[tid] = trainingID;
                                gLossVarianceThroughput[tid] = d.thp;
                            #endif
                #endif

                }

#if VARIANCE_LOSS_MODE
                if(NUM_VARIANCE_LOSS_SAMPLES != 0 ){
                int c = 0;

                PathData tmpPath = pathLastSuraceForVariance;
                do{
                    SampleGenerator sgcopy = tmpPath.sg;
                      tmpPath = pathLastSuraceForVariance;
                    tmpPath.sg = sgcopy;
                    
                    
                    ShadingData sdTmp = sdLastSurfaceForVariance;
                    rayOrigin = sdTmp.computeNewRayOrigin();
                    sdTmp.setActiveLobes(l);

                    #if !RENDER_SPEC
                    if(sdTmp.metallic > 0.5f){
                        sdTmp.diffuse = 0.1;
                        sdTmp.metallic = 0.0f;
                    }
                    #endif

                    #if DISABLE_NORMALS
                        sdTmp.N = sdTmp.faceN;
                    #endif

                    if (!(kRayFootprintMode == TexLODMode::Mip0))
                    {
                        // Create the ray footprint data for TexLOD.
                        tmpPath.rayFootprint = RayFootprint.create(hit.getTriangleHit(), launchIndex, launchDim, rayOrigin, gScene.camera.getPosition(), sdTmp.faceN, sdTmp.N, gData.params.screenSpacePixelSpreadAngle, sdTmp.linearRoughness, tmpPath.isSpecular());
                    }

                    bKeepTracing = true;
                    
                    float3 faceN = sdTmp.faceN;

#if INCIDENT_NRC
#if TRAIN_DIRECT && SUPPORT_NEE && 1
                    // MIS
                    if(c % 2 == 0){
                        // Light Sampling Strategy
                        ShadowRay shadowRay = {};
                        bool realValid;
                        generateShadowRay(gData.params, gData.envMapSampler, gData.emissiveSampler, sdTmp, gData.standardMaterial, 0, tmpPath, tmpPath.sg, shadowRay, realValid, true );
                        bKeepTracing = realValid;
                        if(bKeepTracing){
                            tmpPath.dir = shadowRay.rayParams.xyz;
                            tmpPath.thp = shadowRay.Lr;
                        }
                    }
                    else{
                        // BRDF Sampling Strategy
                        bKeepTracing = tracePathDirectCache(gData, sdTmp, tmpPath);      

                        if(bKeepTracing){
                            // We came here through BRDF sampling. The other sampling strategy is
                            // env map sampling. Evaluate it's probability for the current ray dir.
                            float lightPdf = gData.envMapSampler.evalPdf(tmpPath.dir) * getEnvLightSelectionPdf();

                            // Compute MIS weighted contribution from the environment map.
                            float misWeight = evalMIS(gData.params, 1, tmpPath.pdf, kLightSamplesPerVertex, lightPdf);                  
                            tmpPath.thp *= misWeight;
                        }
                    }
#else
                    bKeepTracing = tracePathDirectCache(gData, sdTmp, tmpPath);                    
#endif

#else
                    bKeepTracing = tracePathCache(gData, sdTmp, tmpPath, true);
#endif

                    #if NORMALS_HACK
                    bKeepTracing =  bKeepTracing && (dot(faceN, path.dir) > 0.0);
                    #endif
                    c += 1;

                    if(!bKeepTracing)
                        continue;

#if INCIDENT_NRC                    
                    prediction.Position = saturate((sdLastSurfaceForVariance.posW - bbStart) / (bbEnd - bbStart));
                    #if SPHERICAL_HASH_GRID
                    prediction.Direction.xy = saturate(ndir_to_oct_unorm(tmpPath.dir));
                    prediction.Direction.z = 1.0;
                    #else
                    prediction.Direction = ndir_to_oct_snorm(tmpPath.dir);
                    #endif
                    prediction.SurfaceNormal = ndir_to_oct_unorm(sdTmp.N);
#else                    
                    prediction = constructPredictData(sdTmp);              
#endif
                    uint tid = addToXTrainBuffer(prediction);

                    float thpMultiplier = (bKeepTracing) ? 1.0 : 0.0f;

                    #if INCIDENT_NRC
                    d.thp = tmpPath.thp; 
                    #else
                    d.thp = tmpPath.thp*(sdTmp.diffuse+sdTmp.specular);
                    #endif

                    #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                    d.thp *= gScene.envMap.eval(tmpPath.dir);
                    #endif
                        

                    #if INCIDENT_NRC && TRAIN_DIRECT && SUPPORT_NEE && 1
                    d.thp /= (NUM_VARIANCE_LOSS_SAMPLES/2);
                    #else
                    d.thp /= (NUM_VARIANCE_LOSS_SAMPLES);
                    #endif

                    d.thp *= thpMultiplier;
                    d.pdf = tmpPath.pdf;

                    gXTrainAdditionalBuffer[tid] = d;
                    gLossVarianceScatterID[tid] = trainingID;
        
                    
                    gLossVarianceThroughput[tid] = d.thp;
                }while(c < NUM_VARIANCE_LOSS_SAMPLES);

                }
#endif
            }

            depth += 1;

            
            if(relyOnNRCForLastVertex && depth == GTNumIter-1){
                bExecuteSelfLearn = true;
                sdLastSurface = sd;
                pathLastSurace = path;

                // have to make one more iteration for taking into account direct incident light by path-tracing in a case if our cache doesn't include incident direct lighting                
                if(TRAIN_DIRECT)
                    break;
            }

            if(depth == GTNumIter){
                break;
            }            

            // if(depth == 1){
            //     break;
            // } 

        } while (bKeepTracing);


        if(bExecuteSelfLearn){
                sd = sdLastSurface;               

                uint posID = gPredBuffer1.IncrementCounter();
                gPredBuffer1[posID] = saturate((sd.posW - bbStart) / (bbEnd - bbStart));

                uint num_iterations = 1;
                #if INCIDENT_NRC
                    num_iterations = NUM_NRC_SAMPLES;
                #endif
                for(uint k=0; k < num_iterations; k++)
                {
                    SampleGenerator sgcopy = path.sg;
                    path = pathLastSurace;
                    path.sg = sgcopy;
                    
                    
                    sd = sdLastSurface;
                    rayOrigin = sd.computeNewRayOrigin();
                    sd.setActiveLobes(l);
                    #if !RENDER_SPEC
                    if(sd.metallic > 0.5f){
                        sd.diffuse = 0.1;
                        sd.metallic = 0.0f;
                    }

                    #endif

                    #if DISABLE_NORMALS
                        sd.N = sd.faceN;
                    #endif

#if INCIDENT_NRC
                    if (!(kRayFootprintMode == TexLODMode::Mip0))
                    {
                        // Create the ray footprint data for TexLOD.
                        path.rayFootprint = RayFootprint.create(hit.getTriangleHit(), launchIndex, launchDim, rayOrigin, gScene.camera.getPosition(), sd.faceN, sd.N, gData.params.screenSpacePixelSpreadAngle, sd.linearRoughness, path.isSpecular());
                    }

                    bKeepTracing = true;
                    
                    float3 faceN = sd.faceN;
                    // Scatter ray to a random direction, estimate updated throughup. Extract prediction data based on intersected surface. 

                    bKeepTracing = tracePathDirectCache(gData, sd, path);                    
                    #if NORMALS_HACK
                    bKeepTracing = bKeepTracing && (dot(faceN, path.dir) > 0.0);
                    #endif
                    if(!bKeepTracing){
                        continue;
                    }

                    outPredictionThp = float4(path.thp, 0);
                    
                    prediction.Position = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
                    #if SPHERICAL_HASH_GRID
                    prediction.Direction.xy = saturate(ndir_to_oct_unorm(path.dir));
                    prediction.Direction.z = 1.0;
                    #else
                    prediction.Direction = ndir_to_oct_snorm(path.dir);
                    #endif
                    prediction.SurfaceNormal = ndir_to_oct_unorm(sd.N);
                    
                    depth++;
                    
#else
                    prediction = constructPredictData(sd);              
#endif
                    float3 thpc = outPredictionThp.xyz;
                    #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                    thpc.xyz *= gScene.envMap.eval(path.dir);
                    #endif

                    uint tid = addToPredBuffer(prediction, posID);
                    gPredBufferPixelID[tid] = pathIDTraining;
                    gPredBufferThroughput[tid] = thpc.xyz/num_iterations;
                }
        }
#else
#if DEBUG_RENDER && VARIANCE_LOSS_MODE && !HEMISPHERICAL_RENDER
        prediction = constructPredictData(sd);              
        uint tid = addToPredBufferWithoutMapping(prediction);
        gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;
        gPredBufferThroughput[tid] = (sd.diffuse+sd.specular);
#else

        sd.setActiveLobes(l);
        #if !RENDER_SPEC
                    if(sd.metallic > 0.5f){
                    sd.diffuse = 0.1;
                    sd.metallic = 0.0f;
                }

            #endif

        


        // SampleGenerator sgcopy = path.sg;
        // path = {};
        // path.sg = sgcopy;
        // depth = 0;
        //sd = sdVBufer;
        // rayOrigin = sd.computeNewRayOrigin();
        // path.origin = rayOrigin;
        // path.thp = float3(1.f);
        // path.L = 0.0;
        // path.hit = hit;
                        if (!(kRayFootprintMode == TexLODMode::Mip0))
                {
                    // Create the ray footprint data for TexLOD.
                    path.rayFootprint = RayFootprint.create(hit.getTriangleHit(), launchIndex, launchDim, rayOrigin, gScene.camera.getPosition(), sd.faceN, sd.N, gData.params.screenSpacePixelSpreadAngle, sd.linearRoughness, path.isSpecular());
                }

        bKeepTracing = true;
        sd.setActiveLobes(l);
        #if !RENDER_SPEC
                    if(sd.metallic > 0.5f){
                    sd.diffuse = 0.1;
                    sd.metallic = 0.0f;
                }

            #endif

        float p = sampleNext1D(path.sg);
        float russian_roullete_for_residual_error = 1.0;
        depth = 0;
        if(p <= russian_roullete_for_residual_error || !ENABLE_NRC){
            #if ENABLE_NRC
            path.thp /= russian_roullete_for_residual_error;
            #endif


#if HEMISPHERICAL_RENDER
            float2 sphereCoordStart = float2(launchIndex)/float2(launchDim-1);
            float2 sphereCoordStep = 1.0/float2(launchDim-1);
            float2 sphereCoordSampled = sampleNext2D(path.sg)*sphereCoordStep+sphereCoordStart;
            float3 scatterDir = oct_to_ndir_snorm(sphereCoordSampled*2-1.0);
            scatterDir = sd.fromLocal(scatterDir);
            path.thp = 1.0f;
            path.dir = scatterDir;
            path.pdf = 1.0f;
            path.L = 0.0;
#endif



            #if (!DEBUG_VIEW) || !ENABLE_NRC
            do
            {

                #if HEMISPHERICAL_RENDER && INCIDENT_NRC
                if(depth == 0 && launchIndex.x == 0 && launchIndex.y == 0){
                    gPredBuffer1.IncrementCounter();
                    gPredBuffer1[0] = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
                }
                #endif

                #if HEMISPHERICAL_RENDER
                if(depth == 0 && dot(path.dir, sd.N) <= 0.0){
                    outPredictionL = float4(path.L, 1.0); // just take into account 
                    outPredictionThp = float4(path.thp, 0);
                    break;
                }
                #endif


                ShadingData sdVBufer = sd;
                PathData pathcopy = path;
                
                uint posID = 0;
                bool shouldRequest = true;
                if(ENABLE_NRC && depth < NUM_GUIDING_VERTICES && (!HEMISPHERICAL_RENDER || depth == 0)){
                    #if HEMISPHERICAL_RENDER && INCIDENT_NRC
                    posID = 0;
                    shouldRequest = false;
                    #endif

                    int num_samples = NUM_NRC_SAMPLES;
                    #if HEMISPHERICAL_RENDER
                    num_samples = 1;
                    #endif

                    for(uint k=0; k < num_samples; k++)
                    {
                        
                        if(k != 0){
                            SampleGenerator sgcopy = path.sg;
                            path = pathcopy;
                            path.sg = sgcopy;
                        }
                        
                        
                        sd = sdVBufer;
                        rayOrigin = sd.computeNewRayOrigin();
                        sd.setActiveLobes(l);
                        #if !RENDER_SPEC
                            if(sd.metallic > 0.5f){
                            sd.diffuse = 0.1;
                            sd.metallic = 0.0f;
                        }

                    #endif
                        #if DISABLE_NORMALS
                            sd.N = sd.faceN;
                        #endif

                        if (!(kRayFootprintMode == TexLODMode::Mip0))
                        {
                            // Create the ray footprint data for TexLOD.
                            path.rayFootprint = RayFootprint.create(hit.getTriangleHit(), launchIndex, launchDim, rayOrigin, gScene.camera.getPosition(), sd.faceN, sd.N, gData.params.screenSpacePixelSpreadAngle, sd.linearRoughness, path.isSpecular());
                        }


                        bKeepTracing = true;
                        

                        ////print(sd.diffuse);
                        ////print(sd.posW);            
                        ////print(rayOrigin);
                        float3 faceN = sd.faceN;
                        #if OUTPUT_TRANSFORM
                        outputTransformData = constructOutputTransformData(sd);
                        #endif
                        // Scatter ray to a random direction, estimate updated throughup. Extract prediction data based on intersected surface. 
                        bool succesSample = true;
                        #if INCIDENT_NRC
                            #if !HEMISPHERICAL_RENDER
                                #if TRAIN_DIRECT && SUPPORT_NEE
                                    // MIS
                                    print(k);
                                    if(k % 2 == 0){
                                        // Light Sampling Strategy
                                        ShadowRay shadowRay = {};
                                                                            
                                        bool realvalid;
                                        generateShadowRay(gData.params, gData.envMapSampler, gData.emissiveSampler, sd, gData.standardMaterial, 0, path, path.sg, shadowRay, realvalid, true);
                                        bKeepTracing = realvalid;

                                        if(bKeepTracing){
                                            path.dir = shadowRay.rayParams.xyz;
                                            path.thp = shadowRay.Lr;
                                        }

                                        print(bKeepTracing);
                                        print(path.dir);
                                        print(path.thp);
                                        print(getEnvLightSelectionPdf());
                                    }
                                    else{
                                        // BRDF Sampling Strategy
                                        bKeepTracing = tracePathDirectCache(gData, sd, path);      

                                        if(bKeepTracing){
                                            // We came here through BRDF sampling. The other sampling strategy is
                                            // env map sampling. Evaluate it's probability for the current ray dir.
                                            float lightPdf = gData.envMapSampler.evalPdf(path.dir) * getEnvLightSelectionPdf();

                                            // Compute MIS weighted contribution from the environment map.
                                            float misWeight = evalMIS(gData.params, 1, path.pdf, kLightSamplesPerVertex, lightPdf);                  
                                            path.thp *= misWeight;
                                        }

                                        print(bKeepTracing);
                                        print(path.dir);
                                        print(path.thp);
                                    }
                                #else
                                    bKeepTracing = tracePathDirectCache(gData, sd, path);
                                #endif
                            #endif
                        #else
                            bKeepTracing = tracePathCache(gData, sd, path, !HEMISPHERICAL_RENDER);
                        #endif

                        #if NORMALS_HACK
                        bKeepTracing = bKeepTracing && (dot(faceN, path.dir) > 0.0);
                        #endif

 
                        

                        if(!succesSample || !bKeepTracing){
                            continue;
                        }

                        
                        
                        outPredictionThp = float4(path.thp, 0);
                        #if INCIDENT_NRC                
                            prediction.Position = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
                            #if SPHERICAL_HASH_GRID
                            prediction.Direction.xy = saturate(ndir_to_oct_unorm(path.dir));
                            prediction.Direction.z = 1.0;
                            #else
                            prediction.Direction = ndir_to_oct_snorm(path.dir);
                            #endif
                            prediction.SurfaceNormal = ndir_to_oct_unorm(sd.N);
                        #else
                        prediction = constructPredictData(sd);              
                        outPredictionThp.xyz *= sd.diffuse+sd.specular;
                        posID = gPredBuffer1.IncrementCounter();
                        gPredBuffer1[posID] = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
                        #endif

                        ////print(k);

                        ////print(k);

                        #if OUTPUT_TRANSFORM
                        outputTransformData.incidentDir = path.dir;
                        outputTransformData.pdf = path.pdf;
                        #endif

                        if(shouldRequest){
                            #if INCIDENT_NRC
                            posID = gPredBuffer1.IncrementCounter();
                            gPredBuffer1[posID] = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
                            shouldRequest = false;
                            #endif
                        }
                        
                        uint tid = addToPredBuffer(prediction, posID);
                        ////print(tid);
                        
                        #if OUTPUT_TRANSFORM
                        gPredBufferOutActAddData[tid] = outputTransformData;
                        #endif

                        float3 thpCur = outPredictionThp.xyz;
                        #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                        thpCur *= gScene.envMap.eval(path.dir);
                        #endif

                        gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;


                        
                        #if INCIDENT_NRC && TRAIN_DIRECT && SUPPORT_NEE && !HEMISPHERICAL_RENDER
                        float num_div = 2.0/num_samples;
                        #else
                        float num_div = 1.0/num_samples;
                        #endif
                        
                        // if(kUseMIS){
                        //     num_div = 0;
                        // }
                    
                        gPredBufferThroughput[tid] = thpCur.xyz*num_div;
                    }

                    SampleGenerator sgcopy = path.sg;
                    path = pathcopy;
                    path.sg = sgcopy;

                    sd = sdVBufer;
                    rayOrigin = sd.computeNewRayOrigin();
                }


                //sd.setActiveLobes((uint)LobeType::Diffuse);
                //sd.linearRoughness = 0.6f;  
                sd.setActiveLobes(l);   
                #if !RENDER_SPEC
                if(sd.metallic > 0.5f){
                    sd.diffuse = 0.1;
                    if(sd.metallic > 0.5f){
                        sd.diffuse = 0.1;
                        sd.metallic = 0.0f;
                    }
                }
                    
                #endif     
                #if DISABLE_NORMALS
                    sd.N = sd.faceN;
                #endif 
                
                #if OUTPUT_TRANSFORM
                outputTransformData = constructOutputTransformData(sd);
                #endif
                float3 faceN = sd.faceN;
                // Scatter ray to a random direction, estimate updated throughup. Extract prediction data based on intersected surface. 
                bool successSample = true;

                ShadowRay EnvShadowRay;
                bKeepTracing = tracePathSimple(gData, sd, path, depth, aN, rayOrigin, EnvShadowRay, successSample, RENDER_ONLY_INDIRECT, false, HEMISPHERICAL_RENDER && depth==0);
                // if(dot(faceN, path.dir) <= 0.0)
                //     bKeepTracing = false;
                #if NORMALS_HACK
                successSample = successSample && (dot(faceN, path.dir) > 0.0);
                #endif

                outPredictionL = float4(path.L, 1.0); // just take into account 


                #if INCIDENT_NRC && !HEMISPHERICAL_RENDER && TRAIN_DIRECT && SUPPORT_NEE && 0

                #else
                if(!successSample)
                    break;
                #endif

                #if !INCIDENT_NRC
                if(!bKeepTracing)
                    break;
                #endif

                outPredictionThp = float4(path.thp, 0);

                #if RENDER_ONLY_INDIRECT && !HEMISPHERICAL_RENDER
                if(depth == 0){
                    path.L = 0;
                }
                #endif
                
                
                if (ENABLE_NRC && depth < NUM_GUIDING_VERTICES && !BIASED && !HEMISPHERICAL_RENDER)
                {   
                    #if INCIDENT_NRC && TRAIN_DIRECT && SUPPORT_NEE && 0
                        {   
                            if(EnvShadowRay.rayParams.w == 1.0f){
                                // Light Sampling Strategy
                                float3 toLightDir = EnvShadowRay.rayParams.xyz;
                                float4 toLightThp = float4(EnvShadowRay.Lr /gData.envMapSampler.eval(toLightDir), 1.0f);
                                XInferenceData predData = encodeData(sdVBufer, toLightDir, toLightThp);

                                #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                                toLightThp.xyz *= gScene.envMap.eval(toLightDir);
                                #endif
                                    
                                uint tid = addToPredBuffer(predData, posID);
                                residualErrorID = tid;

                                gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;
                                gPredBufferThroughput[tid] = -toLightThp.xyz; // NEGATIVE FOR ESTIMATING ERROR
                            }
                        }
                        {
                            if(successSample){
                                // BRDF Sampling Strategy

                                // We came here through BRDF sampling. The other sampling strategy is
                                // env map sampling. Evaluate it's probability for the current ray dir.
                                float lightPdf = gData.envMapSampler.evalPdf(path.dir) * getEnvLightSelectionPdf();

                                // Compute MIS weighted contribution from the environment map.
                                float misWeight = evalMIS(gData.params, 1, path.pdf, kLightSamplesPerVertex, lightPdf);          
                                float4 toLightThp = float4(path.thp, 0.0f);        
                                toLightThp *= misWeight;

                                XInferenceData predData = encodeData(sdVBufer, path.dir, toLightThp);

                                #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                                toLightThp.xyz *= gScene.envMap.eval(toLightDir);
                                #endif
                                    
                                uint tid = addToPredBuffer(predData, posID);
                                residualErrorID = tid;

                                gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;
                                gPredBufferThroughput[tid] = -toLightThp.xyz; // NEGATIVE FOR ESTIMATING ERROR
                            }
                        }

                    #else
                        XInferenceData predData = encodeData(sdVBufer, path.dir, outPredictionThp);
                            
                        #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                        outPredictionThp.xyz *= gScene.envMap.eval(path.dir);
                        #endif

                        if(shouldRequest){
                            posID = gPredBuffer1.IncrementCounter();
                            gPredBuffer1[posID] = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
                        }

                        uint tid = addToPredBuffer(predData, posID);
                        residualErrorID = tid;

                        gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;
                        gPredBufferThroughput[tid] = -outPredictionThp.xyz; // NEGATIVE FOR ESTIMATING ERROR
                    #endif
                }

                depth++;

                if (depth == GTNumIter){
                    break;
                }

                // if(HEMISPHERICAL_RENDER)
                // {
                //     break;
                // }
                

            } while (bKeepTracing);
            #endif
        }
#endif

        //outPredictionL += 0*directLightEstimator;
#endif
        
        logPathLength(path.length);

        // Accumulate after clamping.
        // Note the comparison is written so that NaNs propagate (unless the compiler rewrites it).
        // TODO: Check the generated code that this is the case.
        outColor += gData.params.clampSamples && path.L > gData.params.clampThreshold ? gData.params.clampThreshold : path.L;

        // We're done accumulating over all samples.
        const float invSpp = 1.f / kSamplesPerPixel;
        outColor *= invSpp;
        outAlbedo = sd.diffuse+sd.specular;
        outAlpha = 1.f;
    }
    else
    {   
        outPredictionL.xyz = evalBackground(-sd.V);
        outPredictionL.w = 1.0f;
        //outPredictionL = 1000.0f;
        // Background pixel.
        outColor = evalBackground(-sd.V);
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
    ////print(outPredictionL.xyz);

    #if VARIANCE_LOSS_MODE && !ESTIMATOR_NETWORK
        float4 expectedValue = gExpectedValueTmp[launchIndex];
        outPredictionL -= expectedValue;
    #endif

    LThp data;
    data.L = outPredictionL.xyz;
    data.thp = outPredictionThp.xyz;

    if (!relyOnNRCForLastVertex || !bIsSurfaceShadingDataLoaded)
    {
        data.thp = 0;
    }
    
    const uint pathIDTrainingAfterPredict = pathIDTraining+predictLOD0Offset;
    gLThpSelfTrainingBuffer[pathIDTraining] = data;

    launchDim /= (8, 4);
#else

#if VARIANCE_LOSS_MODE && !HEMISPHERICAL_RENDER    
    float4 newExpectedValue = 0;
        float weight = 1.0/(num_training_id+1);

        if(num_training_id >= expected_value_num_samples || 1){
            newExpectedValue = lerp(gExpectedValueTmp[launchIndex], outPredictionL, weight);
            //newExpectedValue = gExpectedValueTmp[launchIndex];
        }
        else{
            newExpectedValue = 0.0f;
            if(num_training_id != 0){
                newExpectedValue = gExpectedValueTmp[launchIndex];
            }

            newExpectedValue += outPredictionL*weight;
            //newExpectedValue *= weight;
        }
    

    gExpectedValueTmp[launchIndex] = float4(newExpectedValue.rgb, 1.0f);
#endif



    #if !HEMISPHERICAL_RENDER    
    if(ENABLE_NRC && BIASED){
       outPredictionL = 0;
    }
    #endif

    #if !HEMISPHERICAL_RENDER    
    float cd = length(int2(launchIndex.xy)-debug_pixel_id);
    if(debug_pixel_id.x != -1 && cd <= 9 && cd >= 7)
        outPredictionL.xyz += 10*float3(1.0, 1.0, 0.0f);
    #endif

    gOutputColor[launchIndex] = outPredictionL;

    #if HEMISPHERICAL_RENDER 

    #endif

  
    #if !HEMISPHERICAL_RENDER && RESIDUAL_ERROR_ESTIMATION
        uint2 lc = launchIndex.xy/uint2(8, 4);
        uint2 ld = launchDim/uint2(8, 4);
        uint pathIDTraining2 = lc.y*ld.x+lc.x;
        uint trainignPixelPos = gXTrainResErrorID[pathIDTraining2];
        if(trainignPixelPos == launchIndex.y * launchDim.x + launchIndex.x){
            ResidualErrorData data;
            data.id = residualErrorID;
            data.estimation = outPredictionL.rgb;
            gResErrorID[pathIDTraining2] = data;
        }
    #endif
    

#endif
    //gOutputColor[launchIndex] = 0;
}
