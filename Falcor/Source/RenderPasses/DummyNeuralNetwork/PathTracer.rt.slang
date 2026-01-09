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
import Experimental.Scene.Lights.EmissiveLightSamplerHelpers;

#define ONLY_POS 1


#ifndef MLOD
#define MLOD 1
#endif


#ifndef STOP_BY_METRIC
#define STOP_BY_METRIC 0
#endif

#ifndef STOP_BY_OUR_METRIC
#define STOP_BY_OUR_METRIC 0
#endif

#define NUM_GUIDING_VERTICES 1
#define EMISSIVE_HACK 0
#define DUMMY_MAPPING 0
#define SMALL_TRAIN_BUFF 1
#define USE_METRIC 0

#define DISABLE_NORMALS 0
#define RENDER_ONLY_CV 0

#define NORMALS_HACK 0

#define RENDER_SPEC 1
#define RENDER_DIFF 1


#ifndef EVAL_DIRECT_ONLY_ONCE
#define EVAL_DIRECT_ONLY_ONCE 0
#endif


#ifndef TRAIN_DIRECT
#define TRAIN_DIRECT 0
#endif

#define ENV_MAP_SAMPLING_IS_OPTIMIZED_FOR_INFERENCE 0

#ifndef USE_GUIDANCE_OF_ENV_MAP
#define USE_GUIDANCE_OF_ENV_MAP 1
#endif


#ifndef SUPPORT_NEE
#define SUPPORT_NEE 0
#endif

#ifndef DEBUG_RENDER
#define DEBUG_RENDER 0
#endif



#ifndef DEBUG_VIEW
#define DEBUG_VIEW 1
#endif

#ifndef BIASED
#define BIASED 1
#endif

#ifndef ENABLE_NRC
#define ENABLE_NRC 1
#endif


//#define ENABLE_NRC 0
#ifndef RES_VARIANCE_LOSS
#define RES_VARIANCE_LOSS 0
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

#define SPHERICAL_MAPPING 1


cbuffer GeneralData : register(b4)
{
    float3 bbStart;
    float3 bbEnd;
    float roughnessThreshold;
    uint lodSize0;
    uint lodSize1;
    uint lodSize2;
    uint gd_hash_grid_res;
    uint gd_hash_grid_size;
    uint max_bounce_with_cv;
    int2 debug_pixel_id;
    int2 original_dim;
    int2 train_dim_window;
    float gRoulleteProb;
    uint expected_value_num_samples;
    uint num_training_id;
    uint samples_per_pixel;
    uint g_num_samples[3];
    float errorThreshold;
};

ParameterBlock<PathTracerData> gData;


// Outputs (optional)
RWTexture2D<float4> gOutputM2;
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputAlbedo;
RWTexture2D<float4> gOutputNormal;
RWTexture2D<uint> gOutputTime;


#ifndef VARIANCE_LOSS_MODE
#define VARIANCE_LOSS_MODE 0
#endif

#ifndef ERROR_BASED_METRIC
#define ERROR_BASED_METRIC 1
#endif

#ifndef DO_ERROR_TEST
#define DO_ERROR_TEST 1
#endif

#ifndef NUM_VARIANCE_LOSS_SAMPLES
#define NUM_VARIANCE_LOSS_SAMPLES 1
#endif


#ifndef FILTERING_NET
#define FILTERING_NET 0
#endif

#ifndef ESTIMATOR_NETWORK
#define ESTIMATOR_NETWORK 0
#endif


Texture2D<float4> ourEstimations;
Texture2D<float4> gtEstimations;


RWTexture2D<float4> gExpectedValueTmp;
RWStructuredBuffer<uint> gLossVarianceScatterID;
RWStructuredBuffer<float3> gLossVarianceThroughput;


RWStructuredBuffer<uint> gPredBuffer0;
RWStructuredBuffer<float3> gPredBuffer1;
RWStructuredBuffer<float> gPredBuffer2;
RWStructuredBuffer<float3> gPredBuffer3;
RWStructuredBuffer<uint> gPredBuffer4;

RWStructuredBuffer<uint> gXTrainResErrorID;
RWStructuredBuffer<uint> gPredBufferMapping1;

RWStructuredBuffer<uint> gXTrainBuffer0;
RWStructuredBuffer<float3> gXTrainBuffer1;
RWStructuredBuffer<float> gXTrainBuffer2;
RWStructuredBuffer<float3> gXTrainBuffer3;
RWStructuredBuffer<uint> gXTrainBuffer4;

RWStructuredBuffer<uint> gEstimatorNNXTrainBuffer0;
RWStructuredBuffer<float3> gEstimatorNNXTrainBuffer1;
RWStructuredBuffer<float> gEstimatorNNXTrainBuffer2;
RWStructuredBuffer<float3> gEstimatorNNXTrainBuffer3;
RWStructuredBuffer<uint> gEstimatorNNXTrainBuffer4;

RWStructuredBuffer<TrainingAdditionalData> gEstimatorNNXTrainAdditionalBuffer;

RWStructuredBuffer<uint> gPredBufferPixelID;
RWStructuredBuffer<uint2> gTrainPixel;


RWStructuredBuffer<float3> gPredBufferThroughput;
RWStructuredBuffer<OutputTransformData> gPredBufferOutActAddData;
RWStructuredBuffer<LThp> gLThpSelfTrainingBuffer;
RWStructuredBuffer<TrainingAdditionalData> gXTrainAdditionalBuffer;

RWStructuredBuffer<int> gNNMapTrainBufferKeys;
RWStructuredBuffer<int> gNNMapTrainBufferValues;

RWStructuredBuffer<int> gNNMapPredBufferKeys;
RWStructuredBuffer<int> gNNMapPredBufferValues;


uint addToPredBufferWithoutMappingByID(XInferenceData data, uint id){
    gPredBuffer0[id] = (f32tof16(data.SurfaceNormal.x) << 16) | f32tof16(data.SurfaceNormal.y);
    gPredBuffer1[id] = data.Position;
    gPredBuffer2[id] = data.Roughness;
    gPredBuffer3[id] = data.Color;
    gPredBuffer4[id] = (f32tof16(data.Direction.x) << 16) | f32tof16(data.Direction.y);
    return id;
}


uint addToPredBufferWithoutMapping(XInferenceData data){
    uint id = gPredBuffer4.IncrementCounter();
    addToPredBufferWithoutMappingByID(data, id);
    return id;
}

uint addToPredBuffer(XInferenceData data, uint posID){
    uint id = gPredBuffer4.IncrementCounter();
    gPredBuffer4[id] = (f32tof16(data.Direction.x) << 16) | f32tof16(data.Direction.y);

    gPredBufferMapping1[id] = posID;

    return id;
}

uint addToXTrainBuffer(XInferenceData data){
    uint id = gXTrainBuffer4.IncrementCounter();

    gXTrainBuffer0[id] = (f32tof16(data.SurfaceNormal.x) << 16) | f32tof16(data.SurfaceNormal.y);
    gXTrainBuffer1[id] = data.Position;
    gXTrainBuffer2[id] = data.Roughness;
    gXTrainBuffer3[id] = data.Color;
    gXTrainBuffer4[id] = (f32tof16(data.Direction.x) << 16) | f32tof16(data.Direction.y);

    return id;
}

uint addToEstimatorNNXTrainBuffer(XInferenceData data){
    uint id = gEstimatorNNXTrainBuffer4.IncrementCounter();
    addToEstimatorNNXTrainBufferByID(data, id);

    return id;
}

uint addToEstimatorNNXTrainBufferByID(XInferenceData data, uint id){
    gEstimatorNNXTrainBuffer0[id] = (f32tof16(data.SurfaceNormal.x) << 16) | f32tof16(data.SurfaceNormal.y);
    gEstimatorNNXTrainBuffer1[id] = data.Position;
    gEstimatorNNXTrainBuffer2[id] = data.Roughness;
    gEstimatorNNXTrainBuffer3[id] = data.Color;
    gEstimatorNNXTrainBuffer4[id] = (f32tof16(data.Direction.x) << 16) | f32tof16(data.Direction.y);

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

float2 ndir_to_spherical_snorm(float3 n)
{
    // Assuming n is already normalized
    float theta = acos(n.z); // Angle from the positive z-axis
    float phi = atan2(n.y, n.x); // Azimuthal angle in the x-y plane

    // Normalize theta and phi to the range [0, 1]
    float normalizedTheta = theta / PI; // PI is a predefined constant for π
    float normalizedPhi = (phi + PI) / (2.0 * PI); // Shift range from [-π, π] to [0, 2π] then normalize

    return float2(normalizedTheta, normalizedPhi);
}

XInferenceData constructPredictData(ShadingData sd, bool permute = false)
{
    XInferenceData surfaceInfo;

    surfaceInfo.Position = saturate((sd.posW - bbStart) / (bbEnd - bbStart));
    surfaceInfo.SurfaceNormal = ndir_to_spherical_snorm(sd.N);
    surfaceInfo.Direction = ndir_to_spherical_snorm(sd.V);
    surfaceInfo.Roughness = 1.0 - exp(-sd.linearRoughness);
    //surfaceInfo.Color = 1.0;
    surfaceInfo.Color = sd.diffuse+sd.specular;


    // surfaceInfo.Color = 1.0;
    // surfaceInfo.Roughness = 1.0f;
    // surfaceInfo.Direction = 1.0f;
    // surfaceInfo.SurfaceNormal = 1.0f;


#if !ONLY_POS
    surfaceInfo.ScatteredDir = sd.V;
    surfaceInfo.SurfaceRoughness = 1.0 - exp(-sd.linearRoughness);
    surfaceInfo.DiffuseReflectance = sd.diffuse;
    surfaceInfo.SpecularRefelctance = sd.specular;
#endif

    return surfaceInfo;
}

XInferenceData encodeData(ShadingData sdIncident, ShadingData sdOutgoing, float3 pathDir, inout float4 thp){
    XInferenceData prediction;
    #if INCIDENT_NRC
    prediction.Position = saturate((sdIncident.posW - bbStart) / (bbEnd - bbStart));

    #if SPHERICAL_HASH_GRID
    prediction.Direction.xy = saturate(ndir_to_oct_unorm(pathDir));
    prediction.Direction.z = 1.0;
    #else
    prediction.Direction = ndir_to_oct_snorm(pathDir);
    #endif
    prediction.SurfaceNormal = ndir_to_oct_unorm(sdIncident.N);
    prediction.Roughness = 1.0 - exp(-sdIncident.linearRoughness);
    prediction.Color = sdIncident.diffuse+sdIncident.specular;
    //prediction.Roughness = 1.0;
    #else
    prediction = constructPredictData(sdOutgoing);
    thp.xyz *= sdOutgoing.diffuse+sdOutgoing.specular;
    #endif

    // // prediction.Position = 1.0;
    // prediction.Roughness = 1.0;
    // prediction.Color = 1.0f;
    //prediction.SurfaceNormal = 1.0f;


    return prediction;
}

XInferenceData encodeData(ShadingData sdIncident, ShadingData sdOutgoing, float3 pathDir, inout float3 thp)
{
    float4 thp_t = float4(thp, 0.0);
    XInferenceData res = encodeData(sdIncident, sdOutgoing, pathDir, thp_t);
    thp = thp_t.xyz;
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

float lum(float3 v) {
    return 0.299 * v.x + 0.587*v.y + 0.114*v.z;
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

    uint2 trainDim = launchDim/train_dim_window;

    float3 trainOffset = 0;

#if CONTINUE_RAY

#if 1
    const uint pathIDTraining = launchIndex.y * launchDim.x + launchIndex.x;
#else
    const uint pathIDTraining = gLThpSelfTrainingBuffer.IncrementCounter();
#endif

    launchDim *= train_dim_window;

    predictDim = launchDim;
    trainDim = launchDim/train_dim_window;

    uint frameSeed2 = gData.params.useFixedSeed ? 0 : gData.params.frameCount;
    SampleGenerator sg0 = SampleGenerator.create(launchIndex, frameSeed2 * 234+23423*CONTINUE_RAY);
    // FIX IT!!! MULT * 0
    launchIndex.xy = launchIndex.xy * train_dim_window + sampleNext2D(sg0) * (train_dim_window-uint2(1, 1));

    gTrainPixel[pathIDTraining] = launchIndex;

    float offsetSize = 1.0f/lodSize1*0.2;

#if HASH_GRID
    offsetSize = 1.0f/(lodSize1*2)*0.2;
#endif
    bool bResetTrainingLight = false;
    //trainOffset = sampleNext3D(sg0)*(2*offsetSize)-offsetSize;

    relyOnNRCForLastVertex = sampleNext1D(sg0) > 0.05;
#if !ENABLE_NRC
    relyOnNRCForLastVertex = false;
#endif
#endif
    const uint predictLOD0Offset = predictDim.x*predictDim.y;
    const uint predictLOD0WithNRCOffset = predictDim.x*predictDim.y+trainDim.x*trainDim.y;

#if (!Q_LEARNING || INCIDENT_NRC) && !FILTERING_NET
    relyOnNRCForLastVertex = false;
#endif



    //relyOnNRCForLastVertex = false;
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
    float4 outPredictionLAdd = float4(0);
    float4 outPredictionThp = float4(0);

    bool set_pred = false;
    XInferenceData prediction = {};


    Ray camRay = gScene.camera.computeRayPinhole(launchIndex, launchDim);

    HitInfo hit;
    ShadingData sd;


    int residualErrorID = -1;


    uint32_t GTNumIter = gData.params.maxBounces;


    if((FILTERING_NET || Q_LEARNING) && ENABLE_NRC && relyOnNRCForLastVertex){
        GTNumIter -= 1;
        if(GTNumIter > 3){
            GTNumIter = 3;
        }
    }

    bool bIsSurfaceShadingDataLoaded = false;

    uint2 surfacePixelID = launchIndex;

#if HEMISPHERICAL_RENDER
    surfacePixelID = debug_pixel_id;
#endif

    float3 outAlbedoValue = float3(0.0f, 0.0f, 0.0f);
    float3 outNormalValue = float3(0.0f, 0.0f, 0.0f);

    float4 backL = 0;
    if (loadShadingData(surfacePixelID, original_dim, gScene.camera, sd, hit))
    {


        outAlbedoValue = sd.diffuse+sd.specular;
        outNormalValue = sd.N;

        //sd.linearRoughness *= 0.5;
        // sd.diffuse = 1.0;
        // sd.specular = 0.0;
        // sd.linearRoughness = 1.0;
        // sd.metallic = 0;
        // sd.setActiveLobes((uint)LobeType::Diffuse);

        // sd.specular = 1.0;
        // sd.linearRoughness = 0.35;
        // sd.diffuse = 0.1;
        // sd.specular = 1.0;

        float3 rayOrigin = sd.computeNewRayOrigin();
        const float3 rayOriginV = rayOrigin;
        ShadingData sdOriginalBufer = sd;
        int num_iterations = (ENABLE_NRC) ? 1 : samples_per_pixel;




        [loop]
        for (uint sampleIdx = 0; sampleIdx < num_iterations; sampleIdx++)
        {
            sd = sdOriginalBufer;
            PathData path = {};
            uint depth = 0;
            rayOrigin = rayOriginV;

            //sd.linearRoughness = clamp(sd.linearRoughness, 0.4, 1.0);
            //        sd.diffuse = 0.00001;
        // // sd.linearRoughness *= 1.0;
        // sd.diffuse = 0.0;
        // sd.specular = 1.0;
        // sd.linearRoughness = 0.45;
        // // sd.diffuse = 0.1;
        // sd.specular = 1.0;
        bIsSurfaceShadingDataLoaded = true;
        uint frameSeed = gData.params.useFixedSeed ? 0 : gData.params.frameCount;


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
        path.sg = SampleGenerator.create(launchIndex, frameSeed * 29*samples_per_pixel+sampleIdx);

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

        uint l = (uint)LobeType::All;

        #if !RENDER_SPEC
        sd.linearRoughness = 1.0f;
        //sd.specular = 0.01;
        //l &= ~(uint)LobeType::Specular;
        #endif

        #if !RENDER_DIFF
        l &= ~(uint)LobeType::Diffuse;
        #endif
#if MAKE_GLOSSY
        sd.linearRoughness = 0.4;
#endif

        sd.setActiveLobes(l );

        #if !RENDER_SPEC
            if(sd.metallic > 0.5f && 0){
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

        bool bAlreadyAddedTrainNRC = false;
        bool bAlreadyAddedNIRC = false;
#if CONTINUE_RAY
        float a_current = 0;
        const float a0_b = length(gScene.camera.getPosition() - sd.posW) / (4 * 3.14 * dot(sd.V, sd.N));
        do
        {

#if !RENDER_SPEC
            sd.linearRoughness = 1.0f;
            //sd.specular = 0.01;
            //l &= ~(uint)LobeType::Specular;
#endif
#if MAKE_GLOSSY
            //sd.linearRoughness = 0.4;
#endif



            sd.setActiveLobes(l);
            #if !RENDER_SPEC
                    if(sd.metallic > 0.5f && 0){
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




            float3 posPrev = sd.posW;
            float3 faceNprev = sd.N;

            float3 predAlbedo = sd.diffuse+sd.specular;
            outPredictionThp = float4(path.thp, 0);
            float3 pathLPrev = path.L;
            float3 pathThpPrev = path.thp;
            float3 emissive = sd.emissive;

            ShadingData sdPrev = sd;


            bool successSample = false;

            sdLastSurfaceForVariance = sd;
            pathLastSuraceForVariance = path;

            rayOrigin = sd.computeNewRayOrigin();

            float3 prevFaceN = sd.faceN;

            ShadowRay EnvShadowRay;
            TriangleLightHit emissiveTriangleHit;
            float3 emissiveLight;
            float emissiveMISWeight;
            float3 emissiveThp;
            float3 envLight;




            float3 thpTempIntermediate = path.thp;

            bKeepTracing = tracePathSimple(gData, sd, path, depth, aN, rayOrigin, EnvShadowRay,emissiveTriangleHit,emissiveLight,emissiveMISWeight,emissiveThp, envLight, successSample, thpTempIntermediate, false);

            a_current += sqrt(length(sd.posW - sdPrev.posW) / (abs(dot(sdPrev.N, sdPrev.V)) * path.pdf));

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

            #if (VARIANCE_LOSS_MODE && ESTIMATOR_NETWORK) || FILTERING_NET
            {
                float addU = sampleNext1D(path.sg);

                if(depth < 3 && (sdPrev.linearRoughness >= roughnessThreshold) && (!bAlreadyAddedTrainNRC || 1)){
                    bAlreadyAddedTrainNRC = true;
                    TrainingAdditionalData d;
                    uint tid = addToEstimatorNNXTrainBuffer(constructPredictData(sdLastSurfaceForVariance));

                    d.lightPathID = pathIDTraining;
                    d.att = pathLPrev;
                    d.thp = pathThpPrev*predAlbedo;
                    d.pdf = 1.0f;
                    d.thpDiv = 1.0f;
                    gEstimatorNNXTrainAdditionalBuffer[tid] = d;
                }
            }
            #endif

            float addU = sampleNext1D(path.sg);
            // if((Q_LEARNING || FILTERING_NET) && !relyOnNRCForLastVertex){
            //     if(depth <= 0)
            //         addU += 1.0;
            //     else
            //         addU = 0.0;

            // }

            // if(!Q_LEARNING && !FILTERING_NET){
            //     if(depth == 0){
            //         addU += 1.0;
            //     }
            //     else
            //         addU = 0.0;
            // }

            uint add_offset = 0;
            // if(!INCIDENT_NRC){
            //     add_offset += 2;
            // }

            if(max_bounce_with_cv == 1)
                add_offset += 1;



            bool bMetric = a_current * a_current > 0.01 * a0_b || !bKeepTracing;
            if (bMetric && INCIDENT_NRC) {
                if (path.isDelta() && depth == 0)
                    bMetric = false;
                else
                    bMetric = true;
            }



            //if(  ((depth != 0 || INCIDENT_NRC && !STOP_BY_METRIC) || (STOP_BY_METRIC && bMetric)) &&  depth < (max_bounce_with_cv + add_offset) && (1)){
            if((INCIDENT_NRC && depth < max_bounce_with_cv && !path.isDelta()) || (!INCIDENT_NRC && depth < max_bounce_with_cv)){
            //if( depth < 3 && (sdPrev.linearRoughness >= roughnessThreshold)){


                float addProb = 1.0f;
                if (!STOP_BY_METRIC) {
                    if (depth == 1) {
                        addProb = 1.0;
                    }
                    if (depth == 2) {
                        addProb = 0.5;
                    }
                    if (depth == 3) {
                        addProb = 0.15;
                    }
                }
                bool add_train = addU <= addProb;
                add_train = true;
                uint trainingID = 0;

                TrainingAdditionalData d;
                if (1 && add_train)
                {
                    bAlreadyAddedNIRC = true;
                    #if VARIANCE_LOSS_MODE
                        #if ESTIMATOR_NETWORK
                            XInferenceData predictionNetworkEstimater = constructPredictData(sdLastSurfaceForVariance);
                            trainingID =  addToPredBufferWithoutMapping(predictionNetworkEstimater);
                            gPredBufferThroughput[trainingID] = -pathThpPrev*predAlbedo;
                        #else
                            trainingID = gLossVarianceScatterID.IncrementCounter();
                        #endif
                    #endif


                    #if INCIDENT_NRC && TRAIN_DIRECT && SUPPORT_NEE && 1
                        {
                            d.att = pathLPrev;

                            // check if light sampling generated a correct sample
                            if(EnvShadowRay.rayParams.w == 1.0){
                                // Light Sampling Strategy

                                uint pathIDTrainingTemp = gLThpSelfTrainingBuffer.IncrementCounter();
                                float3 toLightDir = EnvShadowRay.rayParams.xyz;
                                float4 toLightThp;
                                toLightThp = float4(emissiveThp, 1.0f);

                                XInferenceData predData = encodeData(sdPrev, sdPrev, toLightDir, toLightThp);

                                #if USE_GUIDANCE_OF_ENV_MAP
                                toLightThp.xyz *= gScene.envMap.eval(toLightDir);
                                #endif

                                uint tid = addToXTrainBuffer(predData);
                                d.thp = toLightThp.xyz;
                                d.lightPathID = pathIDTrainingTemp;

                                #if VARIANCE_LOSS_MODE
                                    d.thp *= -1.0f;
                                #endif

                                LThp data;

                                data.L = (EnvShadowRay.Lr > 0.0) ? emissiveLight*emissiveThp : 0.0;
                                data.thp = outPredictionThp.xyz;

                                if (!relyOnNRCForLastVertex || !bIsSurfaceShadingDataLoaded)
                                {
                                    data.thp = 0;
                                }


                                gLThpSelfTrainingBuffer[pathIDTrainingTemp] = data;

                                //emissiveMISWeight = 1.0f;
                                d.pdf = emissiveMISWeight;
                                gXTrainAdditionalBuffer[tid] = d;

                                #if VARIANCE_LOSS_MODE
                                    gLossVarianceScatterID[tid] = trainingID;
                                    gLossVarianceThroughput[tid] = d.thp;
                                #endif


                            }

                            path.L -= EnvShadowRay.Lr;
                            d.lightPathID = pathIDTraining;
                        }

                        if(true)
                        {
                            // BRDF Sampling Strategy

                            // We came here through BRDF sampling. The other sampling strategy is
                            // env map sampling. Evaluate it's probability for the current ray dir.

                            float lightPdf;
                            if(all(emissiveTriangleHit.normalW) == 0.0){
                                // We came here through BRDF sampling. The other sampling strategy is
                                // env map sampling. Evaluate it's probability for the current ray dir.
                                lightPdf = gData.envMapSampler.evalPdf(path.dir) * getEnvLightSelectionPdf();
                            }
                            else{
                                    // Emissive Light
                                lightPdf = gData.emissiveSampler.evalPdf(sdPrev.posW, sdPrev.N, true, emissiveTriangleHit)*getEmissiveLightSelectionPdf()*getEmissiveLightSelectionPdf();
                            }

                            // Compute MIS weighted contribution from the environment map.
                            float misWeight = evalMIS(gData.params, 1, path.pdf, kLightSamplesPerVertex, lightPdf);
                            path.L /= misWeight;

                            //misWeight = 1.0f;
                            float4 toLightThp = float4(path.thp, 0.0f);
                            //toLightThp *= misWeight;



                            XInferenceData predData = encodeData(sdPrev, sdPrev, path.dir, toLightThp);

                            #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                            toLightThp.xyz *= gScene.envMap.eval(path.dir);
                            #endif

                            uint tid = addToXTrainBuffer(predData);

                            d.thp = toLightThp.xyz;

                            #if VARIANCE_LOSS_MODE
                                d.thp *= -1.0f;
                            #endif
                            d.pdf = misWeight;

                            gXTrainAdditionalBuffer[tid] = d;

                            #if VARIANCE_LOSS_MODE
                                gLossVarianceScatterID[tid] = trainingID;
                                gLossVarianceThroughput[tid] = d.thp;
                            #endif
                        }

                    #else


                            XInferenceData predData = encodeData(sdPrev, sdPrev, path.dir, d.thp);
                            uint tid = addToXTrainBuffer(predData);

                            d.lightPathID = pathIDTraining;
                            d.sub = 0.0f;
                            #if INCIDENT_NRC
                                #if TRAIN_DIRECT
                                    d.att = pathLPrev;
                                #else
                                    d.att = path.L- ((!kUseEnvLightNEE) ? envLight : 0.0f);
                                #endif
                                d.thp = path.thp;

                            #else
#if TRAIN_DIRECT
                            d.att = pathLPrev + sdPrev.emissive * pathThpPrev;
#else
                            d.att = path.L - ((!kUseEnvLightNEE) ? envLight : 0.0f);
#endif
                                d.thp = pathThpPrev*predAlbedo;
                            #endif

                            d.thpDiv = pathThpPrev;

                            //d.thpDiv = 1.0f;

                            #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                                d.thp *= gScene.envMap.eval(path.dir);
                            #endif

                            #if VARIANCE_LOSS_MODE
                                d.thp *= -1.0f;
                            #endif
                            d.pdf = 1.0f;

                            gXTrainAdditionalBuffer[tid] = d;

                            #if VARIANCE_LOSS_MODE
                                gLossVarianceScatterID[tid] = trainingID;
                                gLossVarianceThroughput[tid] = d.thp;
                            #endif




                            #if RES_VARIANCE_LOSS
                                gLossVarianceThroughput[tid] = -d.thp;


                                // Request for Residual Error Expected Value
                                XInferenceData filterRequestData = constructPredictData(sdPrev);
                                addToPredBufferWithoutMappingByID(filterRequestData, tid);
                                gPredBufferThroughput[tid] = -pathThpPrev*predAlbedo;

                                d.thp = pathThpPrev*predAlbedo;
                                d.att = path.L- ((!kUseEnvLightNEE) ? envLight : 0.0f);
                                 addToEstimatorNNXTrainBufferByID(filterRequestData, tid);
                                gEstimatorNNXTrainAdditionalBuffer[tid] = d;
                            #endif
                #endif




                }

#if VARIANCE_LOSS_MODE
                if(NUM_VARIANCE_LOSS_SAMPLES != 0 ){
                int c = 0;

                PathData tmpPath = pathLastSuraceForVariance;

                TriangleLightHit dirTriangleHitPrev;
                do{
                    SampleGenerator sgcopy = tmpPath.sg;
                      tmpPath = pathLastSuraceForVariance;
                    tmpPath.sg = sgcopy;


                    ShadingData sdTmp = sdLastSurfaceForVariance;
                    float3 rayOriginTmp = sdTmp.computeNewRayOrigin();
                    sdTmp.setActiveLobes(l);

                    #if !RENDER_SPEC
                    if(sdTmp.metallic > 0.5f && 0){
                        sdTmp.diffuse = 0.1;
                        sdTmp.metallic = 0.0f;
                    }
                    #endif

#if !RENDER_SPEC
                    sd.linearRoughness = 1.0f;
                    //sd.specular = 0.01;
                    //l &= ~(uint)LobeType::Specular;
#endif
#if MAKE_GLOSSY
                    //sd.linearRoughness = 0.4;
#endif
                    #if DISABLE_NORMALS
                        sdTmp.N = sdTmp.faceN;
                    #endif

                    if (!(kRayFootprintMode == TexLODMode::Mip0))
                    {
                        // Create the ray footprint data for TexLOD.
                        tmpPath.rayFootprint = RayFootprint.create(hit.getTriangleHit(), launchIndex, launchDim, rayOriginTmp, gScene.camera.getPosition(), sdTmp.faceN, sdTmp.N, gData.params.screenSpacePixelSpreadAngle, sdTmp.linearRoughness, tmpPath.isSpecular());
                    }

                    bool bKeepTracingTmp = true;

                    float3 faceN = sdTmp.faceN;
#if INCIDENT_NRC
#if TRAIN_DIRECT && SUPPORT_NEE && 1
                    // MIS
                    if(c % 2 == 0){
                        // Light Sampling Strategy
                        ShadowRay shadowRay = {};
                        bool realValid;
                        float3 Le;
                        float misV;
                        float3 LThp;
                        generateShadowRay(gData.params, gData.envMapSampler, gData.emissiveSampler, sdTmp, gData.standardMaterial, 0, tmpPath, tmpPath.sg, shadowRay, dirTriangleHitPrev, Le,misV, LThp, realValid, true);
                        bKeepTracingTmp = realValid;
                        if(bKeepTracingTmp){
                            tmpPath.dir = shadowRay.rayParams.xyz;
                            tmpPath.thp = shadowRay.Lr;
                        }
                    }
                    else{
                        // BRDF Sampling Strategy
                        bKeepTracingTmp = tracePathDirectCache(gData, sdTmp, tmpPath);

                        if(bKeepTracingTmp){
                            // We came here through BRDF sampling. The other sampling strategy is
                            // env map sampling. Evaluate it's probability for the current ray dir.
                            float lightPdf;

                            if(all(dirTriangleHitPrev.normalW) == 0.0){
                                // We came here through BRDF sampling. The other sampling strategy is
                                // env map sampling. Evaluate it's probability for the current ray dir.
                                lightPdf = gData.envMapSampler.evalPdf(tmpPath.dir) * getEnvLightSelectionPdf();
                            }
                            else{
                                // Emissive Light
                                lightPdf = evalTrianglePdf(sd.posW, dirTriangleHitPrev)*getEmissiveLightSelectionPdf();
                            }


                            // Compute MIS weighted contribution from the environment map.
                            float misWeight = evalMIS(gData.params, 1, tmpPath.pdf, kLightSamplesPerVertex, lightPdf);
                            tmpPath.thp *= misWeight;
                        }


                    }
#else
                    bKeepTracingTmp = tracePathDirectCache(gData, sdTmp, tmpPath);
#endif

#else
                    bKeepTracingTmp = tracePathCache(gData, sdTmp, tmpPath, true);
#endif

                    #if NORMALS_HACK
                    bKeepTracingTmp =  bKeepTracingTmp && (dot(faceN, path.dir) > 0.0);
                    #endif
                    c += 1;

                    if(!bKeepTracingTmp)
                        continue;





                    XInferenceData predData = encodeData(sdLastSurfaceForVariance, sdTmp, tmpPath.dir, tmpPath.thp);

                    uint tid = addToXTrainBuffer(predData);
                    d.thp = tmpPath.thp;
                    #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                    d.thp *= gScene.envMap.eval(tmpPath.dir);
                    #endif


                    #if INCIDENT_NRC && TRAIN_DIRECT && SUPPORT_NEE && 1
                    d.thp /= (NUM_VARIANCE_LOSS_SAMPLES/2);
                    #else
                    d.thp /= (NUM_VARIANCE_LOSS_SAMPLES);
                    #endif

                    d.pdf = 1.0f;

                    gXTrainAdditionalBuffer[tid] = d;
                    gLossVarianceScatterID[tid] = trainingID;
                    gLossVarianceThroughput[tid] = d.thp;
                }while(c < NUM_VARIANCE_LOSS_SAMPLES);

                }
#endif
            }

            depth += 1;


            if((relyOnNRCForLastVertex) && depth == GTNumIter - 1){
                bExecuteSelfLearn = true;
                sdLastSurface = sdLastSurfaceForVariance;
                pathLastSurace = path;

                // have to make one more iteration for taking into account direct incident light by path-tracing in a case if our cache doesn't include incident direct lighting
                if(TRAIN_DIRECT)
                    break;
                break;
            }

            if(depth == GTNumIter){
                break;
            }



            // if(depth == 1){
            //     break;
            // }

        } while (bKeepTracing);


        if(bExecuteSelfLearn && bKeepTracing){
            uint posID = gPredBuffer1.IncrementCounter();
            float3 dum = 0.0;

            outPredictionThp = float4(path.thp, 0);
            float3 thpc = outPredictionThp.xyz;
            #if FILTERING_NET
                XInferenceData predData = constructPredictData(sd);
                thpc *= (sd.diffuse+sd.specular);
            #else
                //XInferenceData predData = encodeData(sdLastSurface, sdLastSurface, path.dir, dum);
                XInferenceData predData = constructPredictData(sd);
                thpc *= (sd.diffuse+sd.specular);
            #endif

            gPredBuffer1[posID] = predData.Position;
            gPredBuffer0[posID] = (f32tof16(predData.SurfaceNormal.x) << 16) | f32tof16(predData.SurfaceNormal.y);
            gPredBuffer2[posID] = predData.Roughness;
            gPredBuffer3[posID] = predData.Color;


            depth++;

            #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                thpc.xyz *= gScene.envMap.eval(path.dir);
            #endif

            uint tid = addToPredBuffer(predData, posID);
            gPredBufferPixelID[tid] = pathIDTraining;
            gPredBufferThroughput[tid] = thpc.xyz;
        }
#else

float ac = 0.01;

        bool keepDo = true;
#if  ENABLE_NRC && (DEBUG_RENDER || DEBUG_VIEW) && (VARIANCE_LOSS_MODE || FILTERING_NET || !INCIDENT_NRC || RES_VARIANCE_LOSS) && !HEMISPHERICAL_RENDER
        bool doDEBUG = false;

            prediction = constructPredictData(sd);
            uint tid = addToPredBufferWithoutMapping(prediction);
            gPredBufferPixelID[tid] = launchIndex.x + launchIndex.y * launchDim.x;
            gPredBufferThroughput[tid] = sd.diffuse + sd.specular;
            outPredictionLAdd = float4(sd.emissive, 0.0f);

            keepDo = false;

#else

        sd.setActiveLobes(l);
        #if !RENDER_SPEC
                    if(sd.metallic > 0.5f && 0){
                    sd.diffuse = 0.1;
                    sd.metallic = 0.0f;
               }

        #endif
#if MAKE_GLOSSY
                    sd.linearRoughness = 0.25;
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
                    if(sd.metallic > 0.5f && 0){
                    sd.diffuse = 0.1;
                    sd.metallic = 0.0f;
                }

            #endif


        depth = 0;
        const float a0_b = length(gScene.camera.getPosition()-sd.posW)/(4*3.14*dot(sd.V, sd.N));
        float a_current  = 0;
        logPathLength(1);

        if(true && keepDo){

#if HEMISPHERICAL_RENDER
            uint2 coord = launchIndex;
            #if SPHERICAL_MAPPING
            //coord.x = (launchIndex.x+launchDim.x/2 )% launchDim.x;
            #endif
            float2 sphereCoordStart = float2(coord)/float2(launchDim-1);

            float kaf = 0.25;
            sphereCoordStart.x = sphereCoordStart.x*(1.0-2*kaf)+kaf;
            float2 sphereCoordStep = 1.0/float2(launchDim-1);
            float2 sphereCoordSampled = sampleNext2D(path.sg)*sphereCoordStep+sphereCoordStart;

            float3 scatterDir;
            #if SPHERICAL_MAPPING
                float tem = sphereCoordSampled.x;
                // sphereCoordSampled.x = sphereCoordSampled.y;
                // sphereCoordSampled.y = tem;

                // sphereCoordSampled.x += 0.5;
                // if(sphereCoordSampled.x > 1.0)
                //     sphereCoordSampled.x  -= 1.0f;
                sphereCoordSampled.y = 1.0-sphereCoordSampled.y;


                scatterDir = latlong_map_to_world_our(sphereCoordSampled);
            #else

                scatterDir = oct_to_ndir_snorm(sphereCoordSampled*2-1.0);
            #endif
            scatterDir = sd.fromLocalFace(scatterDir);
            path.thp = 1.0f;
            path.dir = scatterDir;
            path.pdf = 1.0f;
            path.L = 0.0;
#endif
            bool bNeuralCacheBeenUsed = false;
            int shift = 0;
            #if (!DEBUG_VIEW) || !ENABLE_NRC
            do
            {

                //sd.setActiveLobes((uint)LobeType::Diffuse);
                //sd.linearRoughness = 0.6f;
                sd.setActiveLobes(l);
#if !RENDER_SPEC
                sd.linearRoughness = 1.0f;
                //sd.specular = 0.01;
                //l &= ~(uint)LobeType::Specular;
#endif
                #if !RENDER_SPEC
                if(sd.metallic > 0.5f && 0){
                    sd.diffuse = 0.1;
                    if(sd.metallic > 0.5f && 0){
                        sd.diffuse = 0.1;
                        sd.metallic = 0.0f;
                    }
                }
#if MAKE_GLOSSY
                sd.linearRoughness = 0.05;
#endif


                #endif
                #if DISABLE_NORMALS
                    sd.N = sd.faceN;
                #endif
                ShadingData sdVBufer = sd;
                PathData VPath = path;


                #if HEMISPHERICAL_RENDER && INCIDENT_NRC
                if(depth == 0 && launchIndex.x == 0 && launchIndex.y == 0){
                    float3 dum = 0.0;
                    XInferenceData predDataIncident = encodeData(sdVBufer, sdVBufer, float3(1.0f, 1.0f, 1.0f), dum);
                    gPredBuffer1.IncrementCounter();
                    gPredBuffer1[0] = saturate((sdVBufer.posW - bbStart) / (bbEnd - bbStart));
                    gPredBuffer0[0] = (f32tof16(predDataIncident.SurfaceNormal.x) << 16) | f32tof16(predDataIncident.SurfaceNormal.y);
                    gPredBuffer2[0] = predDataIncident.Roughness;
                    gPredBuffer3[0] = predDataIncident.Color;
                }
                #endif

                #if HEMISPHERICAL_RENDER
                if(depth == 0 && dot(path.dir, sd.N) <= 0.0){
                    outPredictionL = float4(path.L, 1.0); // just take into account
                    outPredictionThp = float4(path.thp, 0);
                    break;
                }
                #endif


                bool bUSENeuralCache = false;
                uint posID = 0;
                bool shouldRequest = true;


                float3 prevDir = path.dir;

                SampleGenerator tmpSG = SampleGenerator.create(launchIndex/(8, 8), frameSeed * 234);
                float sampleV = sampleNext1D(tmpSG);

                bool bSkipsIsWorking = bNeuralCacheBeenUsed && ENABLE_NRC && !(BIASED && !HEMISPHERICAL_RENDER)  && !HEMISPHERICAL_RENDER;
                bool bSkipPathTracing = ((bSkipsIsWorking && gRoulleteProb != 0.0) && sampleV < gRoulleteProb) ? true : false;
                outPredictionL = float4(path.L, 1.0); // just take into account

                outPredictionThp = float4(path.thp, 0);




                // if(bNeuralCacheBeenUsed && ENABLE_NRC && BIASED && !HEMISPHERICAL_RENDER && (RENDER_ONLY_INDIRECT && depth == 0))
                //     break;



                float3 faceN = sd.faceN;
                // Scatter ray to a random direction, estimate updated throughup. Extract prediction data based on intersected surface.
                bool successSample = true;

                ShadowRay EnvShadowRay;
                TriangleLightHit emissiveTriangleHit;
                float3 emissiveLight;
                float emissiveMISWeight;
                float3 emissiveThp;
                float3 envLight = 0;
                float3 thpTempIntermediate = path.thp;

                bool passErrorTest = true;
#if DO_ERROR_TEST
                if (depth == 0) {
                    float3 yhat = ourEstimations[launchIndex].xyz;
                    float3 y = gtEstimations[launchIndex].xyz;
                    float3 error = abs(y - yhat) / (lum(yhat) + 0.000000000);
                    passErrorTest = all(error < errorThreshold);
                    //passErrorTest = false;
                }

#endif

#if !INCIDENT_NRC
                if (depth == 0)
                    passErrorTest &= sd.linearRoughness > 0.1;
#endif


                bool evalDirect = true;
                bool bforceNRC = false;

                bforceNRC = ENABLE_NRC && !HEMISPHERICAL_RENDER && BIASED && !INCIDENT_NRC && ERROR_BASED_METRIC && passErrorTest && depth == 0;
                if(!HEMISPHERICAL_RENDER && EVAL_DIRECT_ONLY_ONCE && depth == 0 && sampleIdx != 0){
                    evalDirect = false;
                }

                if (bforceNRC && TRAIN_DIRECT) {
                    evalDirect = false;
                }

                bUSENeuralCache = (((BIASED && !HEMISPHERICAL_RENDER) && g_num_samples[0] != 0) || (depth- shift < 3 && g_num_samples[depth- shift] != 0 && (depth- shift < max_bounce_with_cv))) && ENABLE_NRC  && ((HEMISPHERICAL_RENDER && depth == 0) || !HEMISPHERICAL_RENDER);

                if (!bforceNRC || !TRAIN_DIRECT ) {
                    if (bUSENeuralCache && ENABLE_NRC && !HEMISPHERICAL_RENDER && (bSkipPathTracing || (BIASED && !HEMISPHERICAL_RENDER))) {
                        bKeepTracing = tracePathSimple(gData, sd, path, depth, aN, rayOrigin, EnvShadowRay, emissiveTriangleHit, emissiveLight, emissiveMISWeight, emissiveThp, envLight, successSample, thpTempIntermediate, (RENDER_ONLY_INDIRECT && depth == 0) || !evalDirect, false, HEMISPHERICAL_RENDER && depth == 0, true);
                    }
                    else {
                        bKeepTracing = tracePathSimple(gData, sd, path, depth, aN, rayOrigin, EnvShadowRay, emissiveTriangleHit, emissiveLight, emissiveMISWeight, emissiveThp, envLight, successSample, thpTempIntermediate, (RENDER_ONLY_INDIRECT && depth == 0) || !evalDirect, false, HEMISPHERICAL_RENDER && depth == 0, false, (bSkipsIsWorking) ? 1.0 / (1.0 - gRoulleteProb) : 1.0);
                    }
                }




                VPath.sg = path.sg;
                bool bUseBSDFSampleForCache = true;
#if !INCIDENT_NRC && TRAIN_DIRECT
                a_current += sqrt(length(sd.posW - sdVBufer.posW) / (abs(dot(sdVBufer.N, sdVBufer.V)) * path.pdf));
#endif

                if ((BIASED && !HEMISPHERICAL_RENDER) && STOP_BY_METRIC) {

                    bool bMetric = (a_current * a_current > ac * a0_b) && bKeepTracing;

                    if (INCIDENT_NRC) {

                        if (bMetric) {
                            //if (path.isDelta())
                            //    bMetric = false;
                        }
                        else {
                            //bool bMetric2 = depth > 0 && a_current * a_current > 0.01 * a0_b;

#if STOP_BY_OUR_METRIC



                            if (!path.isDelta() && bKeepTracing && passErrorTest) {
                                // Veach balance heuristic
                                float Nf = (float)g_num_samples[0];
                                
                                float pdf_f = 1.0 / PI;
                                float pdf_g = path.pdf;

                                float t = Nf * pdf_f / (Nf * pdf_f + pdf_g);

                                float r = sampleNext1D(path.sg);
                                if (r < t || 1) {
                                    bMetric = true;
                                    bUseBSDFSampleForCache = false;
                                }

                            }
#endif
                        }
                        if (ERROR_BASED_METRIC && passErrorTest && depth == 0) {
                            if (path.isDelta()) {
                                bMetric = false;
                            }
                            else {
                                bMetric = true;
                                bUseBSDFSampleForCache = false;
                            }

                        }
                    }

                    bUSENeuralCache = bUSENeuralCache && bMetric;
                }


#if INCIDENT_NRC || !TRAIN_DIRECT

                a_current += sqrt(length(sd.posW - sdVBufer.posW) / (abs(dot(sdVBufer.N, sdVBufer.V)) * path.pdf));
#endif



                /* a_current += sqrt(length(sd.posW - sdVBufer.posW)/(abs(dot(sdVBufer.N, sdVBufer.V))*path.pdf));
                 if((BIASED && !HEMISPHERICAL_RENDER) && STOP_BY_METRIC){
                     bool bMetric = a_current * a_current > 0.01 * a0_b || !bKeepTracing;
                     if (bMetric && INCIDENT_NRC) {
                         if(path.isDelta() && depth == 0)
                             bMetric = false;
                     }

                     bUSENeuralCache = bUSENeuralCache && bMetric;
                 }*/

                /*if (bUSENeuralCache && !BIASED && ENABLE_NRC && !HEMISPHERICAL_RENDER && path.isDelta() && !STOP_BY_METRIC) {
                    shift = 1;
                    bUSENeuralCache = false;
                }*/




                bUSENeuralCache = bUSENeuralCache && (any(thpTempIntermediate) > 0.0f);

                if (!HEMISPHERICAL_RENDER && ENABLE_NRC && BIASED && !kUseEnvLightNEE && bUSENeuralCache && INCIDENT_NRC) {
                    path.L -= envLight;
                }

                if (bforceNRC) {
                    bUSENeuralCache = true;
                    if( TRAIN_DIRECT)
                    path.L = sdVBufer.emissive;
                }



                if(bUSENeuralCache || (HEMISPHERICAL_RENDER && depth == 0 && INCIDENT_NRC) ) {
                    bNeuralCacheBeenUsed = true;
                    #if HEMISPHERICAL_RENDER && INCIDENT_NRC
                    posID = 0;
                    shouldRequest = false;
                    #endif

                    #if HEMISPHERICAL_RENDER
                    const static int num_samples = 1;
                    #else
                    //int num_samples =  ((BIASED && !HEMISPHERICAL_RENDER )) ? g_num_samples[0] : g_num_samples[depth];


                    #if STOP_BY_METRIC
                    int num_samples = g_num_samples[0];
                    #else
                    int num_samples = g_num_samples[depth- shift];
                    #endif


                    #endif

                    #if 0
                    if(depth == 0)
                        num_samples = 0;
                    #endif


                    float3 dum = 0.0;
                    #if INCIDENT_NRC
                        XInferenceData predDataIncident = encodeData(sdVBufer, sdVBufer, float3(1.0f, 1.0f, 1.0f), dum);
                        if(shouldRequest){
                            posID = gPredBuffer1.IncrementCounter();
                            gPredBuffer1[posID] = predDataIncident.Position;
                            gPredBuffer0[posID] = (f32tof16(predDataIncident.SurfaceNormal.x) << 16) | f32tof16(predDataIncident.SurfaceNormal.y);
                            gPredBuffer2[posID] = predDataIncident.Roughness;
                            gPredBuffer3[posID] = predDataIncident.Color;

                            shouldRequest = false;
                        }
                    #endif


#if 0

                    TriangleLightHit dirTriangleHitPrev;
                    dirTriangleHitPrev.normalW = 0.0f;
                PathData pathcopy = path;
                    [unroll]
                    for(uint k=0; k < num_samples; k++)
                    {
                        PathData tmpPath = pathcopy;

                        tmpPath.sg = SampleGenerator.create(launchIndex, frameSeed * 29*num_samples+k+1);

                        ShadingData sdTmp = sdVBufer;
                        float3 rayOriginTmp = sdTmp.computeNewRayOrigin();
                        sdTmp.setActiveLobes(l);
                        #if !RENDER_SPEC
                            if(sdTmp.metallic > 0.5f && 0){
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
                            tmpPath.rayFootprint = RayFootprint.create(hit.getTriangleHit(), launchIndex, launchDim, rayOriginTmp, gScene.camera.getPosition(), sdTmp.faceN, sdTmp.N, gData.params.screenSpacePixelSpreadAngle, sdTmp.linearRoughness, tmpPath.isSpecular());
                        }


                        bool bKeepTracingTmp = true;


                        ////print(sd.diffuse);
                        ////print(sd.posW);
                        ////print(rayOrigin);
                        float3 faceN = sdTmp.faceN;

                        // Scatter ray to a random direction, estimate updated throughup. Extract prediction data based on intersected surface.
                        bool succesSample = true;
                        #if INCIDENT_NRC
                            #if !HEMISPHERICAL_RENDER
                                #if TRAIN_DIRECT && SUPPORT_NEE && 1
                                    // MIS

                                    if(k % 2 == 0){
                                        // Light Sampling Strategy
                                        ShadowRay shadowRay = {};

                                        bool realvalid;
                                        float3 Le;
                                        float misV;
                                        float3 LThp;
                                        generateShadowRay(gData.params, gData.envMapSampler, gData.emissiveSampler, sdTmp, gData.standardMaterial, 0, tmpPath, tmpPath.sg, shadowRay, dirTriangleHitPrev, Le, misV, LThp, realvalid, true);
                                        bKeepTracingTmp = realvalid;


                                        if(bKeepTracingTmp ){
                                            tmpPath.dir = shadowRay.rayParams.xyz;
                                            tmpPath.thp = misV*LThp;
                                        }
                                    }
                                    else{
                                        // BRDF Sampling Strategy
                                        bKeepTracingTmp = tracePathDirectCache(gData, sdTmp, tmpPath);


                                        if(bKeepTracingTmp){

                                            float lightPdf;
                                            // if(all(dirTriangleHitPrev.normalW) == 0.0){
                                            //     // We came here through BRDF sampling. The other sampling strategy is
                                            //     // env map sampling. Evaluate it's probability for the current ray dir.
                                            lightPdf = gData.envMapSampler.evalPdf(tmpPath.dir) * getEnvLightSelectionPdf();
                                            // }
                                            // else{
                                            //     lightPdf = evalTrianglePdf(sdTmp.posW, dirTriangleHitPrev)*getEmissiveLightSelectionPdf();
                                            //     // Emissive Light
                                            // }

                                            // Compute MIS weighted contribution from the environment map.
                                            float misWeight = evalMIS(gData.params, 1, tmpPath.pdf, 1, lightPdf);
                                            tmpPath.thp *= misWeight;
                                        }


                                    }
                                #else
                                    bKeepTracingTmp = tracePathDirectCache(gData, sdVBufer, tmpPath);
                                #endif
                            #endif
                        #else
                            float3 diff = tmpPath.L;
                            bKeepTracingTmp = tracePathCache(gData, sdTmp, tmpPath, !HEMISPHERICAL_RENDER);

                            pathcopy.L += tmpPath.L-diff;
                        #endif

                        #if NORMALS_HACK
                        bKeepTracingTmp = bKeepTracingTmp && (dot(faceN, tmpPath.dir) > 0.0);
                        #endif

                        if(!succesSample || !bKeepTracingTmp){
                            continue;
                        }

                        XInferenceData predData = encodeData(sdVBufer, sdTmp, tmpPath.dir, tmpPath.thp);

                        #if !INCIDENT_NRC
                        posID = gPredBuffer1.IncrementCounter();
                        gPredBuffer1[posID] = saturate((sdTmp.posW - bbStart) / (bbEnd - bbStart));
                        gPredBuffer0[posID] = (f32tof16(predData.SurfaceNormal.x) << 16) | f32tof16(predData.SurfaceNormal.y);
                        gPredBuffer2[posID] = predData.Roughness;
                        gPredBuffer3[posID] = predData.Color;
                        #endif

                        uint tid = addToPredBuffer(predData, posID);
                        ////print(tid);


                        #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP && !ENV_MAP_SAMPLING_IS_OPTIMIZED_FOR_INFERENCE
                        tmpPath.thp *= gScene.envMap.eval(tmpPath.dir);
                        #endif

                        gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;

                        #if INCIDENT_NRC && TRAIN_DIRECT && SUPPORT_NEE && !HEMISPHERICAL_RENDER
                        float num_div = 2.0/(num_samples);
                        #else
                        float num_div = 1.0/num_samples;
                        #endif

                        // if(kUseMIS){
                        //     num_div = 0;
                        // }

                        gPredBufferThroughput[tid] = tmpPath.thp.xyz*num_div;
                    }

                    SampleGenerator sgcopy = path.sg;
                    path = pathcopy;
                    path.sg = sgcopy;

                    sd = sdVBufer;
                    rayOrigin = sd.computeNewRayOrigin();
#else
                    XInferenceData predDataTmp = encodeData(sdVBufer, sdVBufer, float3(1.0f, 1.0f, 1.0f), dum);
                    [unroll]
                    for(uint k=0; k < num_samples; k++)
                    {


                        bool bKeepTracingTmp = true;


                        ////print(sd.diffuse);
                        ////print(sd.posW);
                        ////print(rayOrigin);
                        float3 faceN = sdVBufer.faceN;
                        float3 thp  = thpTempIntermediate;
                        float3 dir;

                        #if INCIDENT_NRC
                                if(k==0)
                                    logPathLength((depth + 1));
                            #if !HEMISPHERICAL_RENDER

                                BSDFSample result;
                               /* if (!bUseBSDFSampleForCache) {
                                    uint all = (uint)LobeType::All;
                                    uint dt = (uint)LobeType::Delta;
                                    all = all & (~(dt));
                                    sdVBufer.setActiveLobes(all);
                                }*/
                                bKeepTracingTmp = gData.standardMaterial.sample(sdVBufer, path.sg, result, kUseBRDFSampling);
                                dir = result.wo;
                                thp *= result.weight;

                                #if NORMALS_HACK
                                bKeepTracingTmp = bKeepTracingTmp && (dot(faceN, result.wo) > 0.0);
                                #endif

                                bKeepTracingTmp &= any(thp > 0.f);

        #if 1
                                if(!bKeepTracingTmp){
                                    continue;
                                }
        #else
                                if(!bKeepTracingTmp){
                                    thp = 0;
                                }
        #endif
                            #else
                                dir = prevDir;
                            #endif
                            predDataTmp.Direction = ndir_to_oct_snorm(dir);
                        #else

                            ShadingData tmpSD = sdVBufer;
                            PathData tmpPath = VPath;
                            if(k != 0){
                                bKeepTracingTmp = tracePathCache(gData, tmpSD, tmpPath, !HEMISPHERICAL_RENDER);
                                //path.L += tmpPath.L-diff;

                                #if NORMALS_HACK
                                    bKeepTracingTmp = bKeepTracingTmp && (dot(faceN, tmpPath.dir) > 0.0);
                                #endif

                                path.sg = tmpPath.sg;
                            }
                            else{
                                bKeepTracingTmp = bKeepTracing;
                                if (bforceNRC) {
                                    tmpPath = VPath;
                                    tmpSD = sdVBufer;
                                    if (bKeepTracingTmp || bforceNRC)
                                        logPathLength(depth + 1);
                                }
                                else {
                                    if (TRAIN_DIRECT) {
                                        tmpPath = path;
                                        tmpSD = sd;

                                        if (bKeepTracingTmp || bforceNRC)
                                            logPathLength((depth+2));
                                    }
                                    else {
                                        bKeepTracingTmp = true;
                                        tmpPath = VPath;
                                        tmpSD = sdVBufer;
                                        if (bKeepTracingTmp || bforceNRC)
                                            logPathLength((depth + 1));
                                    }
                                }
                            }

                                if(!bKeepTracingTmp && !bforceNRC){
                                    continue;
                                }

                                predDataTmp = constructPredictData(tmpSD);
                                thp = tmpPath.thp*(tmpSD.diffuse+tmpSD.specular);

                                posID = gPredBuffer1.IncrementCounter();
                                gPredBuffer1[posID] = predDataTmp.Position;
                                gPredBuffer0[posID] = (f32tof16(predDataTmp.SurfaceNormal.x) << 16) | f32tof16(predDataTmp.SurfaceNormal.y);
                                gPredBuffer2[posID] = predDataTmp.Roughness;
                                gPredBuffer3[posID] = predDataTmp.Color;


                        #endif

                        uint tid = addToPredBuffer(predDataTmp, posID);
                        ////print(tid);

                        #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP && !ENV_MAP_SAMPLING_IS_OPTIMIZED_FOR_INFERENCE
                        thp *= gScene.envMap.eval(dir);
                        #endif

                        gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;

                        #if INCIDENT_NRC && TRAIN_DIRECT && SUPPORT_NEE && !HEMISPHERICAL_RENDER
                        float num_div = 2.0/(num_samples);
                        #else
                        float num_div = 1.0/num_samples;
                        #endif
                        gPredBufferThroughput[tid] = thp*num_div;
                    }

                    //sd = sdVBufer;
                    //rayOrigin = sd.computeNewRayOrigin();
#endif


                }


                if(EVAL_DIRECT_ONLY_ONCE && depth == 0 && sampleIdx == 0 ){
                    outPredictionLAdd = float4((path.L - envLight), 0.0f);
                    path.L = envLight;
                }




                // if(dot(faceN, path.dir) <= 0.0)
                //     bKeepTracing = false;
                #if NORMALS_HACK
                successSample = successSample && (dot(faceN, path.dir) > 0.0);
                #endif

                outPredictionL = float4(path.L, 1.0); // just take into account
                outPredictionThp = float4(path.thp, 0);

                #if INCIDENT_NRC && !HEMISPHERICAL_RENDER && TRAIN_DIRECT && SUPPORT_NEE && 1

                #else
                if(!successSample)
                    break;
                #endif

                #if !INCIDENT_NRC
                if(!bKeepTracing)
                    break;
                #endif

                if(bSkipPathTracing)
                    break;

                #if RENDER_ONLY_INDIRECT && !HEMISPHERICAL_RENDER
                if(depth == 0){
                    path.L = 0;
                }
                #endif


                if (successSample && bUSENeuralCache && !(BIASED && !HEMISPHERICAL_RENDER) && !HEMISPHERICAL_RENDER && 1)
                {
                    #if INCIDENT_NRC && TRAIN_DIRECT && SUPPORT_NEE && 1
                        {
                            if(EnvShadowRay.rayParams.w == 1.0f){
                                // Light Sampling Strategy
                                float3 toLightDir = EnvShadowRay.rayParams.xyz;
                                float4 toLightThp = float4(emissiveThp*emissiveMISWeight, 1.0f);
                                XInferenceData predData = encodeData(sdVBufer, sdVBufer, toLightDir, toLightThp);

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

                                float lightPdf;
                                // if(all(emissiveTriangleHit.normalW) == 0.0){
                                //     // We came here through BRDF sampling. The other sampling strategy is
                                //     // env map sampling. Evaluate it's probability for the current ray dir.
                                    lightPdf = gData.envMapSampler.evalPdf(path.dir) * getEnvLightSelectionPdf();
                                // }
                                // else{
                                //     // Emissive Light
                                //     lightPdf = gData.emissiveSampler.evalPdf(sdVBufer.posW, sdVBufer.N, true, emissiveTriangleHit)*getEmissiveLightSelectionPdf();
                                // }

                                // Compute MIS weighted contribution from the environment map.
                                float misWeight = evalMIS(gData.params, 1, path.pdf, kLightSamplesPerVertex, lightPdf);
                                float4 toLightThp = float4(path.thp, 0.0f);
                                toLightThp *= misWeight;

                                XInferenceData predData = encodeData(sdVBufer, sdVBufer, path.dir, toLightThp);

                                #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP
                                toLightThp.xyz *= gScene.envMap.eval(path.dir);
                                #endif

                                uint tid = addToPredBuffer(predData, posID);
                                residualErrorID = tid;

                                gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;
                                gPredBufferThroughput[tid] = -toLightThp.xyz; // NEGATIVE FOR ESTIMATING ERROR
                            }
                        }

                    #else
                        XInferenceData predData = encodeData(sdVBufer, sd, path.dir, outPredictionThp);

                        #if INCIDENT_NRC && TRAIN_DIRECT && USE_GUIDANCE_OF_ENV_MAP && !ENV_MAP_SAMPLING_IS_OPTIMIZED_FOR_INFERENCE
                        outPredictionThp.xyz *= gScene.envMap.eval(path.dir);
                        #endif

                        if(shouldRequest){
                            posID = gPredBuffer1.IncrementCounter();
                            gPredBuffer1[posID] = predData.Position;
                            gPredBuffer0[posID] = (f32tof16(predData.SurfaceNormal.x) << 16) | f32tof16(predData.SurfaceNormal.y);
                            gPredBuffer2[posID] = predData.Roughness;
                            gPredBuffer3[posID] = predData.Color;
                        }

                        uint tid = addToPredBuffer(predData, posID);
                        residualErrorID = tid;

                        gPredBufferPixelID[tid] = launchIndex.x+launchIndex.y*launchDim.x;
                        gPredBufferThroughput[tid] = -outPredictionThp.xyz; // NEGATIVE FOR ESTIMATING ERROR
                    #endif
                }

                depth++;



                if(bNeuralCacheBeenUsed && ENABLE_NRC && (BIASED && !HEMISPHERICAL_RENDER) && !HEMISPHERICAL_RENDER)
                    break;





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


        // Accumulate after clamping.
        // Note the comparison is written so that NaNs propagate (unless the compiler rewrites it).
        // TODO: Check the generated code that this is the case.
        outColor += gData.params.clampSamples && path.L > gData.params.clampThreshold ? gData.params.clampThreshold : path.L;
        outAlbedo += sdOriginalBufer.diffuse+sdOriginalBufer.specular;
        }

        // We're done accumulating over all samples.
        const float invSpp = 1.f / samples_per_pixel;
        outColor *= invSpp;
        outAlbedo *= invSpp;
        outAlpha = 1.f;

        outPredictionL.xyz = outColor;
        outPredictionL.w = 1.0f;
    }
    else
    {
        backL.xyz = evalBackground(-sd.V);
        backL.w = 1.0f;

        backL = 1000.0;
        //outPredictionL = 1000.0f;
        // Background pixel.
        outColor = evalBackground(-sd.V);

        outColor = 1000.0;
        outAlbedo = outColor.rgb;
        outAlpha = kForceAlphaOne ? 1.f : 0.f;
    }



    assert(!any(isnan(outColor)));

    // Write outputs.
    // These are all optional so using compile-time checks to decide which ones to write.

    if (is_valid(gOutputAlbedo))
        gOutputAlbedo[launchIndex] = float4(outAlbedoValue, 1);

    if (is_valid(gOutputNormal))
        gOutputNormal[launchIndex] = float4(outNormalValue * 0.5 + 0.5, 1);

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
    #if VARIANCE_LOSS_MODE && !ESTIMATOR_NETWORK
        float4 expectedValue = gExpectedValueTmp[launchIndex];
        outPredictionL -= expectedValue;
    #endif

    LThp data;
    if(bResetTrainingLight)
        outPredictionL.xyz = 0;
    data.L = outPredictionL.xyz;
    data.thp = outPredictionThp.xyz;



    if (!relyOnNRCForLastVertex || !bIsSurfaceShadingDataLoaded)
    {
        data.thp = 0;
    }


    gLThpSelfTrainingBuffer[pathIDTraining] = data;

    launchDim /= train_dim_window;
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
    // if(ENABLE_NRC && BIASED){

    //    outPredictionL = 0;
    // }
    #endif

    outPredictionL += backL;

    #if !HEMISPHERICAL_RENDER
    float cd = length(int2(launchIndex.xy)-debug_pixel_id);
    if(debug_pixel_id.x != -1 && cd <= 9 && cd >= 7 && 0)
        outPredictionL.xyz += 10*float3(1.0, 1.0, 0.0f);
    #endif

    gOutputColor[launchIndex] = outPredictionL+outPredictionLAdd;
    gOutputM2[launchIndex] = outPredictionL*outPredictionL;


    #if HEMISPHERICAL_RENDER

    #endif



#endif
    //gOutputColor[launchIndex] = 0;
}
