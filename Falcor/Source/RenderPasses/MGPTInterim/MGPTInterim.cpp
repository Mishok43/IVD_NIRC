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
#include "MGPTInterim.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Scene/HitInfo.h"
#include <sstream>
#include "json/json.hpp"
#include "RenderPasses/Shared/NeuralNetwork/NeuralStructures.slang"
namespace
{
    const char kShaderFile[] = "RenderPasses/MGPTInterim/PathTracer.rt.slang";
    const char kParameterBlockName[] = "gData";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    // The payload for the scatter rays is 8-12B.
    // The payload for the shadow rays is 4B.
    const uint32_t kMaxPayloadSizeBytes = HitInfo::kMaxPackedSizeInBytes;
    const uint32_t kMaxAttributeSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;
    const std::string kPyramidConfig = "RenderPasses/DummyNeuralNetwork/pyramid_config.json";

    // Render pass output channels.
    const std::string kColorOutput = "color";
    const std::string kAlbedoOutput = "albedo";
    const std::string kTimeOutput = "time";
        
    const Falcor::ChannelList kOutputChannels =
    {
        { kColorOutput,     "gOutputColor",               "Output color (linear)", true /* optional */                              },
        { kAlbedoOutput,    "gOutputAlbedo",              "Surface albedo (base color) or background color", true /* optional */    },
        { kTimeOutput,      "gOutputTime",                "Per-pixel execution time", true /* optional */, ResourceFormat::R32Uint  }
    };

    const char* kOutputBufferChannels[] = 
    { 
        "gPredBuffer0",
        "gPredBuffer1",
        "gPredBuffer2",
        "gPredBufferMapping1",

        "gPredBufferPixelID",
        "gPredBufferThroughput",    
        "gPredBufferOutActAddData",

        "gXTrainBuffer0",
        "gXTrainBuffer1",
        "gXTrainBuffer2",
        "gXTrainResErrorID",
        "gResErrorID",
        "gLThpSelfTrainingBuffer", 
        "gXTrainAdditionalBuffer",

        // Mapping of idNN to index in X Buffers
        "gNNMapPredBufferKeys",
        "gNNMapPredBufferValues",
        "gNNMapTrainBufferKeys",
        "gNNMapTrainBufferValues",
    };
};

const char* MGPTInterim::sDesc = "Megakernel path tracer for NRC (interim version)";

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerClass("MGPTInterim", MGPTInterim::sDesc, MGPTInterim::create);
}

MGPTInterim::SharedPtr MGPTInterim::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new MGPTInterim(dict));
}

MGPTInterim::MGPTInterim(const Dictionary& dict)
    : PathTracer(dict, kOutputChannels)
{
    mNumNRCSamples = 3;
    mNumQLearningSamples = 3;
    bNRCBiased = false;
    bDirectLightCache = false;
    bNRCEnabled = true;
    bNRCDebugMode = false;
    bOnlyIndirect = true;
    bQLearning = false;
    bIncidentNRC = true;

    bDispatchTrainingRays = true;

    std::ifstream networkConfigStream;
    std::string networkConfigFullPath;

    std::string dummy_path = Falcor::getWorkingDirectory();
    dummy_path = dummy_path.substr(0, dummy_path.find_last_of('\\') + 1);

    //std::string dummy_path = "C:/Users/devmi/Documents/Projects/MSUNeuralRendering/ibvfteh-GMLNeuralRendering/Falcor/Source/";
    //dummy_path = dummy_path.substr(0, dummy_path.find_last_of('\\') + 1);

    if (findFileInShaderDirectories(dummy_path+kPyramidConfig, networkConfigFullPath)) {
        networkConfigStream.open(networkConfigFullPath);
    }
    else {
        throw std::runtime_error("Config for " + dummy_path + kPyramidConfig + " not found");
    }

    nlohmann::json config;
    networkConfigStream >> config;
    networkConfigStream.close();

    m_LOD = config["numLODs"] + 1;
    lodSizes.push_back(1);

    bGradientDenoising = false;
    for (uint32_t i = 0; i < m_LOD - 1; i++) {
        lodSizes.push_back(config["lodSize"][i]);
    }

    if (config["gradient_denoising"]["enable"]) {
        bGradientDenoising = true;
        gradientHashGridRes = config["gradient_denoising"]["hash_grid_res"];
        gradientHashGridSize = config["gradient_denoising"]["hash_grid_size"];
    }
    else {
        gradientHashGridRes = 256;
        gradientHashGridSize = 1024;
    }

    maxBounceWithCV = mSharedParams.maxBounces;

}

void MGPTInterim::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    PathTracer::setScene(pRenderContext, pScene);

    if (mpScene)
    {
        if (mpScene->hasGeometryType(Scene::GeometryType::Procedural))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        // Create ray tracing program.
        RtProgram::Desc desc;
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(kMaxAttributeSizeBytes);
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
        desc.addDefines(mpScene->getSceneDefines());
        desc.addDefine("MAX_BOUNCES", std::to_string(mSharedParams.maxBounces));
        desc.addDefine("SAMPLES_PER_PIXEL", std::to_string(mSharedParams.samplesPerPixel));

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(kRayTypeScatter, desc.addMiss("scatterMiss"));
        sbt->setMiss(kRayTypeShadow, desc.addMiss("shadowMiss"));
        sbt->setHitGroupByType(kRayTypeScatter, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("scatterClosestHit", "scatterAnyHit"));
        sbt->setHitGroupByType(kRayTypeShadow, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("", "shadowAnyHit"));

        desc.addDefine("CONTINUE_RAY", "0");
        mTracer.pProgram = RtProgram::create(desc);
    }
}


void MGPTInterim::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Call shared pre-render code.
    if (!beginFrame(pRenderContext, renderData)) return;

    // Set compile-time constants.
    RtProgram::SharedPtr pProgram = mTracer.pProgram;
    setStaticParams(pProgram.get());

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    pProgram->addDefines(getValidResourceDefines(mInputChannels, renderData));
    pProgram->addDefines(getValidResourceDefines(mOutputChannels, renderData));

    if (mUseEmissiveSampler)
    {
        // Specialize program for the current emissive light sampler options.
        assert(mpEmissiveSampler);
        if (pProgram->addDefines(mpEmissiveSampler->getDefines())) mTracer.pVars = nullptr;
    }


    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars) prepareVars();
    assert(mTracer.pVars);

    // Set shared data into parameter block.
    setTracerData(renderData);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            auto var = mTracer.pVars->getRootVar();
            var[desc.texname] = renderData[desc.name]->asTexture();
        }
    };
    for (auto channel : mInputChannels) bind(channel);
    for (auto channel : mOutputChannels) bind(channel);

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);
    mScreenDim = targetDim;

    for (auto buffChannel : kOutputBufferChannels)
    {
        mTracer.pVars[buffChannel] = renderData[buffChannel]->asBuffer();
    }

    pRenderContext->clearUAVCounter(renderData["gXTrainBuffer0"]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData["gPredBuffer0"]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData["gPredBuffer1"]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData["gQLearningPredBuffer0"]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData["gQLearningPredBuffer1"]->asBuffer(), 0);

    mpPixelDebug->prepareProgram(pProgram, mTracer.pVars->getRootVar());
    mpPixelStats->prepareProgram(pProgram, mTracer.pVars->getRootVar());
    pProgram->addDefine("MLOD", std::to_string(m_LOD));
    pProgram->addDefine("GRADIENT_DENOISING", std::to_string(bGradientDenoising));
    pProgram->addDefine("CONTINUE_RAY", "0");
    pProgram->addDefine("NUM_NRC_SAMPLES", std::to_string(mNumNRCSamples));
    pProgram->addDefine("RENDER_ONLY_INDIRECT", std::to_string(bOnlyIndirect));
    pProgram->addDefine("ENABLE_NRC", std::to_string(bNRCEnabled));
    pProgram->addDefine("DEBUG_VIEW", std::to_string(bNRCDebugMode));
    pProgram->addDefine("BIASED", std::to_string(bNRCBiased));
    pProgram->addDefine("INCIDENT_NRC", std::to_string(bIncidentNRC));
    pProgram->addDefine("TRAIN_DIRECT", std::to_string(bDirectLightCache));

    mTracer.pVars["GeneralData"]["bbStart"] = mpScene->getSceneBounds().minPoint;
    mTracer.pVars["GeneralData"]["bbEnd"] = mpScene->getSceneBounds().maxPoint;

    mTracer.pVars["GeneralData"]["max_bounce_with_cv"] = maxBounceWithCV;
    mTracer.pVars["GeneralData"]["gd_hash_grid_res"] = gradientHashGridRes;
    mTracer.pVars["GeneralData"]["gd_hash_grid_size"] = gradientHashGridSize;


    bool prevBRDFSampling = mSharedParams.useBRDFSampling;

    for (uint32_t i = 0; i < lodSizes.size(); i++)
    {
        mTracer.pVars["GeneralData"]["lodSize"+std::to_string(i)] = lodSizes[i];
    }
    // Spawn the rays.
    {
        
        PROFILE("MegakernelPathTracer::execute()_RayTrace - Predicting");
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));
    }

    mSharedParams.useBRDFSampling = false;
    setTracerData(renderData);
    pProgram->addDefine("CONTINUE_RAY", "1");
    pProgram->addDefine("Q_LEARNING", std::to_string(bQLearning));
    pProgram->addDefine("NUM_NRC_SAMPLES", std::to_string(mNumQLearningSamples));
    

    // Spawn the rays.
    if(bDispatchTrainingRays)
    {
        mTracer.pVars["gPredBuffer0"] = renderData["gQLearningPredBuffer0"]->asBuffer();
        mTracer.pVars["gPredBuffer1"] = renderData["gQLearningPredBuffer1"]->asBuffer();
        mTracer.pVars["gPredBuffer2"] = renderData["gQLearningPredBuffer2"]->asBuffer();
        mTracer.pVars["gPredBufferMapping1"] = renderData["gQLearningPredBufferMapping1"]->asBuffer();
        mTracer.pVars["gPredBufferPixelID"] = renderData["gQLearningMapping"]->asBuffer();
        mTracer.pVars["gPredBufferThroughput"] = renderData["gQLearningThroughput"]->asBuffer();

        PROFILE("MegakernelPathTracer::execute()_RayTrace - Training");
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim.x / 8, targetDim.y / 4, 1));
    }

    mSharedParams.useBRDFSampling = prevBRDFSampling;

    // Call shared post-render code.
    endFrame(pRenderContext, renderData);
}

void MGPTInterim::renderUI(Gui::Widgets& widget)
{
    if (auto nnGroup = widget.group("NeuralNetwork", true))
    {
        widget.checkbox("Dispatch Training Rays", bDispatchTrainingRays);
        widget.var("Max bounce guided by CV", maxBounceWithCV, uint32_t(0), uint32_t(5), 1);
        widget.checkbox("Enable NRC", bNRCEnabled);
        widget.checkbox("Debug NRC", bNRCDebugMode);
        widget.checkbox("Biased NRC", bNRCBiased);
        widget.checkbox("Direct Light Cache", bDirectLightCache);
        widget.checkbox("Only Indirect", bOnlyIndirect);
        widget.checkbox("Incident NRC", bIncidentNRC);
        widget.var("Number NRC Samples", mNumNRCSamples);

        widget.checkbox("QLearning", bQLearning);
        widget.var("Number QLearning NRC Samples", mNumQLearningSamples);
    }

    PathTracer::renderUI(widget);
}

RenderPassReflection MGPTInterim::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector = PathTracer::reflect(compileData);

    uint32_t dims = (mScreenDim.x / 8) * (mScreenDim.y / 4) * 3;

    uint32_t num_bounces = 2;

    uint32_t numPredElementsNotPadded = mScreenDim.x * mScreenDim.y;
    numPredElementsNotPadded = (numPredElementsNotPadded + numPredElementsNotPadded / (8 * 4));

    uint32_t resolutionPerBounds = mScreenDim.x * mScreenDim.y * 8;
    resolutionPerBounds = (resolutionPerBounds + 127)/128*128;

    uint32_t resolutionPerBoundsBase = mScreenDim.x * mScreenDim.y * 2;
    resolutionPerBoundsBase = (resolutionPerBoundsBase + 127) / 128 * 128;

    reflector.addOutput("gPredBuffer0", "Prediction buffer").structuredBuffer(resolutionPerBounds, 4).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gPredBuffer1", "Prediction buffer").structuredBuffer(resolutionPerBoundsBase, sizeof(float3)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gPredBuffer2", "Prediction buffer").structuredBuffer(resolutionPerBounds, 4).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));

    uint32_t selfLearningDims = (dims + 127) / 128 * 128;

    reflector.addOutput("gQLearningPredBuffer0", "Prediction buffer").structuredBuffer(selfLearningDims, 4).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gQLearningPredBuffer1", "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(float3)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gQLearningPredBuffer2", "Prediction buffer").structuredBuffer(selfLearningDims, 4).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));

    reflector.addOutput("gQLearningPredBufferMapping1", "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(uint32_t)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gQLearningMapping", "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(uint32_t)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gQLearningThroughput", "gQLearningThroughput").structuredBuffer(selfLearningDims, sizeof(float3)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));

    reflector.addOutput("gXTrainBuffer0", "X Training buffer").structuredBuffer(dims, 4).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gXTrainBuffer1", "X Training buffer").structuredBuffer(dims, sizeof(float3)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gXTrainBuffer2", "X Training buffer").structuredBuffer(dims, 4).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));

    reflector.addOutput("gXTrainResErrorID", "X Training buffer").structuredBuffer(dims, sizeof(int)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gResErrorID", "X Training buffer").structuredBuffer(dims, sizeof(float4)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));

    reflector.addOutput("gPredBufferMapping1", "Prediction buffer").structuredBuffer(resolutionPerBounds, sizeof(uint32_t)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gPredBufferPixelID", "gPrefBufferPixelID").structuredBuffer(resolutionPerBounds, sizeof(uint)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gPredBufferThroughput", "gPrefBufferThroughput").structuredBuffer(resolutionPerBounds, sizeof(float3)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    
    reflector.addOutput("gLThpSelfTrainingBuffer", "L and thp Self Training buffer").structuredBuffer(dims, sizeof(LThp)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gXTrainAdditionalBuffer", "X Additional Training buffer").structuredBuffer(dims, sizeof(TrainingAdditionalData)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));

    uint s = (m_LOD == 1) ? 1 : numPredElementsNotPadded * (m_LOD - 1);
    reflector.addOutput("gNNMapPredBufferKeys", "Map sample index to NN for Prediction Keys").structuredBuffer(4, sizeof(int)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gNNMapPredBufferValues", "Map sample index to NN for Prediction Values").structuredBuffer(4, sizeof(int)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gNNMapTrainBufferKeys", "Map sample index to NN for Train Keys").structuredBuffer(dims * m_LOD, sizeof(int)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addOutput("gNNMapTrainBufferValues", "Map sample index to NN for Train Values").structuredBuffer(dims * m_LOD, sizeof(int)).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    
    return reflector;
}

void MGPTInterim::prepareVars()
{
    assert(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    bool success = mpSampleGenerator->setShaderData(var);
    if (!success) throw std::exception("Failed to bind sample generator");

    // Create parameter block for shared data.
    ProgramReflection::SharedConstPtr pReflection = mTracer.pProgram->getReflector();
    ParameterBlockReflection::SharedConstPtr pBlockReflection = pReflection->getParameterBlock(kParameterBlockName);
    assert(pBlockReflection);
    mTracer.pParameterBlock = ParameterBlock::create(pBlockReflection);
    assert(mTracer.pParameterBlock);

    // Bind static resources to the parameter block here. No need to rebind them every frame if they don't change.
    // Bind the light probe if one is loaded.
    if (mpEnvMapSampler) mpEnvMapSampler->setShaderData(mTracer.pParameterBlock["envMapSampler"]);

    // Bind the parameter block to the global program variables.
    mTracer.pVars->setParameterBlock(kParameterBlockName, mTracer.pParameterBlock);
}

void MGPTInterim::setTracerData(const RenderData& renderData)
{
    auto pBlock = mTracer.pParameterBlock;
    assert(pBlock);

    // Upload parameters struct.
    pBlock["params"].setBlob(mSharedParams);

    // Bind emissive light sampler.
    if (mUseEmissiveSampler)
    {
        assert(mpEmissiveSampler);
        bool success = mpEmissiveSampler->setShaderData(pBlock["emissiveSampler"]);
        if (!success) throw std::exception("Failed to bind emissive light sampler");
    }
}
