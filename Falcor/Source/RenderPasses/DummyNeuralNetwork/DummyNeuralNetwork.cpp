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
#include "FalcorCUDA.h"
#include "NNLaplacian.h"
#include "DummyNeuralNetwork.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/NeuralNetworks/NeuralNetworkModel.h"
#include "Scene/HitInfo.h"
#include "json/json.hpp"
#include "cub/cub.cuh"
#include "tiny-cuda-nn/common.h"
#include <tiny-cuda-nn/common_device.h>
#include "NNLaplacian.h"
#include "RenderPasses/Shared/NeuralNetwork/NeuralStructures.slang"



#define DEMO 0

namespace
{

    const char kShaderFile[] = "RenderPasses/DummyNeuralNetwork/PathTracer.rt.slang";
    const char kParameterBlockName[] = "gData";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    // The payload for the scatter rays is 8-12B.
    // The payload for the shadow rays is 4B.
    const uint32_t kMaxPayloadSizeBytes = Falcor::HitInfo::kMaxPackedSizeInBytes;
    const uint32_t kMaxAttributeSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;

    const Falcor::uint2 DEBUG_RES = Falcor::uint2(256, 256);

    const std::string kIsNRCEnable = "IsNRCEnable";
    const std::string kIsBiased = "IsBiased";
    const std::string kNumNRCSamples = "NumNRCSamples";

    const std::string kMullerHeuristic = "MullerHeuristic";
    const std::string kOurHeuristic = "OurHeuristic";
    const std::string kErrorBasedHeuristic = "ErrorBasedHeuristic";
    const std::string kDoErrorTest = "DoErrorTest";
    const std::string kErrorThreshold = "ErrorThreshold";

    const std::string kIsIncidentCache = "IsIncidentCache";
    const std::string kDebugHemispherical = "DebugHemispherical";
    const std::string kQLearning = "QLearning";
    const std::string kNumGTSamples = "NumGTSamples";

    // Render pass output channels.
    const std::string kColorOutput = "color";
    const std::string kAlbedoOutput = "albedo";
    const std::string kNormalOutput = "normal";
    const std::string kTimeOutput = "time";

    const Falcor::ChannelList kOutputChannels =
    {

        { kAlbedoOutput,    "gOutputAlbedo",              "Surface albedo (base color) or background color", true /* optional */   },
        { kNormalOutput,    "gOutputNormal",              "Normal output", true /* optional */    },
        { kTimeOutput,      "gOutputTime",                "Per-pixel execution time", true /* optional */, Falcor::ResourceFormat::R32Uint  }
    };

    const char kDesc[] = "Insert pass description here";

    // Note: Path is now determined dynamically at runtime
    // const std::string addPath = "D://GradientDiffusion//gradientdiffusion//Falcor//Source//";

    const std::string kColorOutputFinal = "colorOutputFinal";

    const std::string kColor2MomentOutput = "color2MomentOutput";
    const std::string kColorOutputDebug = "colorOutputDebug";
    const std::string kColorOutputDebugGT = "colorOutputDebugGT";
    const std::string kPredictedRadianceMerged = "gPredictedRadianceMerged";
    const std::string kConvertBufferToTexture = "RenderPasses/DummyNeuralNetwork/buffToTexDynamicReal.cs.slang";
    const std::string kConvertBufferToTextureFused = "RenderPasses/DummyNeuralNetwork/buffToTexDynamicRealFused.cs.slang";
    const std::string kFinalOutput = "RenderPasses/DummyNeuralNetwork/combine.cs.slang";
    const std::string kEnvMapRequest = "RenderPasses/DummyNeuralNetwork/envMapRequest.cs.slang";
    const std::string kGatherOurEstimations = "RenderPasses/DummyNeuralNetwork/gatherOurEstimations.cs.slang";

    const std::string kYTrainBufferAssembly = "RenderPasses/DummyNeuralNetwork/assembler.cs.slang";

    const std::string kNetworkConfigNIRC = "RenderPasses/DummyNeuralNetwork/network_config_nirc.json";
    const std::string kNetworkConfigNIRCEnvMap = "RenderPasses/DummyNeuralNetwork/network_config_nirc_env_map.json";
    const std::string kNetworkConfigNRC = "RenderPasses/DummyNeuralNetwork/network_config_nrc.json";
    const std::string kNetworkConfigEstimator = "RenderPasses/DummyNeuralNetwork/network_estimator_config.json";

    const std::string kPyramidConfig = "RenderPasses/DummyNeuralNetwork/pyramid_config.json";

    const std::string kNetworkEstimatorPredBuffer = "gNetworkEstimatorPredBuffer";

    const std::string kNetworkEstimatorThroughput = "gNetworkEstimatorThroughput";

    const std::string kQLearningPredBuffer = "gQLearningPredBuffer";
    const std::string kQLearningPredBufferMapping = "gQLearningPredBufferMapping";
    const std::string kQLearningMapping = "gQLearningMapping";
    const std::string kQLearningThroughput = "gQLearningThroughput";

    const std::string kResErrorID = "gResErrorID";
    const std::string kXTrainResError = "gXTrainResErrorID";
    const std::string kPredBuffer = "gPredBuffer";
    const std::string kPredBufferMapping = "gPredBufferMapping";
    const std::string kPredPixelID = "gPredBufferPixelID";
    const std::string kPredThroughput = "gPredBufferThroughput";

    const std::string kPredBufferOutActAddData = "gPredBufferOutActAddData";
    const std::string kXTrainBuffer = "gXTrainBuffer";
    const std::string kPredictedRadianceSTYBuffer = "gPredictedRadianceSTYBuffer";
    const std::string kLThpSelfTrainingBuffer = "gLThpSelfTrainingBuffer";
    const std::string kXTrainAdditionalBuffer = "gXTrainAdditionalBuffer";
    const std::string kYTrainBuffer = "gYTrainBuffer";
    const std::string kExpectedValuesTmp = "gExpectedValueTmp";
    const std::string kLossVarianceScatterIDBuffer = "gLossVarianceScatterID";
    const std::string kLossVarianceThroughputBuffer = "gLossVarianceThroughput";
    const std::string kEstimatorNNTrainBuffer = "gEstimatorNNXTrainBuffer";
    const std::string kTrainPixel = "gTrainPixel";
    const std::string kOurEstimations = "ourEstimations";
    const std::string kGTEstimations = "gtEstimations";
    const std::string kEstimatorNNXTrainAdditionalBuffer = "gEstimatorNNXTrainAdditionalBuffer";


    // Mapping of idNN to index in X Buffers
    /*const std::string kNNMapPredBufferKeys = "gNNMapPredBufferKeys";
    const std::string kNNMapPredBufferValues = "gNNMapPredBufferValues";

    const std::string kNNMapTrainBufferKeys = "gNNMapTrainBufferKeys";
    const std::string kNNMapTrainBufferValues = "gNNMapTrainBufferValues";*/
}

//using namespace Falcor;

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}


namespace Falcor {

    static void regNNPass(pybind11::module& m)
    {
        pybind11::class_<DummyNeuralNetwork, RenderPass, DummyNeuralNetwork::SharedPtr> type(m, "DummyNeuralNetwork");
        type.def_property_readonly("training_error", &Falcor::DummyNeuralNetwork::getTrainingError);
        type.def_property_readonly("training_epoch", &Falcor::DummyNeuralNetwork::getTrainingEpoch);
        type.def_property("diffuse_time", &Falcor::DummyNeuralNetwork::getDiffuseTime, &Falcor::DummyNeuralNetwork::setDiffuseTime);
        type.def("reloadModel", &Falcor::DummyNeuralNetwork::reloadModel);
    }
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("DummyNeuralNetwork", kDesc, Falcor::DummyNeuralNetwork::create);
    Falcor::ScriptBindings::registerBinding(Falcor::regNNPass);
}

Falcor::DummyNeuralNetwork::DummyNeuralNetwork(const Falcor::Dictionary& dict)
    : PathTracer(dict, kOutputChannels)
{
    mpConvertNNOutputPass = ComputePass::create(kConvertBufferToTexture);
    mpConvertNNOutputPassFused = ComputePass::create(kConvertBufferToTextureFused);
    mpGatherOutEstimations = ComputePass::create(kGatherOurEstimations);
    mpFinalOutputPass = ComputePass::create(kFinalOutput);
    mpYTrainAssemblyPass = ComputePass::create(kYTrainBufferAssembly);
    mpEnvMapRequests = ComputePass::create(kEnvMapRequest);

    uint32_t resolutionPerBounds = mScreenDim.x * mScreenDim.y * MAX_SAMPLES;
    resolutionPerBounds = (resolutionPerBounds + 255) / 256 * 256;

    uint32_t resolutionPerBoundsBase = mScreenDim.x * mScreenDim.y * 3;
    resolutionPerBoundsBase = (resolutionPerBoundsBase + 255) / 256 * 256;

#if 1
    mainRenderData = Falcor::NeuralRequestData("main", 3, resolutionPerBounds, resolutionPerBoundsBase, true);
    debugRenderData = Falcor::NeuralRequestData("debug", 3, DEBUG_RES.x * DEBUG_RES.y, -1, true);
#else
    // Supporting Direct NRC without recompiling by sacrificing some memory
    mainRenderData = Falcor::NeuralRequestData("main", 3, resolutionPerBounds, -1, true);
    debugRenderData = Falcor::NeuralRequestData("debug", 3, DEBUG_RES.x* DEBUG_RES.y, -1, true);
#endif

    bResidualVarianceLoss = false;

    mNumNRCSamplesPerDepth[0] = 4;
    mNumNRCSamplesPerDepth[1] = 4;
    mNumNRCSamplesPerDepth[2] = 0;

    mNumQLearningSamples = 3;
    bNRCBiased = false;
    bDirectLightCache = false;
    bResErrorForLoss = false;
    bDebugHemispherical = false;
    bNRCEnabled = true;
    bNRCDebugMode = false;
    bOnlyIndirect = false;
    bVarianceLoss = true;
    mNumVarianceLossSamples = 4;
    bQLearning = false;
    bIncidentNRC = true;

    bDispatchTrainingRays = true;
    maxBounceWithCV = 1;

    bDirectLightCacheNEE = false;

    bFusedOutput = true;
    mSharedParams.maxBounces = 5;
    mSharedParams.maxNonSpecularBounces = 5;
    roughnessThreshold = 0;
    mSharedParams.useEnvLightNEE = 0;

    bVarianceLoss = false;
    bDebugHemispherical = false;
    //bFilteringNetwork = true;
#if 1
    mSharedParams.useNEE  = 1;
#endif

#if 0
    bIncidentNRC = true;
    bVarianceLoss = true;
    mNumNRCSamples = 1;
    bOnlyIndirect = true;
    bUseNNAsEstimator = false;
#endif

#if 0
    mNumNRCSamples = 3;
    bDebugHemispherical = true;
    bDirectLightCacheNEE = false;
    bIncidentNRC = true;
    bDirectLightCache = true;
    bOnlyIndirect = false;
    //mNumVarianceLossSamples = 4;
    maxBounceWithCV = 1;
    mSharedParams.maxBounces = 1;
    mSharedParams.useNEE = 0;
    mSharedParams.maxNonSpecularBounces = 1;
#endif

    //bNRCEnabled = false;
    maxBounceWithCV = 2;
    bStopByMetric = 1;
    mNumNRCSamplesPerDepth[0] = 1;
    mNumNRCSamplesPerDepth[1] = 0;


#if 0
    bNRCBiased = true;
    bStopByMetric = false;
    bFusedOutput = false;
    bForceNonSeparateInput = true;
#endif

#if 0
    bStopByMetric = 0;
    bDebugHemispherical = true;
    bDirectLightCacheNEE = false;
    bIncidentNRC = true;
    bDirectLightCache = true;
    bOnlyIndirect = false;
    //mNumVarianceLossSamples = 4;
    maxBounceWithCV = 1;
    mSharedParams.maxBounces = 1;
    mSharedParams.useNEE = 0;
    mSharedParams.maxNonSpecularBounces = 1;
    bNRCBiased = true;
#endif

#if DEMO
    bStopByMetric = 1;
    bIncidentNRC = true;
    bNRCBiased = true;
    mSharedParams.maxBounces = 15;
    mSharedParams.maxNonSpecularBounces = 15;
    bOnlyIndirect = false;
    bDirectLightCache = false;
    maxBounceWithCV = 3;

    mNumNRCSamplesPerDepth[0] = 1;
    mNumNRCSamplesPerDepth[1] = 1;
    mNumNRCSamplesPerDepth[2] = 1;
#endif

#if 1
    mSharedParams.useEnvLightNEE = false;
    bStopByMetric = 1;
    bIncidentNRC = true;
    bNRCBiased = true;
    bOnlyIndirect = false;
    bDirectLightCache = false;
    maxBounceWithCV = 3;

    mNumNRCSamplesPerDepth[0] = 3;
    mNumNRCSamplesPerDepth[1] = 1;
    mNumNRCSamplesPerDepth[2] = 1;
#endif

    bStopByOurMetric = true;

    std::ifstream networkConfigStream;
    std::string networkConfigFullPath;

    std::string dummy_path = Falcor::getWorkingDirectory();
    dummy_path = dummy_path.substr(0, dummy_path.find_last_of('\\') + 1);

    // Note: Old hardcoded path removed for privacy

    if (findFileInShaderDirectories(dummy_path + kPyramidConfig, networkConfigFullPath)) {
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

}

Falcor::DummyNeuralNetwork::SharedPtr Falcor::DummyNeuralNetwork::create(Falcor::RenderContext* pRenderContext, const Falcor::Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new Falcor::DummyNeuralNetwork(dict));
    return pPass;
}

std::string Falcor::DummyNeuralNetwork::getDesc() { return kDesc; }

__declspec(dllexport) Falcor::Dictionary Falcor::DummyNeuralNetwork::getScriptingDictionary()
{
    Dictionary dict;

    dict[kIsNRCEnable] = bNRCEnabled;
    dict[kIsBiased] = bNRCBiased;
    dict[kNumNRCSamples] = mNumNRCSamplesPerDepth[0];
    dict[kIsIncidentCache] = bIncidentNRC;
    dict[kDebugHemispherical] = bDebugHemispherical;
    dict[kQLearning] = bQLearning;
    dict[kNumGTSamples] = mNumGTSamples;

    dict[kErrorThreshold] = errorThreashold;
    dict[kMullerHeuristic] = bStopByMetric;
    dict[kOurHeuristic] = bStopByOurMetric;
    dict[kDoErrorTest] = bErrorTest;
    dict[kErrorBasedHeuristic] = bErrorBased;



    return dict;
}


__declspec(dllexport) void Falcor::DummyNeuralNetwork::nrc_experiment_setup() {
    bIncidentNRC = false;
    bDirectLightCache = false;

    mNumNRCSamplesPerDepth[0] = 1;
    mNumNRCSamplesPerDepth[1] = 0;
    mNumNRCSamplesPerDepth[2] = 0;
}

__declspec(dllexport) void Falcor::DummyNeuralNetwork::nirc_experiment_setup(int samples) {
    bIncidentNRC = true;
    bDirectLightCache = false;

    mNumNRCSamplesPerDepth[0] = samples;
    mNumNRCSamplesPerDepth[1] = 0;
    mNumNRCSamplesPerDepth[2] = 0;
}

__declspec(dllexport) void Falcor::DummyNeuralNetwork::enableTraining(bool bEnable) {
    if(model)
        model->enableTraining(bEnable);

    if(modelEstimator)
        modelEstimator->enableTraining(bEnable);

    bDispatchTrainingRays = bEnable;
}

__declspec(dllexport) float Falcor::DummyNeuralNetwork::getAveragePath() const {
    PixelStats::Stats stats;
    mpPixelStats->getStats(stats);
    return stats.avgPathLength;
}

__declspec(dllexport) void Falcor::DummyNeuralNetwork::enableInference(bool bEnable) {
    if(model)
        model->enablePrediction(bEnable);

    if(modelEstimator)
        modelEstimator->enablePrediction(bEnable);
}

__declspec(dllexport) float Falcor::DummyNeuralNetwork::getLoss() const {
    return model->m_loss;
}

__declspec(dllexport) int Falcor::DummyNeuralNetwork::getTrainingStep() const {
    return model->mTrainingStep;
}

__declspec(dllexport) void Falcor::DummyNeuralNetwork::resetNN() {
    model->reloadModel();
    modelEstimator->reloadModel();
}

__declspec(dllexport) void Falcor::DummyNeuralNetwork::parseDictionary(const Falcor::Dictionary& dict) {
    for (const auto& [key, value] : dict)
    {
        if (key == kIsNRCEnable) bNRCEnabled = value;
        else if (key == kIsBiased) bNRCBiased = value;
        else if (key == kIsIncidentCache) bIncidentNRC = value;
        else if (key == kDebugHemispherical) bDebugHemispherical = value;
        else if (key == kQLearning) bQLearning = value;
        else if (key == kNumGTSamples) mNumGTSamples = value;
        else if (key == kErrorThreshold) errorThreashold = value;
        else if (key == kMullerHeuristic) bStopByMetric = value;
        else if (key == kOurHeuristic) bStopByOurMetric = value;
        else if (key == kDoErrorTest) bErrorTest = value;
        else if (key == kErrorBasedHeuristic) bErrorBased= value;
        //else if (key == kNumNRCSamples) mNumNRCSamplesPerDepth[0] = value; // TODO: Fix this
    }
}

void Falcor::DummyNeuralNetwork::prepareVars()
{
    assert(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = Falcor::RtProgramVars::create(mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    bool success = mpSampleGenerator->setShaderData(var);
    if (!success) throw std::exception("Failed to bind sample generator");

    // Create parameter block for shared data.
    Falcor::ProgramReflection::SharedConstPtr pReflection = mTracer.pProgram->getReflector();
    Falcor::ParameterBlockReflection::SharedConstPtr pBlockReflection = pReflection->getParameterBlock(kParameterBlockName);
    assert(pBlockReflection);
    mTracer.pParameterBlock = Falcor::ParameterBlock::create(pBlockReflection);
    assert(mTracer.pParameterBlock);

    // Bind static resources to the parameter block here. No need to rebind them every frame if they don't change.
    // Bind the light probe if one is loaded.
    if (mpEnvMapSampler) mpEnvMapSampler->setShaderData(mTracer.pParameterBlock["envMapSampler"]);

    // Bind the parameter block to the global program variables.
    mTracer.pVars->setParameterBlock(kParameterBlockName, mTracer.pParameterBlock);
}


bool Falcor::NeuralRequestData::allocateBuffers(Falcor::RenderPassReflection& reflector) {
    if (!bInit)
        return false;

    reflector.addInternal(kPredBuffer + "0"+name, "Prediction buffer").structuredBuffer(numUniqueElements, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kPredBuffer + "1" + name, "Prediction buffer").structuredBuffer(numUniqueElements, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kPredBuffer + "2" + name, "Prediction buffer").structuredBuffer(numUniqueElements, sizeof(float)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kPredBuffer + "3" + name, "Prediction buffer").structuredBuffer(numUniqueElements, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kPredBuffer + "4" + name, "Prediction buffer").structuredBuffer(numElements, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kPredBufferMapping + "1" + name, "Prediction buffer").structuredBuffer(numElements, sizeof(uint32_t)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kPredPixelID + name, "gPrefBufferPixelID").structuredBuffer(numElements, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kPredThroughput + name, "gPrefBufferThroughput").structuredBuffer(numElements, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));


    isChannelMapped.resize(5);
    if(!mpPredictionResults)
        mpPredictionResults = Falcor::Buffer::createStructured(3 * 4, numElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));

    return true;
}

void Falcor::NeuralRequestData::requestNumRequests(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData) {
    if (mpAtomicCounterCPURead == nullptr)
        mpAtomicCounterCPURead = Buffer::create(2 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, Buffer::CpuAccess::Read);

    pRenderContext->copyBufferRegion(mpAtomicCounterCPURead.get(), 0, renderData[kPredBuffer + "4" + name]->asBuffer()->getUAVCounter().get(), 0, 4);
    pRenderContext->copyBufferRegion(mpAtomicCounterCPURead.get(), 4, renderData[kPredBuffer + "1" + name]->asBuffer()->getUAVCounter().get(), 0, 4);
}


void Falcor::NeuralRequestData::updateNumActiveElements() {
    uint32_t* uavCounters = (uint32_t*)mpAtomicCounterCPURead->map(Buffer::MapType::Read);
    numElementsActive = uavCounters[0];
    numUniqueElementsActive = uavCounters[1];
    mpAtomicCounterCPURead->unmap();
}

bool Falcor::NeuralRequestData::bindBuffers(Falcor::RenderContext* pRenderContext, Falcor::RtProgramVars::SharedPtr vars, const Falcor::RenderData& renderData, Falcor::uint2 resolution) {
    if (!bInit)
        return false;

    if (mpRadianceMerge == nullptr) {
        mpRadianceMerge = Texture::create2D(resolution.x, resolution.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        res = resolution;
    }

    vars[kPredBuffer+"0"] = renderData[kPredBuffer + "0" + name]->asBuffer();
    vars[kPredBuffer + "1"] = renderData[kPredBuffer + "1" + name]->asBuffer();
    vars[kPredBuffer + "2"] = renderData[kPredBuffer + "2" + name]->asBuffer();
    vars[kPredBuffer + "3"] = renderData[kPredBuffer + "3" + name]->asBuffer();
    vars[kPredBuffer + "4"] = renderData[kPredBuffer + "4" + name]->asBuffer();


    if (isMapped()) {
        isChannelMapped[0] = true;
        isChannelMapped[1] = true;
        isChannelMapped[2] = true;
        isChannelMapped[3] = true;
        isChannelMapped[4] = false;
    }

    vars[kPredBufferMapping+"1"] = renderData[kPredBufferMapping + "1" + name]->asBuffer();


    vars[kPredPixelID] = renderData[kPredPixelID + name]->asBuffer();
    vars[kPredThroughput] = renderData[kPredThroughput + name]->asBuffer();

    pRenderContext->clearUAVCounter(renderData[kPredBuffer + "4"+name]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData[kPredBuffer + "1"+name]->asBuffer(), 0);
    return true;
}

Falcor::RenderPassReflection Falcor::DummyNeuralNetwork::reflect(const CompileData& compileData)
{
    // Define the required resources here
    Falcor::RenderPassReflection reflector = Falcor::PathTracer::reflect(compileData);

    uint32_t dims = (mScreenDim.x / 8) * (mScreenDim.y / 4) * 2;

    uint32_t num_bounces = 1;

    uint32_t numPredElementsNotPadded = mScreenDim.x * mScreenDim.y;
    numPredElementsNotPadded = (numPredElementsNotPadded + numPredElementsNotPadded / (8 * 4));

    uint32_t resolutionPerBounds = mScreenDim.x * mScreenDim.y * 4;
    resolutionPerBounds = (resolutionPerBounds + 255) / 256 * 256;

    uint32_t resolutionPerBoundsBase = mScreenDim.x * mScreenDim.y * 2;
    resolutionPerBoundsBase = (resolutionPerBoundsBase + 255) / 256 * 256;

    mainRenderData.allocateBuffers(reflector);
    debugRenderData.allocateBuffers(reflector);


    uint32_t selfLearningDims = (dims + 255) / 256 * 256;



    //reflector.addInternal(kFilterNetPredBuffer + "0", "Prediction buffer").structuredBuffer(dims, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //reflector.addInternal(kFilterNetPredBuffer + "1", "Prediction buffer").structuredBuffer(dims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //reflector.addInternal(kFilterNetPredBuffer + "2", "Prediction buffer").structuredBuffer(dims, sizeof(float)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //reflector.addInternal(kFilterNetPredBuffer + "3", "Prediction buffer").structuredBuffer(dims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //reflector.addInternal(kFilterNetPredBuffer + "4", "Prediction buffer").structuredBuffer(dims, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    //reflector.addInternal(kFilterNetPredThroughput, "gQLearningThroughput").structuredBuffer(dims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    const uint32_t max_num_loss_variance_samples = 0;
    const uint32_t num_training_number_multiplier = 1 + max_num_loss_variance_samples;

    reflector.addInternal(kXTrainBuffer + "0", "X Training buffer").structuredBuffer(dims*num_training_number_multiplier, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kXTrainBuffer + "1", "X Training buffer").structuredBuffer(dims*num_training_number_multiplier, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kXTrainBuffer + "2", "X Training buffer").structuredBuffer(dims*num_training_number_multiplier, sizeof(float)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kXTrainBuffer + "3", "X Training buffer").structuredBuffer(dims * num_training_number_multiplier, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kXTrainBuffer + "4", "X Training buffer").structuredBuffer(dims * num_training_number_multiplier, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    reflector.addInternal(kTrainPixel, "X Training buffer").structuredBuffer(dims, sizeof(uint2)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kLThpSelfTrainingBuffer, "L and thp Self Training buffer").structuredBuffer(dims, sizeof(LThp)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kXTrainAdditionalBuffer, "X Additional Training buffer").structuredBuffer(dims * num_training_number_multiplier, sizeof(TrainingAdditionalData)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));


#if 1
    dims = 1;
    selfLearningDims = 1;
#endif
    reflector.addInternal(kQLearningPredBuffer + "0", "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kQLearningPredBuffer + "1", "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kQLearningPredBuffer + "2", "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(float)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kQLearningPredBufferMapping + "1", "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(uint32_t)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kQLearningPredBuffer + "3", "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kQLearningPredBuffer + "4", "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    reflector.addInternal(kQLearningMapping, "Prediction buffer").structuredBuffer(selfLearningDims, sizeof(uint32_t)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kQLearningThroughput, "gQLearningThroughput").structuredBuffer(selfLearningDims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    reflector.addInternal(kNetworkEstimatorPredBuffer + "0", "Prediction buffer").structuredBuffer(dims, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kNetworkEstimatorPredBuffer + "1", "Prediction buffer").structuredBuffer(dims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kNetworkEstimatorPredBuffer + "2", "Prediction buffer").structuredBuffer(dims, sizeof(float)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kNetworkEstimatorPredBuffer + "3", "Prediction buffer").structuredBuffer(dims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kNetworkEstimatorPredBuffer + "4", "Prediction buffer").structuredBuffer(dims, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    reflector.addInternal(kNetworkEstimatorThroughput, "gQLearningThroughput").structuredBuffer(dims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    reflector.addInternal(kEstimatorNNXTrainAdditionalBuffer, "X Additional Training buffer").structuredBuffer(dims , sizeof(TrainingAdditionalData)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kEstimatorNNTrainBuffer+ "0", "X Training buffer").structuredBuffer(dims, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kEstimatorNNTrainBuffer + "1", "X Training buffer").structuredBuffer(dims , sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kEstimatorNNTrainBuffer + "2", "X Training buffer").structuredBuffer(dims , sizeof(float)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kEstimatorNNTrainBuffer + "3", "X Training buffer").structuredBuffer(dims, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kEstimatorNNTrainBuffer + "4", "X Training buffer").structuredBuffer(dims, sizeof(uint)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    /*reflector.addInternal(kXTrainResError, "X Training buffer").structuredBuffer(dims, sizeof(int)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kResErrorID, "X Training buffer").structuredBuffer(dims, sizeof(float4)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    */
    reflector.addInternal(kExpectedValuesTmp, "Expected Values Tmp").texture2D().format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource);

    reflector.addInternal(kLossVarianceScatterIDBuffer, "Loss Variance Scatter ID").structuredBuffer(dims* num_training_number_multiplier, sizeof(uint32_t)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kLossVarianceThroughputBuffer, "Loss Variance Throughput").structuredBuffer(dims* num_training_number_multiplier, sizeof(float3)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    uint s = (m_LOD == 1) ? 1 : numPredElementsNotPadded * (m_LOD - 1);
    /*reflector.addInternal(kNNMapPredBufferKeys, "Map sample index to NN for Prediction Keys").structuredBuffer(4, sizeof(int)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kNNMapPredBufferValues, "Map sample index to NN for Prediction Values").structuredBuffer(4, sizeof(int)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));*/
    /*reflector.addInternal(kNNMapTrainBufferKeys, "Map sample index to NN for Train Keys").structuredBuffer(dims * m_LOD, sizeof(int)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    reflector.addInternal(kNNMapTrainBufferValues, "Map sample index to NN for Train Values").structuredBuffer(dims * m_LOD, sizeof(int)).bindFlags((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));*/

    //reflector.addInternal(kColorOutput, "Output FrameBuffer Predicted by Neural Network").format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess);
    reflector.addOutput(kColorOutputFinal, "Output FrameBuffer Predicted by Neural Network").format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess);
    reflector.addOutput(kColor2MomentOutput, "Output 2nd Moment").format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess);

    reflector.addOutput(kColorOutputDebug, "Output FrameBuffer Predicted by Neural Network").texture2D(DEBUG_RES.x, DEBUG_RES.y).format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess);
    reflector.addOutput(kColorOutputDebugGT, "Output FrameBuffer Predicted by Neural Network").texture2D(DEBUG_RES.x, DEBUG_RES.y).format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess);

    reflector.addInput(kOurEstimations, "").texture2D().format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::ShaderResource);
    reflector.addInput(kGTEstimations, "").texture2D().format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::ShaderResource);

    //reflector.addOutput("dst");
    //reflector.addInput("src");
    return reflector;
}

void Falcor::DummyNeuralNetwork::setTracerData(const Falcor::RenderData& renderData)
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

void Falcor::DummyNeuralNetwork::setScene(Falcor::RenderContext* pRenderContext, const Falcor::Scene::SharedPtr& pScene)
{
    PathTracer::setScene(pRenderContext, pScene);

    mpScene = pScene;

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

Falcor::NeuralInputDataFalcor setupNeuralInput(const Falcor::RenderData& renderData, const std::string& predBufferName, uint32_t numPredElements, const std::string& predBufferMappingName = "", uint32_t numPredMappingElements = 0, std::string postFix = "", const std::vector<bool>* isMappedChannel=nullptr) {
    Falcor::NeuralInputDataFalcor neuralInputData;

    for (int i = 0; i < 5; i++) {
        const Falcor::Resource::SharedPtr& res = renderData.getResource(predBufferName + std::to_string(i) + postFix);
        if (res == nullptr)
            break;

        bool isForcedMapped = false;
        if (isMappedChannel) {
            isForcedMapped = (*isMappedChannel)[i];
        }

        if ((isMappedChannel && !isForcedMapped) || (!isMappedChannel && (predBufferMappingName.empty())) || numPredMappingElements == 0) {
            neuralInputData.addInput(res->asBuffer(), nullptr, numPredElements);
        }
        else {
            const Falcor::Resource::SharedPtr& resMap = renderData.getResource(predBufferMappingName + "1" + postFix);
            uint32_t numElements = (resMap == nullptr) ? numPredElements : numPredMappingElements;
            neuralInputData.addInput(res->asBuffer(), (resMap == nullptr) ? nullptr : resMap->asBuffer(), numElements);
        }
    }

    return neuralInputData;
}


#define SYNC_EVERYTHING 0
void Falcor::DummyNeuralNetwork::execute(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData)
{

    output = renderData[kColorOutputFinal]->asTexture();

    if (!mpScene)
    {
        return;
    }

    // Call shared pre-render code.
    if (!beginFrame(pRenderContext, renderData)) return;


    // Set compile-time constants.
    RtProgram::SharedPtr pProgram = mTracer.pProgram;

    /*if (bNRCEnabled)
        mSharedParams.samplesPerPixel = 1;
    else
        mSharedParams.samplesPerPixel = mNumGTSamples;*/


    bool bPrevNEE = mSharedParams.useNEE;
    if (bNRCEnabled) {
        /*if (!bDirectLightCache)
            mSharedParams.useNEE = bPrevNEE;
        else {
            mSharedParams.useNEE = bDirectLightCacheNEE;
        }*/
    }
    else {
        mSharedParams.useNEE = bPrevNEE;
    }

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


    const std::string buffersNames[] = {
        kXTrainBuffer + "0",
        kXTrainBuffer + "1",
        kXTrainBuffer + "2",
        kXTrainBuffer + "3",
        kXTrainBuffer + "4",
        /*kXTrainResError,
        kResErrorID,*/
        kLThpSelfTrainingBuffer,
        kXTrainAdditionalBuffer,
        kLossVarianceScatterIDBuffer,
        kLossVarianceThroughputBuffer,
        kEstimatorNNTrainBuffer + "0",
        kEstimatorNNTrainBuffer + "1",
        kEstimatorNNTrainBuffer + "2",
        kEstimatorNNTrainBuffer + "3",
        kEstimatorNNTrainBuffer + "4",
        kEstimatorNNXTrainAdditionalBuffer,
        kTrainPixel,
    };

    for (auto buffChannel : buffersNames)
    {
        mTracer.pVars[buffChannel] = renderData[buffChannel]->asBuffer();
    }
    pRenderContext->clearUAVCounter(renderData[kLossVarianceScatterIDBuffer]->asBuffer(), 0);

    mainRenderData.bindBuffers(pRenderContext, mTracer.pVars, renderData, targetDim);
    mTracer.pVars["gOutputColor"] = renderData[kColorOutputFinal]->asTexture();
    mTracer.pVars[kGTEstimations] = renderData[kGTEstimations]->asTexture();
    mTracer.pVars[kOurEstimations] = renderData[kOurEstimations]->asTexture();
    mTracer.pVars[kExpectedValuesTmp] = renderData[kExpectedValuesTmp]->asTexture();
    mTracer.pVars["gOutputM2"] = renderData[kColor2MomentOutput]->asTexture();


    pRenderContext->clearUAVCounter(renderData[kLThpSelfTrainingBuffer]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData[kXTrainBuffer + "4"]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData[kEstimatorNNTrainBuffer + "4"]->asBuffer(), 0);

    pRenderContext->clearUAVCounter(renderData[kQLearningPredBuffer + "4"]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData[kQLearningPredBuffer + "1"]->asBuffer(), 0);

    pRenderContext->clearUAVCounter(renderData[kNetworkEstimatorPredBuffer + "4"]->asBuffer(), 0);

    mpPixelDebug->prepareProgram(pProgram, mTracer.pVars->getRootVar());
    mpPixelStats->prepareProgram(pProgram, mTracer.pVars->getRootVar());
    pProgram->addDefine("RENDER_ONLY_INDIRECT", std::to_string(bOnlyIndirect));
    pProgram->addDefine("HEMISPHERICAL_RENDER", "0");
    pProgram->addDefine("RES_VARIANCE_LOSS", std::to_string(bResidualVarianceLoss));
    pProgram->addDefine("DEBUG_RENDER", std::to_string(bNNEstimatorDebugRender));
    pProgram->addDefine("ENABLE_NRC", std::to_string(bNRCEnabled));
    pProgram->addDefine("CONTINUE_RAY", "0");
    pProgram->addDefine("DEBUG_VIEW", std::to_string(bNRCDebugMode));
    pProgram->addDefine("EVAL_DIRECT_ONLY_ONCE", std::to_string(bEvalDirectOnlyOnce));


    bool bJustStarted = false;
    if (!model || bResetModel || model->getTrainingEpoch() < 2) {
        bJustStarted = true;
    }

    if (bNRCEnabled || 1) {
        pProgram->addDefine("GRADIENT_DENOISING", std::to_string(bGradientDenoising));
        pProgram->addDefine("MLOD", std::to_string(m_LOD));
        pProgram->addDefine("STOP_BY_METRIC", std::to_string(bStopByMetric));
        pProgram->addDefine("STOP_BY_OUR_METRIC", std::to_string(bStopByOurMetric));
        pProgram->addDefine("ERROR_BASED_METRIC", std::to_string(bErrorBased));
        pProgram->addDefine("DO_ERROR_TEST", std::to_string(bErrorTest));
        pProgram->addDefine("BIASED", std::to_string(bNRCBiased));
        pProgram->addDefine("INCIDENT_NRC", std::to_string(bIncidentNRC));
        pProgram->addDefine("TRAIN_DIRECT", std::to_string(bDirectLightCache));
        pProgram->addDefine("RESIDUAL_ERROR_ESTIMATION", std::to_string(bResErrorForLoss));
        pProgram->addDefine("VARIANCE_LOSS_MODE", std::to_string(bVarianceLoss));
        pProgram->addDefine("NUM_VARIANCE_LOSS_SAMPLES", std::to_string(mNumVarianceLossSamples));
        pProgram->addDefine("ESTIMATOR_NETWORK", std::to_string(bUseNNAsEstimator));
        pProgram->addDefine("SUPPORT_NEE", std::to_string(bDirectLightCacheNEE && bDirectLightCache));
        pProgram->addDefine("USE_GUIDANCE_OF_ENV_MAP", std::to_string(bNIRCBasedOnEnvMap));
        pProgram->addDefine("FILTERING_NET", std::to_string(bFilteringNetwork));
    }

    mTracer.pVars["GeneralData"]["bbStart"] = mpScene->getSceneBounds().minPoint;
    mTracer.pVars["GeneralData"]["bbEnd"] = mpScene->getSceneBounds().maxPoint;
    mTracer.pVars["GeneralData"]["original_dim"] = targetDim;
    if (bNRCEnabled)
        mTracer.pVars["GeneralData"]["samples_per_pixel"] = 1;
    else
        mTracer.pVars["GeneralData"]["samples_per_pixel"] = mNumGTSamples;
    mTracer.pVars["GeneralData"]["max_bounce_with_cv"] = maxBounceWithCV;
    mTracer.pVars["GeneralData"]["gd_hash_grid_res"] = gradientHashGridRes;
    mTracer.pVars["GeneralData"]["gd_hash_grid_res"] = gradientHashGridRes;
    mTracer.pVars["GeneralData"]["g_num_samples"][size_t(0)] = (bJustStarted) ? 1 : mNumNRCSamplesPerDepth[0];
    mTracer.pVars["GeneralData"]["g_num_samples"][size_t(1)] = (bJustStarted) ? 0 : mNumNRCSamplesPerDepth[1];
    mTracer.pVars["GeneralData"]["g_num_samples"][size_t(2)] = (bJustStarted) ? 0 : mNumNRCSamplesPerDepth[2];
    mTracer.pVars["GeneralData"]["roughnessThreshold"] = roughnessThreshold;
    mTracer.pVars["GeneralData"]["gRoulleteProb"] = roulleteProb;
    mTracer.pVars["GeneralData"]["expected_value_num_samples"] = mNumSamplesForExpectedValues;
    mTracer.pVars["GeneralData"]["num_training_id"] = (model) ? getTrainingStep() : 0;
    mTracer.pVars["GeneralData"]["errorThreshold"] = errorThreashold;
    if(bDebugHemispherical)
        mTracer.pVars["GeneralData"]["debug_pixel_id"] = mSelectedPixel;
    else
        mTracer.pVars["GeneralData"]["debug_pixel_id"] = int2(-1, -1);

    bool prevBRDFSampling = mSharedParams.useBRDFSampling;

    for (uint32_t i = 0; i < lodSizes.size(); i++)
    {
        mTracer.pVars["GeneralData"]["lodSize" + std::to_string(i)] = lodSizes[i];
    }
    // Spawn the rays.
    {
        PROFILE("MegakernelPathTracer::execute()_RayTrace - Predicting");
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));
    }


    if (!bNRCEnabled) {
        endFrame(pRenderContext, renderData);
        return;
    }

    if (bDebugHemispherical) {
        PROFILE("MegakernelPathTracer::execute()_RayTrace - Debug Hemisphere");

        pProgram->addDefine("HEMISPHERICAL_RENDER", "1");
        pProgram->addDefine("NUM_NRC_SAMPLES", std::to_string(1));
        mTracer.pVars["GeneralData"]["debug_pixel_id"] = mSelectedPixel;
        mTracer.pVars["GeneralData"]["samples_per_pixel"] = 1;
        mTracer.pVars["gOutputColor"] = renderData[kColorOutputDebugGT]->asTexture();

        debugRenderData.bindBuffers(pRenderContext, mTracer.pVars, renderData, DEBUG_RES);

        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(DEBUG_RES.x, DEBUG_RES.y, 1));

        pProgram->addDefine("HEMISPHERICAL_RENDER", "0");
    }


    uint32_t maxTrainingRays = 0;

    // Spawn the rays.
    if (bDispatchTrainingRays)
    {
        mTracer.pVars["GeneralData"]["debug_pixel_id"] = int2(-1, -1);

        uint2 train_dim_window;
        if (targetDim.x == 1920) {
            train_dim_window = uint2(8, 8);
        }
        else {
            train_dim_window = uint2(8, 4);
        }

        if (bDirectLightCache && bDirectLightCacheNEE) {
            train_dim_window.y *= 2;
        }

        mTracer.pVars["GeneralData"]["samples_per_pixel"] = 1;
        mTracer.pVars["GeneralData"]["train_dim_window"] = train_dim_window;
        setTracerData(renderData);
        pProgram->addDefine("CONTINUE_RAY", "1");
        pProgram->addDefine("Q_LEARNING", std::to_string(bQLearning || bFilteringNetwork));
        pProgram->addDefine("NUM_NRC_SAMPLES", std::to_string(mNumQLearningSamples));


        if (bVarianceLoss || bResidualVarianceLoss)
        {
            mTracer.pVars["gPredBuffer0"] = renderData[kNetworkEstimatorPredBuffer + "0"]->asBuffer();
            mTracer.pVars["gPredBuffer1"] = renderData[kNetworkEstimatorPredBuffer + "1"]->asBuffer();
            mTracer.pVars["gPredBuffer2"] = renderData[kNetworkEstimatorPredBuffer + "2"]->asBuffer();
            mTracer.pVars["gPredBuffer3"] = renderData[kNetworkEstimatorPredBuffer + "3"]->asBuffer();
            mTracer.pVars["gPredBuffer4"] = renderData[kNetworkEstimatorPredBuffer + "4"]->asBuffer();
            mTracer.pVars[kPredThroughput] = renderData[kNetworkEstimatorThroughput]->asBuffer();
        }
        else {


                mTracer.pVars["gPredBuffer0"] = renderData[kQLearningPredBuffer + "0"]->asBuffer();
                mTracer.pVars["gPredBuffer1"] = renderData[kQLearningPredBuffer + "1"]->asBuffer();
                mTracer.pVars["gPredBuffer2"] = renderData[kQLearningPredBuffer + "2"]->asBuffer();
                mTracer.pVars["gPredBufferMapping1"] = renderData[kQLearningPredBufferMapping + "1"]->asBuffer();
                mTracer.pVars["gPredBuffer3"] = renderData[kQLearningPredBuffer + "3"]->asBuffer();
                mTracer.pVars["gPredBuffer4"] = renderData[kQLearningPredBuffer + "4"]->asBuffer();
                mTracer.pVars[kPredPixelID] = renderData[kQLearningMapping]->asBuffer();
                mTracer.pVars[kPredThroughput] = renderData[kQLearningThroughput]->asBuffer();


        }

        PROFILE("MegakernelPathTracer::execute()_RayTrace - Training");

        maxTrainingRays = (targetDim.x / train_dim_window.x) * (targetDim.y / train_dim_window.y);
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim.x/train_dim_window.x, targetDim.y/train_dim_window.y, 1));
    }

    mSharedParams.useNEE = bPrevNEE;

    //// Call shared post-render code.
    //endFrame(pRenderContext, renderData);

    //if (!mpScene)
    //{
    //    return;
    //}

    if (!mpFenceForPredictionData) mpFenceForPredictionData = GpuFence::create(true);
    if (!mpFenceOut) mpFenceOut = GpuFence::create(true);
    if (!mpFenceOutDebug) mpFenceOutDebug = GpuFence::create(true);
    if (!mpFenceForOurEstimations) mpFenceForOurEstimations = GpuFence::create(true);

    // Wait getting data for predicting

    if (!mpCounterTrainBuffer) {
        mpCounterTrainBuffer = Buffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, Buffer::CpuAccess::Read);
        mpCounterEstimatorNNTrainBuffer = Buffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, Buffer::CpuAccess::Read);
        mpCounterQLearningPredBuffer = Buffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, Buffer::CpuAccess::Read);
        mpCounterQLearningPredPosBuffer = Buffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, Buffer::CpuAccess::Read);
        mpCounterNetworkEstimator = Buffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, Buffer::CpuAccess::Read);
        mpCounterLThpSelfTrainingBuffer = Buffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    Falcor::Buffer::SharedPtr xTrainBuff = renderData[kXTrainBuffer + "4"]->asBuffer();
    Falcor::Buffer::SharedPtr xEstimatorBuff = renderData[kEstimatorNNTrainBuffer + "4"]->asBuffer();

    Falcor::Buffer::SharedPtr xLThpSelfTrainBuff = renderData[kLThpSelfTrainingBuffer]->asBuffer();
    Falcor::Buffer::SharedPtr xXTrainAddBuff = renderData[kXTrainAdditionalBuffer]->asBuffer();

    Falcor::Buffer::SharedPtr xXTrainAddBuffEstimatorNN = renderData[kEstimatorNNXTrainAdditionalBuffer]->asBuffer();

    Falcor::Buffer::SharedPtr xPredPosBuff = renderData[kPredBuffer + "1"]->asBuffer();
    Falcor::Buffer::SharedPtr xPredBuff = renderData[kPredBuffer + "4"]->asBuffer();


    Falcor::Buffer::SharedPtr xQLearningPredBuffer = renderData[kQLearningPredBuffer + "4"]->asBuffer();
    Falcor::Buffer::SharedPtr xQLearningPredPosBuffer = renderData[kQLearningPredBuffer + "1"]->asBuffer();

    Falcor::Buffer::SharedPtr xNetworkEstimatorPredBuffer = renderData[kNetworkEstimatorPredBuffer+ "4"]->asBuffer();



    pRenderContext->copyBufferRegion(mpCounterQLearningPredPosBuffer.get(), 0, xQLearningPredPosBuffer->getUAVCounter().get(), 0, 4);
    pRenderContext->copyBufferRegion(mpCounterQLearningPredBuffer.get(), 0, xQLearningPredBuffer->getUAVCounter().get(), 0, 4);

    uint64_t signalValue;
    uint32_t NumTrainingElements = 0;
    uint32_t NumQLearningPredictPositions = 0;
    uint32_t NumQLearningPredictElements = 0;
    uint32_t NumNetworkEstimatorRequests = 0;
    uint32_t NumTrainingBuffersForNNEstimator = 0;


    bool bSync = mSync;
    if (bSync) {
#if 1
        pRenderContext->copyBufferRegion(mpCounterTrainBuffer.get(), 0, xTrainBuff->getUAVCounter().get(), 0, 4);
        pRenderContext->copyBufferRegion(mpCounterNetworkEstimator.get(), 0, xNetworkEstimatorPredBuffer->getUAVCounter().get(), 0, 4);
        pRenderContext->copyBufferRegion(mpCounterEstimatorNNTrainBuffer.get(), 0, xEstimatorBuff->getUAVCounter().get(), 0, 4);

        mainRenderData.requestNumRequests(pRenderContext, renderData);
        if (bDebugHemispherical)
            debugRenderData.requestNumRequests(pRenderContext, renderData);
#endif

        //pRenderContext->flush(false || SYNC_EVERYTHING);
        pRenderContext->flush(true);

        signalValue = mpFenceForPredictionData->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
        mpFenceForPredictionData->syncCpu();


#if 1

        uint32_t* uavCounters = (uint32_t*)mpCounterTrainBuffer->map(Buffer::MapType::Read);
        NumTrainingElements = uavCounters[0];
        mpCounterTrainBuffer->unmap();

        mainRenderData.updateNumActiveElements();

        if (bDebugHemispherical)
            debugRenderData.updateNumActiveElements();

        uint32_t* uavCounters4 = (uint32_t*)mpCounterQLearningPredBuffer->map(Buffer::MapType::Read);
        NumQLearningPredictElements = uavCounters4[0];
        mpCounterQLearningPredBuffer->unmap();


        uint32_t* uavCounters5 = (uint32_t*)mpCounterQLearningPredPosBuffer->map(Buffer::MapType::Read);
        NumQLearningPredictPositions = uavCounters5[0];
        mpCounterQLearningPredPosBuffer->unmap();

        uint32_t* uavCounters6 = (uint32_t*)mpCounterNetworkEstimator->map(Buffer::MapType::Read);
        NumNetworkEstimatorRequests = uavCounters6[0];
        mpCounterNetworkEstimator->unmap();

        uint32_t* uavCounters7 = (uint32_t*)mpCounterEstimatorNNTrainBuffer->map(Buffer::MapType::Read);
        NumTrainingBuffersForNNEstimator = uavCounters7[0];
        mpCounterEstimatorNNTrainBuffer->unmap();

#endif
    }

    uint32_t w = mScreenDim.x;
    uint32_t h = mScreenDim.y;
    uint32_t nElements = w * h * MAX_SAMPLES;
    uint32_t nrcElements = w * h / (8 * 4);


    if (!model || bResetModel)
    {
#if 1
        // Note: Path removed for privacy - using dynamic path resolution instead
        std::string dummy_path = Falcor::getWorkingDirectory();
        dummy_path = dummy_path.substr(0, dummy_path.find_last_of('\\') + 1);
#else
        const std::string dummy_path = "";
#endif
        std::string p = dummy_path;
        if (bIncidentNRC) {
            if (bDirectLightCache && bNIRCBasedOnEnvMap)
                p += kNetworkConfigNIRCEnvMap;
            else
                p += kNetworkConfigNIRC;
        }
        else
            p += kNetworkConfigNRC;
        std::vector<std::string> configs = {
            p
        };
        std::cout << dummy_path << std::endl;
        std::cout << dummy_path + kPyramidConfig << std::endl;
        model.reset(new Falcor::NNLaplacian("Model", configs, dummy_path + kPyramidConfig, 2, w * h * MAX_SAMPLES + nrcElements, Falcor::AABB(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f)), &m_diffuse_time, bIncidentNRC && !bForceNonSeparateInput));

        //model->m_denoising_level = &m_diffuse_time;
    }

    if (!modelEstimator || bResetModel) {
#if 1
        // Note: Path removed for privacy - using dynamic path resolution instead
        std::string dummy_path = Falcor::getWorkingDirectory();
        dummy_path = dummy_path.substr(0, dummy_path.find_last_of('\\') + 1);
#else
        const std::string dummy_path = "";
#endif
        std::vector<std::string> configs = {
            dummy_path + kNetworkConfigEstimator
        };
        std::cout << dummy_path << std::endl;
        std::cout << dummy_path + kPyramidConfig << std::endl;
        modelEstimator.reset(new Falcor::NNLaplacian("Model Estimator", configs, dummy_path + kPyramidConfig, 2, w * h * MAX_SAMPLES + nrcElements, Falcor::AABB(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f)), &m_diffuse_time));
        bResetModel = false;
    }

    //enableTraining(false);

    nrcElements = ((w * h) / (8 * 4))*2;


    if (estimationsBuffer == nullptr) {
#if 1
        nrcElements = 1;
#endif
        estimationsBuffer = Falcor::Buffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        ourestimationsBuffer = Falcor::Buffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorVarianceLoss = Falcor::Buffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorVarianceLossTmp = Falcor::Buffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorVarianceLossPart = Falcor::Buffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorQLearningBufferTmp = Falcor::Buffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorQLearningBuffer = Falcor::Buffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        //predColorResError = Falcor::Buffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
    }

    // Wait prediction results

    if (NumTrainingElements > 0 && model->getTrainFlag()) {
        PROFILE("NeuralNetwork Train Sorting IDs");
    }

    auto inferencePredict = [&](std::shared_ptr<Falcor::NNLaplacian> network, NeuralRequestData& reqdata, Falcor::Texture::SharedPtr renderFrame, uint64_t inputSignalValue, Falcor::GpuFence::SharedPtr fenIn = nullptr, Falcor::GpuFence::SharedPtr fenOut = nullptr, bool bCancelMapping=false, bool bFusedOutput=false) {
        if (reqdata.numElementsActive > 0)
        {
            pRenderContext->clearTexture(reqdata.mpRadianceMerge.get());

            PROFILE("NeuralNetwork Predict Start");
            // Just prediction

            if (0 && bNIRCBasedOnEnvMap && bIncidentNRC && bDirectLightCache) {
                PROFILE("EnvMap Requests");

                mpEnvMapRequests["PerFrameCB"]["gElems"] = reqdata.numElementsActive;
                mpScene->getEnvMap()->setShaderData(mpEnvMapRequests["gEnv"]);
                mpEnvMapRequests["gDirections"] = renderData[kPredBuffer + "4" + reqdata.name]->asBuffer();
                mpEnvMapRequests["gThroughput"] = renderData[kPredThroughput + reqdata.name]->asBuffer();
                mpEnvMapRequests->execute(pRenderContext, Falcor::uint3(reqdata.numElementsActive, uint32_t(1), uint32_t(1)));
            }

            NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kPredBuffer, reqdata.numElementsActive, kPredBufferMapping, reqdata.numUniqueElementsActive, reqdata.name, (bCancelMapping) ? nullptr : &reqdata.isChannelMapped);
            bool predictedSucessfuly = false;

            Buffer::SharedPtr thp = nullptr;
            Buffer::SharedPtr pixel_mapping = nullptr;
            if (bFusedOutput) {
                thp = renderData.getResource(kPredThroughput + reqdata.name)->asBuffer();
                pixel_mapping = renderData.getResource(kPredPixelID + reqdata.name)->asBuffer();
            }

            predictedSucessfuly = network->predict(reqdata.numElementsActive, neuralInputData, reqdata.mpPredictionResults, (bSync) ? fenIn : nullptr, inputSignalValue, fenOut, thp, pixel_mapping, false);

            return predictedSucessfuly;
        }

        return false;
    };

    auto inferencePredictMerge = [&](NeuralRequestData& reqdata, Falcor::Texture::SharedPtr renderFrame, uint64_t inputSignalValue, Falcor::GpuFence::SharedPtr fenIn = nullptr, Falcor::GpuFence::SharedPtr fenOut = nullptr, Falcor::Texture::SharedPtr secondMomentFrame = nullptr, bool bFusedOutput=false) {
        if (reqdata.numElementsActive > 0)
        {
            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();


                if (fenOut && bSync)
                    fenOut->syncGpu(pRenderContext->getLowLevelData()->getCommandQueue());
                {
                    if (!bFusedOutput) {


                        PROFILE("Merge predicted CV for all bounces to a intermediate frame buffer");
                        mpConvertNNOutputPass["PerFrameCB"]["gElems"] = reqdata.numElementsActive;
                        mpConvertNNOutputPass["PerFrameCB"]["outScaleInv"] = float(1.0 / model->getOutputScale());
                        mpConvertNNOutputPass->addDefine("CLAMP_RAD", std::to_string(!bVarianceLoss));
#if 0
                        mpConvertNNOutputPass->addDefine("THRESHOLD", std::to_string(bNIRCBasedOnEnvMap && bDirectLightCache));
#endif
                        // Run mapping predicted data from structure buffer to final texture
                        mpConvertNNOutputPass["gPredictedColor"] = reqdata.mpPredictionResults;
                        mpConvertNNOutputPass[kPredPixelID] = renderData[kPredPixelID + reqdata.name]->asBuffer();
                        mpConvertNNOutputPass[kPredThroughput] = renderData[kPredThroughput + reqdata.name]->asBuffer();
                        mpConvertNNOutputPass[kPredictedRadianceMerged] = reqdata.mpRadianceMerge;



                        mpConvertNNOutputPass->execute(pRenderContext, Falcor::uint3(reqdata.numElementsActive, uint32_t(1), uint32_t(1)));
                    }
                    else {
                        PROFILE("Merge predicted CV for all bounces to a intermediate frame buffer");
                        // Run mapping predicted data from structure buffer to final texture
                        mpConvertNNOutputPassFused["gPredictedColor"] = reqdata.mpPredictionResults;
                        mpConvertNNOutputPassFused[kPredictedRadianceMerged] = reqdata.mpRadianceMerge;

                        mpConvertNNOutputPassFused->execute(pRenderContext, Falcor::uint3(mScreenDim.x*mScreenDim.y, uint32_t(1), uint32_t(1)));
                    }
                }
        }

        {
            PROFILE("Merge estimated radiance and predicted one");
            mpFinalOutputPass[kPredictedRadianceMerged] = reqdata.mpRadianceMerge;
            mpFinalOutputPass["colorOutput"] = renderFrame;


            if (secondMomentFrame)
                mpFinalOutputPass["secondMoment"] = secondMomentFrame->asTexture();

            mpFinalOutputPass->addDefine("SECOND_MOMENT", std::to_string(bStDevEstimation && secondMomentFrame != nullptr));
            mpFinalOutputPass->execute(pRenderContext, Falcor::uint3(reqdata.res.x, reqdata.res.y, uint32_t(1)));
        }
    };



    if (bNNEstimatorDebugRender || bNRCDebugMode) {
        bool havePredictions = inferencePredict((bIncidentNRC || bFilteringNetwork) ? modelEstimator : model, mainRenderData, renderData[kColorOutputFinal]->asTexture(), signalValue, mpFenceForPredictionData, mpFenceOut, true, bFusedOutput && bIncidentNRC && !bJustStarted);
        if (havePredictions)
            inferencePredictMerge(mainRenderData, renderData[kColorOutputFinal]->asTexture(), signalValue, mpFenceForPredictionData, mpFenceOut, nullptr, bFusedOutput && bIncidentNRC && model->getTrainingEpoch() > 1);
    }
    else{
        if (bDebugHemispherical) {
            bool havePredictions = inferencePredict(model, mainRenderData, renderData[kColorOutputFinal]->asTexture(), signalValue, mpFenceForPredictionData, nullptr, false, bFusedOutput && bIncidentNRC && !bJustStarted);
            pRenderContext->clearTexture(renderData[kColorOutputDebug]->asTexture().get());
            inferencePredict(model, debugRenderData, renderData[kColorOutputDebug]->asTexture(), 0, nullptr, mpFenceOut);

            if (havePredictions) {
                inferencePredictMerge(mainRenderData, renderData[kColorOutputFinal]->asTexture(), signalValue, mpFenceForPredictionData, mpFenceOut, renderData[kColor2MomentOutput]->asTexture(), bFusedOutput&& bIncidentNRC&& !bJustStarted);
                inferencePredictMerge(debugRenderData, renderData[kColorOutputDebug]->asTexture(), 0, nullptr, nullptr, nullptr);
            }
        }
        else {
            bool havePredictions = inferencePredict(model, mainRenderData, renderData[kColorOutputFinal]->asTexture(), signalValue, mpFenceForPredictionData, mpFenceOut, false, bFusedOutput && bIncidentNRC && !bJustStarted);
            if (havePredictions)
                inferencePredictMerge(mainRenderData, renderData[kColorOutputFinal]->asTexture(), signalValue, mpFenceForPredictionData, mpFenceOut, renderData[kColor2MomentOutput]->asTexture(), bFusedOutput && bIncidentNRC && !bJustStarted);
        }
    }



    if (NumTrainingElements > 0 && model->getTrainFlag() && bTonemapWeight && bFilteringNetwork && bIncidentNRC) {
        mpGatherOutEstimations["PerFrameCB"]["gElems"] = maxTrainingRays;
        mpGatherOutEstimations["gInput"] = renderData[kColorOutputFinal]->asTexture();
        mpGatherOutEstimations[kTrainPixel] = renderData[kTrainPixel]->asBuffer();
        mpGatherOutEstimations["gOutput"] = ourestimationsBuffer;

        mpGatherOutEstimations->execute(pRenderContext, Falcor::uint3(maxTrainingRays, uint32_t(1), uint32_t(1)));
        pRenderContext->flush(false);

        ourestimationSignal = mpFenceForOurEstimations->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

    }

    //if (NumTrainingElements > 0 && ((bVarianceLoss && bUseNNAsEstimator) || (bFilteringNetwork && bIncidentNRC) || bResidualVarianceLoss) && model->getTrainFlag()) {
    //    if (!bEventIsCreated)
    //    {
    //        CUDA_CHECK_THROW(cudaEventCreate(&eventNetworksSync));
    //        bEventIsCreated = true;
    //    }
    //
    //    NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kNetworkEstimatorPredBuffer, NumTrainingElements, "", 0);
    //    modelEstimator->predict(NumTrainingElements, neuralInputData, estimationsBuffer, mpFenceForPredictionData, signalValue, nullptr, nullptr, true);

    //    Falcor::Buffer::SharedPtr throughputBuffer = renderData[kNetworkEstimatorThroughput]->asBuffer();
    //    modelEstimator->transformNetworkOutput(NumTrainingElements, NumTrainingElements, estimationsBuffer, estimationsBuffer, throughputBuffer, eventNetworksSync, model->getOutputScale(), false, !bResidualVarianceLoss);
    //}


    Falcor::Buffer::SharedPtr qLearnPredictBuffer = nullptr;
    std::shared_ptr<Falcor::NNLaplacian> executableModel = (bFilteringNetwork) ? modelEstimator : model;
    if (NumQLearningPredictElements != 0) {
        if (!bEventIsCreated)
        {
            CUDA_CHECK_THROW(cudaEventCreate(&eventNetworksSync));
            bEventIsCreated = true;
        }


        // QLearning
        PROFILE("NeuralNetwork QLearning Predict");
        // Just prediction

        std::vector<bool> isChannelMapped(5);
        isChannelMapped[0] = true;
        isChannelMapped[1] = true;
        isChannelMapped[2] = true;
        isChannelMapped[3] = true;
        isChannelMapped[4] = false;

        if (SYNC_EVERYTHING)
            cudaDeviceSynchronize();
        NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kQLearningPredBuffer, NumQLearningPredictElements, kQLearningPredBufferMapping, NumQLearningPredictPositions, "", &isChannelMapped);
        executableModel->predict(NumQLearningPredictElements, neuralInputData, predColorQLearningBufferTmp, nullptr, 0, nullptr, nullptr, nullptr, true);
        if (SYNC_EVERYTHING)
            cudaDeviceSynchronize();
        Falcor::Buffer::SharedPtr QLearningMapping = renderData[kQLearningMapping]->asBuffer();
        Falcor::Buffer::SharedPtr QLearningThroughput = renderData[kQLearningThroughput]->asBuffer();

        executableModel->accumulatedRadianceForSelfLearning(NumQLearningPredictElements, nrcElements, predColorQLearningBufferTmp, predColorQLearningBuffer, QLearningMapping, QLearningThroughput, executableModel->getOutputScale());
        if(executableModel == modelEstimator)
            executableModel->pushEvent(eventNetworksSync);

        qLearnPredictBuffer = predColorQLearningBuffer;
    }



    if (NumTrainingBuffersForNNEstimator > 0 && ((bVarianceLoss) || (bFilteringNetwork && bIncidentNRC)) && modelEstimator->getTrainFlag() && model->getTrainFlag()) {
        uint32_t dims = (w / 8) * (h / 4) * 2;

        if (YTrainingEstimatorNNBuffer == nullptr) {
            YTrainingEstimatorNNBuffer = Falcor::Buffer::createStructured(sizeof(Falcor::YData), dims, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        }

        if (SYNC_EVERYTHING)
            cudaDeviceSynchronize();


        modelEstimator->assemblyTrainingData(NumTrainingBuffersForNNEstimator, (bSelfLearningForFilterNetwork) ? qLearnPredictBuffer : nullptr, xXTrainAddBuffEstimatorNN, xLThpSelfTrainBuff, YTrainingEstimatorNNBuffer, nullptr, nullptr, nullptr, nullptr, nullptr, modelEstimator->getOutputScale());

        NeuralInputDataFalcor neuralInputData2 = setupNeuralInput(renderData, kEstimatorNNTrainBuffer, NumTrainingBuffersForNNEstimator, "", 0);
        if (SYNC_EVERYTHING)
            cudaDeviceSynchronize();

        modelEstimator->fit(NumTrainingBuffersForNNEstimator, neuralInputData2, YTrainingEstimatorNNBuffer, nullptr, 0, !bJIT);
    }


    if (NumTrainingElements > 0 && model->getTrainFlag()) {
        uint32_t dims = (w / 8) * (h / 4) * 4;

        if (YTrainingBuffer == nullptr) {
            YTrainingBuffer = Falcor::Buffer::createStructured(sizeof(Falcor::YData), dims*2, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        }


        //Falcor::Buffer::SharedPtr qLearnPredictBuffer = nullptr;
        //if (NumQLearningPredictElements != 0) {
        //    // QLearning
        //    PROFILE("NeuralNetwork QLearning Predict");
        //    // Just prediction

        //    std::vector<bool> isChannelMapped(5);
        //    isChannelMapped[0] = true;
        //    isChannelMapped[1] = true;
        //    isChannelMapped[2] = true;
        //    isChannelMapped[3] = true;
        //    isChannelMapped[4] = false;


        //    if (SYNC_EVERYTHING)
        //        cudaDeviceSynchronize();
        //    NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kQLearningPredBuffer, NumQLearningPredictElements, kQLearningPredBufferMapping, NumQLearningPredictPositions, "", &isChannelMapped);
        //    model->predict(NumQLearningPredictElements, neuralInputData, predColorQLearningBufferTmp, nullptr, 0, nullptr, nullptr, true);
        //    if (SYNC_EVERYTHING)
        //        cudaDeviceSynchronize();
        //    Falcor::Buffer::SharedPtr QLearningMapping = renderData[kQLearningMapping]->asBuffer();
        //    Falcor::Buffer::SharedPtr QLearningThroughput = renderData[kQLearningThroughput]->asBuffer();
        //    model->accumulatedRadianceForSelfLearning(NumQLearningPredictElements, nrcElements, predColorQLearningBufferTmp, predColorQLearningBuffer, QLearningMapping, QLearningThroughput, model->getOutputScale());

        //
        //    qLearnPredictBuffer = predColorQLearningBuffer;
        //}

        Falcor::Buffer::SharedPtr VarianceLossMapping = renderData[kLossVarianceScatterIDBuffer]->asBuffer();

        if (bVarianceLoss) {
            PROFILE("Predict For Variance Loss");
            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();
            NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kXTrainBuffer, NumTrainingElements, "", 0);
            model->predict(NumTrainingElements, neuralInputData, predColorVarianceLossTmp, nullptr, 0, nullptr, nullptr, nullptr, true);

            Falcor::Buffer::SharedPtr VarianceLossThroughput = renderData[kLossVarianceThroughputBuffer]->asBuffer();


            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();


            model->accumulatedRadianceForSelfLearning(NumTrainingElements, NumTrainingElements, predColorVarianceLossTmp, predColorVarianceLoss, VarianceLossMapping, VarianceLossThroughput, model->getOutputScale(), true);
            model->accumulatedRadianceOnlyPart(NumTrainingElements, NumTrainingElements, predColorVarianceLossTmp, predColorVarianceLossPart, VarianceLossMapping, VarianceLossThroughput, model->getOutputScale(), true);
        }

        if (bResidualVarianceLoss) {

            if (!bEventIsCreatedResVar)
            {
                CUDA_CHECK_THROW(cudaEventCreate(&eventNetworksSyncResVar));
                bEventIsCreatedResVar = true;
            }

            NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kXTrainBuffer, NumTrainingElements, "", 0);
            model->predict(NumTrainingElements, neuralInputData, predColorVarianceLossTmp, nullptr, 0, nullptr, nullptr, nullptr, true);

            Falcor::Buffer::SharedPtr VarianceLossThroughput = renderData[kLossVarianceThroughputBuffer]->asBuffer();


            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();


            model->transformNetworkOutput(NumTrainingElements, NumTrainingElements, predColorVarianceLossTmp, predColorVarianceLossTmp, VarianceLossThroughput, eventNetworksSyncResVar, model->getOutputScale(), false, true);

            uint32_t dims = (w / 8) * (h / 4) * 2;

            if (YTrainingEstimatorNNBuffer == nullptr) {
                YTrainingEstimatorNNBuffer = Falcor::Buffer::createStructured(sizeof(Falcor::YData), dims, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
            }

            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();
            modelEstimator->syncEvent(eventNetworksSyncResVar);


            modelEstimator->assemblyTrainingData(NumTrainingElements, nullptr, xXTrainAddBuffEstimatorNN, xLThpSelfTrainBuff, YTrainingEstimatorNNBuffer, predColorVarianceLossTmp, nullptr, nullptr, nullptr, nullptr, modelEstimator->getOutputScale());

            NeuralInputDataFalcor neuralInputData2 = setupNeuralInput(renderData, kEstimatorNNTrainBuffer, NumTrainingElements, "", 0);
            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();
            modelEstimator->fit(NumTrainingElements, neuralInputData2, YTrainingEstimatorNNBuffer, nullptr, 0, !bJIT);
        }

         if (SYNC_EVERYTHING)
             cudaDeviceSynchronize();

         if ((bUseNNAsEstimator || bFilteringNetwork || bResidualVarianceLoss ) && bEventIsCreated && executableModel == modelEstimator) {
            // We must wait for estimations from the supporting NN
            model->syncEvent(eventNetworksSync);
         }

         if (bTonemapWeight && bFilteringNetwork) {
             model->assemblyTrainingData(NumTrainingElements, (bQLearning || bFilteringNetwork) ? qLearnPredictBuffer : nullptr, xXTrainAddBuff, xLThpSelfTrainBuff, YTrainingBuffer, nullptr, (bVarianceLoss) ? predColorVarianceLoss : nullptr, (bVarianceLoss) ? VarianceLossMapping : nullptr, estimationsBuffer, ourestimationsBuffer, model->getOutputScale(), mpFenceForOurEstimations, ourestimationSignal);
         }
         else {
            model->assemblyTrainingData(NumTrainingElements, (bQLearning || bFilteringNetwork) ? qLearnPredictBuffer : nullptr, xXTrainAddBuff, xLThpSelfTrainBuff, YTrainingBuffer, nullptr, (bVarianceLoss) ? predColorVarianceLoss : nullptr, (bVarianceLoss) ? VarianceLossMapping : nullptr, ((bVarianceLoss&& bUseNNAsEstimator) || bFilteringNetwork || bResidualVarianceLoss) ? estimationsBuffer : nullptr, (bVarianceLoss) ? predColorVarianceLossPart : nullptr, model->getOutputScale());
         }



        {
            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();
            NeuralInputDataFalcor neuralInputData2 = setupNeuralInput(renderData, kXTrainBuffer, NumTrainingElements, "", 0);
            model->fit(NumTrainingElements, neuralInputData2, YTrainingBuffer, nullptr, 0, !bJIT);
        }
    }





#if 1
    m_training_epoch = model->getTrainingEpoch();
    m_training_error = model->getTrainingError();
#endif

   // mpPixelStats->setEnabled(true);
    endFrame(pRenderContext, renderData);
}


bool Falcor::DummyNeuralNetwork::onMouseEvent(const MouseEvent& mouseEvent) {
    Falcor::PathTracer::onMouseEvent(mouseEvent);

    if (bDebugHemispherical)
    {
        if (mouseEvent.type == Falcor::MouseEvent::Type::Move && !bFreezeDebugMouse)
        {
            mSelectedPixel = uint2(mouseEvent.pos * float2(mScreenDim));
            return false;
        }
    }

    return false;
}

bool Falcor::DummyNeuralNetwork::onKeyEvent(const Falcor::KeyboardEvent& keyEvent) {
    if (bDebugHemispherical) {
        if (keyEvent.type == Falcor::KeyboardEvent::Type::KeyPressed && keyEvent.key == Falcor::KeyboardEvent::Key::F) {
            bFreezeDebugMouse = !bFreezeDebugMouse;
            return true;
        }
    }

    return false;
}

__declspec(dllexport) void Falcor::DummyNeuralNetwork::reloadModel() {
    model->reloadModel();
    modelEstimator->reloadModel();
}


__declspec(dllexport) void Falcor::DummyNeuralNetwork::enablestats() {
    mpPixelStats->setEnabled(true);
}

void Falcor::DummyNeuralNetwork::renderUI(Falcor::Gui::Widgets& widget)
{
    if (auto nnGroup = widget.group("NeuralNetwork", true))
    {
        widget.checkbox("Dispatch Training Rays", bDispatchTrainingRays);
        if (widget.var("Max bounce guided by CV", maxBounceWithCV, uint32_t(0), uint32_t(5), 1)) {

        }
        if (widget.var("Number GT Samples", mNumGTSamples, uint32_t(1), uint32_t(1000))) {
            if (mNumGTSamples > 10) {
                mNumGTSamples = 10;
            }
        }


        if (widget.button("NRC")) {
            bIncidentNRC = false;
            bDirectLightCache = false;

            bResetModel = true;


            mNumNRCSamplesPerDepth[0] = 1;
            mNumNRCSamplesPerDepth[1] = 0;
            mNumNRCSamplesPerDepth[2] = 0;
        }
        if (widget.button("NIRC")) {
            bIncidentNRC = true;
            bDirectLightCache = false;

            bResetModel = true;

            mNumNRCSamplesPerDepth[0] = 7;
            mNumNRCSamplesPerDepth[1] = 7;
            mNumNRCSamplesPerDepth[2] = 7;
        }

        widget.checkbox("Filtering Network", bFilteringNetwork);
        widget.checkbox("Enable NRC", bNRCEnabled);
        widget.checkbox("Debug NRC", bNRCDebugMode);
        widget.checkbox("Biased NRC", bNRCBiased);
        if (bNRCBiased) {
            widget.checkbox("Stop By Metric", bStopByMetric);
            widget.checkbox("Stop By Our Metric", bStopByOurMetric);
            widget.checkbox("Stop Error Based", bErrorBased);
            widget.checkbox("Do REAL Error Test", bErrorTest);
            widget.var("errorThreashold", errorThreashold, float(0.0f), float(10000.0f));
        }

        widget.checkbox("Direct Light Cache", bDirectLightCache);
        widget.checkbox("Eval Direct Only Once Per Frame", bEvalDirectOnlyOnce);
        widget.checkbox("Tonemap Weighting", bTonemapWeight);
        if (bIncidentNRC) {
            auto nircGroup = widget.group("NIRC Settings", true);
            if (bDirectLightCache) {
                nircGroup.checkbox("Direct NEE", bDirectLightCacheNEE);
                if (nircGroup.checkbox("EnvMap Based", bNIRCBasedOnEnvMap)) {

                    bResetModel = true;
                }
            }

            nircGroup.checkbox("Filter Net", bFilteringNetwork);

            if (bFilteringNetwork) {
                nircGroup.checkbox("Debug Render", bNNEstimatorDebugRender);
                nircGroup.checkbox("Self Learning", bSelfLearningForFilterNetwork);
            }
        }
        widget.checkbox("Only Indirect", bOnlyIndirect);
        if (widget.checkbox("NIRC", bIncidentNRC)) {
            bResetModel = true;

            mNumNRCSamplesPerDepth[0] = 1;
            mNumNRCSamplesPerDepth[1] = 0;
            mNumNRCSamplesPerDepth[2] = 0;
        }


        widget.checkbox("Use Res Error for a loss", bResErrorForLoss);
        widget.checkbox("Enable hemispherical debug render", bDebugHemispherical);
        widget.checkbox("Variance Loss", bVarianceLoss);
        widget.checkbox("Residual Variance Loss", bResidualVarianceLoss);
        if (widget.checkbox("Force NonSeparate Input", bForceNonSeparateInput)) {


            if (bForceNonSeparateInput) {
                model->m_separate_input = false;
                bFusedOutput = false;
            }

            if (!bForceNonSeparateInput)
                model->m_separate_input = bIncidentNRC;
        }
        widget.checkbox("Sync", mSync);
        widget.checkbox("Second Moment", bStDevEstimation);


        widget.var("Roughness Threshload", roughnessThreshold, 0.00001f, 1.0f);
        widget.var("RRProb", roulleteProb, 0.00000f, 1.0f);
        if (widget.button("Enable JIT")) {
            bJIT = true;
        }

        if (bVarianceLoss){
            if (auto varianceGroup = widget.group("Variance Estimator", true)) {
                varianceGroup.checkbox("Use a Neural Network as the Estimator", bUseNNAsEstimator);
                varianceGroup.checkbox("Debug render", bNNEstimatorDebugRender);
                varianceGroup.var("Number Samples for the loss", mNumVarianceLossSamples, uint32_t(0), uint32_t(3));
            }
        }

        if (bDebugHemispherical)
        {
            widget.var("Selected pixel", mSelectedPixel);
        }



        if (widget.var("Num Samples 0", mNumNRCSamplesPerDepth[0], uint32_t(0), uint32_t(MAX_SAMPLES - 1))) {
            uint32_t sum = mNumNRCSamplesPerDepth[0] + mNumNRCSamplesPerDepth[1] + mNumNRCSamplesPerDepth[2];
            if (sum > MAX_SAMPLES-3) {
                mNumNRCSamplesPerDepth[0] = MAX_SAMPLES-3 - mNumNRCSamplesPerDepth[1] - mNumNRCSamplesPerDepth[2];
            }
        }
        if(widget.var("Num Samples 1", mNumNRCSamplesPerDepth[1], uint32_t(0), uint32_t(MAX_SAMPLES - 1))) {
            uint32_t sum = mNumNRCSamplesPerDepth[0] + mNumNRCSamplesPerDepth[1] + mNumNRCSamplesPerDepth[2];
            if (sum > MAX_SAMPLES-3) {
                mNumNRCSamplesPerDepth[1] = MAX_SAMPLES-3 - mNumNRCSamplesPerDepth[0] - mNumNRCSamplesPerDepth[2];
            }
        }
        if(widget.var("Num Samples 2", mNumNRCSamplesPerDepth[2], uint32_t(0), uint32_t(MAX_SAMPLES - 1))) {
            uint32_t sum = mNumNRCSamplesPerDepth[0] + mNumNRCSamplesPerDepth[1] + mNumNRCSamplesPerDepth[2];
            if (sum > MAX_SAMPLES-3) {
                mNumNRCSamplesPerDepth[2] = MAX_SAMPLES-3 - mNumNRCSamplesPerDepth[0] - mNumNRCSamplesPerDepth[1];
            }
        }

        //widget.var("Number NRC Samples", mNumNRCSamples, uint32_t(0), uint32_t(7), 1);

        widget.checkbox("QLearning", bQLearning);

        bool bFus = bFusedOutput;
        if (widget.checkbox("FusedOutput", bFus)) {
            if (bForceNonSeparateInput) {
                bFus = false;
            }

            bFusedOutput = bFus;
        }
        widget.var("Number QLearning NRC Samples", mNumQLearningSamples);
    }

    if(model)
        model->renderUI(widget);

    if (modelEstimator)
        modelEstimator->renderUI(widget);


    PathTracer::renderUI(widget);
}

__declspec(dllexport) int32_t Falcor::DummyNeuralNetwork::getTrainingEpoch() const {
    return m_training_epoch;
}

__declspec(dllexport) float Falcor::DummyNeuralNetwork::getTrainingError() const {
    return m_training_error;
}

float Falcor::DummyNeuralNetwork::getDiffuseTime() const {
    return m_diffuse_time;
}


void Falcor::DummyNeuralNetwork::setDiffuseTime(float t){
    m_diffuse_time = t;
}

