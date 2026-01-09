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
#pragma once


#include "Falcor.h"
#include "RenderPasses/Shared/PathTracer/PathTracer.h"



namespace Falcor {
    class NNLaplacian;

    class SimpleNNModel;


    struct NeuralRequestData {

        NeuralRequestData() {
            bInit = false;
            mpRadianceMerge = nullptr;
            mpAtomicCounterCPURead = nullptr;
            mpPredictionResults = nullptr;
            mpAtomicCounterCPURead = nullptr;

        }

        NeuralRequestData(std::string _name, uint32_t numbuff, uint32_t numelements, int32_t numuniqueelements = -1, bool bForcedMapping=false) {
            name = std::move(_name);
            numBuffers = numbuff;
            numElements = numelements;
            bMapped = bForcedMapping;
            if (numuniqueelements == -1) {
                numUniqueElements = numElements;
            }
            else {
                numUniqueElements = numuniqueelements;
                bMapped = true;
            }
            bInit = true;
            mpRadianceMerge = nullptr;
            mpAtomicCounterCPURead = nullptr;
            mpRadianceMerge = nullptr;
            mpAtomicCounterCPURead = nullptr;
            mpPredictionResults = nullptr;
            mpAtomicCounterCPURead = nullptr;
        }

        bool bInit;
        bool bMapped;

        Falcor::Texture::SharedPtr mpRadianceMerge;
        Falcor::Buffer::SharedPtr mpPredictionResults;
        Falcor::Buffer::SharedPtr mpAtomicCounterCPURead;
        std::string name;
        uint32_t numBuffers;
        uint32_t numElements;
        uint32_t numUniqueElements;
        uint32_t numElementsActive = 0;
        uint32_t numUniqueElementsActive = 0;
        Falcor::uint2 res;

        std::vector<bool> isChannelMapped;

        bool isMapped() const {
            if (!bInit)
                return false;
            return bMapped;
        }

        bool allocateBuffers(Falcor::RenderPassReflection& reflector);
        bool bindBuffers(Falcor::RenderContext* pRenderContext, Falcor::RtProgramVars::SharedPtr vars, const Falcor::RenderData& renderData, Falcor::uint2 resolution);

        void requestNumRequests(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData);
        void updateNumActiveElements();
    };

class DummyNeuralNetwork : public Falcor::PathTracer
{
public:
    using SharedPtr = std::shared_ptr<DummyNeuralNetwork>;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(Falcor::RenderContext* pRenderContext = nullptr, const Falcor::Dictionary& dict = {});

    virtual std::string getDesc() override;
    __declspec(dllexport) virtual Falcor::Dictionary getScriptingDictionary() override;
    __declspec(dllexport) void parseDictionary(const Falcor::Dictionary& dict);
    virtual Falcor::RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(Falcor::RenderContext* pContext, const CompileData& compileData) override {}
    virtual void execute(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData) override;
    virtual void renderUI(Falcor::Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    bool onKeyEvent(const Falcor::KeyboardEvent& keyEvent) override;

    __declspec(dllexport) void resetNN();

    __declspec(dllexport) bool isStDevEstimating() const {
        return bStDevEstimation;
    }

    __declspec(dllexport) void setModel(bool NIRC) {
        bIncidentNRC = NIRC;
    }
    __declspec(dllexport) void enablestats();

    __declspec(dllexport) void nrc_experiment_setup();
    __declspec(dllexport) void nirc_experiment_setup(int samples);

    __declspec(dllexport) void enableTraining(bool bEnable);

    __declspec(dllexport) void enableInference(bool bEnable);


    __declspec(dllexport) float getAveragePath() const;

    __declspec(dllexport) float getLoss() const;

    __declspec(dllexport) int getTrainingStep() const;

    DummyNeuralNetwork(const Dictionary& dict);
    
    __declspec(dllexport) void reloadModel();
    __declspec(dllexport) float getTrainingError() const;
    __declspec(dllexport) int32_t getTrainingEpoch() const;
    float getDiffuseTime() const;
    void setDiffuseTime(float t);

    float m_training_error;
    int32_t m_training_epoch;
    float m_diffuse_time;

    int32_t mMaxTrainingSteps = -1;

    uint2 mScreenDim = { 1920, 1080 };
    std::vector<uint32_t> lodSizes;
    uint32_t m_LOD;
    static const char* sDesc;

    uint32_t mNumNRCSamplesPerDepth[3];

    std::shared_ptr<Falcor::NNLaplacian> model;
    std::shared_ptr<Falcor::NNLaplacian> modelEstimator;

    __declspec(dllexport) Texture::SharedPtr getOutput() {
        return output;
    }

private:

    Texture::SharedPtr output;
    void recreateVars() override { mTracer.pVars = nullptr; }
    void prepareVars();
    void setTracerData(const Falcor::RenderData& renderData);
    
    Falcor::Buffer::SharedPtr predColorQLearningBuffer;
    Falcor::Buffer::SharedPtr predColorResError;
    Falcor::Buffer::SharedPtr predColorQLearningBufferTmp;
    Falcor::Buffer::SharedPtr predColorVarianceLossTmp;
    Falcor::Buffer::SharedPtr predColorVarianceLoss;
    Falcor::Buffer::SharedPtr predColorVarianceLossPart;
    Falcor::Buffer::SharedPtr estimationsBuffer;
    Falcor::Buffer::SharedPtr ourestimationsBuffer;
    Falcor::Buffer::SharedPtr YTrainingBuffer;
    Falcor::Buffer::SharedPtr YTrainingEstimatorNNBuffer;
    Falcor::ComputePass::SharedPtr          mpEnvMapRequests;
    Falcor::ComputePass::SharedPtr          mpGenerateTrainingDataPass;
    Falcor::ComputePass::SharedPtr          mpConvertNNOutputPass;
    Falcor::ComputePass::SharedPtr          mpConvertNNOutputPassFused;
    Falcor::ComputePass::SharedPtr          mpGatherOutEstimations;
    Falcor::ComputePass::SharedPtr          mpFinalOutputPass;
    Falcor::ComputePass::SharedPtr          mpYTrainAssemblyPass;
    bool skipFirstPred = true;
    uint64_t ourestimationSignal;
    Falcor::Buffer::SharedPtr mpCounterTrainBuffer;
    Falcor::Buffer::SharedPtr mpCounterEstimatorNNTrainBuffer;
    Falcor::Buffer::SharedPtr mpCounterPredBuffer;
    Falcor::Buffer::SharedPtr mpCounterPredPositionBuffer;
    Falcor::Buffer::SharedPtr mpCounterNetworkEstimator;
    Falcor::Buffer::SharedPtr mpCounterLThpSelfTrainingBuffer;

    Falcor::Buffer::SharedPtr mpCounterQLearningPredPosBuffer;
    Falcor::Buffer::SharedPtr mpCounterQLearningPredBuffer;
    
    Falcor::Scene::SharedPtr mpScene;

    Falcor::GpuFence::SharedPtr mpFenceForOurEstimations;
    Falcor::GpuFence::SharedPtr mpFenceForPredictionData;
    Falcor::GpuFence::SharedPtr mpFenceForTrainingData;
    Falcor::GpuFence::SharedPtr mpFenceOut;
    Falcor::GpuFence::SharedPtr mpFenceOutDebug;


    bool bErrorBased = false;
    float errorThreashold;
    bool bInitCudaFence = false;
    uint32_t MAX_SAMPLES = 40;
    // Ray tracing program.
    struct
    {
        Falcor::RtProgram::SharedPtr pProgram;
        Falcor::RtBindingTable::SharedPtr pBindingTable;
        Falcor::RtProgramVars::SharedPtr pVars;
        Falcor::ParameterBlock::SharedPtr pParameterBlock;      ///< ParameterBlock for all data.
    } mTracer;

    bool bDispatchTrainingRays;
    bool bGradientDenoising;
    bool bFilteringNetwork = false;
    bool bNRCEnabled;
    bool bOnlyIndirect;
    bool bIncidentNRC;
    bool bQLearning;
    bool bNRCDebugMode;
    bool bStopByMetric = false;
    bool bStopByOurMetric = false;
    bool bErrorTest = false;
    bool bNRCBiased;
    bool bForceNonSeparateInput = false;
    bool bDirectLightCache;
    bool mSync = true;
    bool bEvalDirectOnlyOnce = true;
    bool bTonemapWeight = false;
    bool bDirectLightCacheNEE;
    bool bNIRCBasedOnEnvMap = true;
    bool bResErrorForLoss;
    bool bDebugHemispherical;
    bool bVarianceLoss = false;
    bool bStDevEstimation = false;
    bool bFusedOutput = false;

    bool bResidualVarianceLoss = false;
    bool bNNEstimatorDebugRender = false;
    bool bFreezeDebugMouse = false;

    float roughnessThreshold = 0.45f;
    float roulleteProb = 0.0f;
    bool bJIT = true;

    uint32_t mNumGTSamples = 1;
    uint32_t mNumVarianceLossSamples = 2;
    
    uint32_t mNumQLearningSamples;
    uint32_t gradientHashGridRes;
    uint32_t maxBounceWithCV;

    cudaEvent_t eventNetworksSync;
    cudaEvent_t eventNetworksSyncResVar;
    bool bEventIsCreatedResVar = false;
    bool bEventIsCreated = false;
    
    bool bSelfLearningForFilterNetwork = false;
    bool bUseNNAsEstimator = false;
    uint32_t gradientHashGridSize;

    uint32_t mNumSamplesForExpectedValues = 32;
    uint2  mSelectedPixel = { 0, 0 };      ///< Currently selected pixel.

    bool bResetModel = false;
    NeuralRequestData mainRenderData;
    NeuralRequestData debugRenderData;
};
}
