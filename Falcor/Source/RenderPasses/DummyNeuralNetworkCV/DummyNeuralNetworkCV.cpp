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
#include "DummyNeuralNetworkCV.h"
#include "Utils/NeuralNetworks/NeuralNetworkModel.h"
#include "json/json.hpp"
#include "cub/cub.cuh"
#include "tiny-cuda-nn/common.h"
#include <tiny-cuda-nn/common_device.h>
#include "NNLaplacianCV.h"
#include "RenderPasses/Shared/NeuralNetwork/NeuralStructuresCV.slang"



namespace
{
    const char kDesc[] = "Insert pass description here";
    const std::string kColorInput = "colorInput";
    const std::string kColorOutput = "colorOutput";
    const std::string kPredictedRadianceMerged = "gPredictedRadianceMerged";
    const std::string kConvertBufferToTexture = "RenderPasses/DummyNeuralNetworkCV/buffToTexDynamicRealCV.cs.slang";
    const std::string kFinalOutput = "RenderPasses/DummyNeuralNetworkCV/combineCV.cs.slang";

    const std::string kYTrainBufferAssembly = "RenderPasses/DummyNeuralNetworkCV/assemblerCV.cs.slang";

    const std::string kNetworkConfig = "RenderPasses/DummyNeuralNetworkCV/network_config.json";
    const std::string kNetworkConfig1 = "RenderPasses/DummyNeuralNetworkCV/network_config_1.json";
    const std::string kPyramidConfig = "RenderPasses/DummyNeuralNetworkCV/pyramid_config.json";


    
    
    const std::string kPredBuffer = "gPredBuffer";
    const std::string kPredPixelID = "gPredBufferPixelID";
    const std::string kPredThroughput = "gPredBufferThroughput";

    const std::string kPredBufferOutActAddData = "gPredBufferOutActAddData";
    const std::string kXTrainBuffer = "gXTrainBuffer";
    const std::string kPredictedRadianceSTYBuffer = "gPredictedRadianceSTYBuffer";
    const std::string kLThpSelfTrainingBuffer = "gLThpSelfTrainingBuffer";
    const std::string kXTrainAdditionalBuffer = "gXTrainAdditionalBuffer";
    const std::string kYTrainBuffer = "gYTrainBuffer";
    
    // Mapping of idNN to index in X Buffers
    const std::string kNNMapPredBufferKeys = "gNNMapPredBufferKeys";
    const std::string kNNMapPredBufferValues = "gNNMapPredBufferValues";

    const std::string kNNMapTrainBufferKeys = "gNNMapTrainBufferKeys";
    const std::string kNNMapTrainBufferValues = "gNNMapTrainBufferValues";
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
        pybind11::class_<DummyNeuralNetworkCV, RenderPass, DummyNeuralNetworkCV::SharedPtr> type(m, "DummyNeuralNetworkCV");
        type.def_property_readonly("training_error", &Falcor::DummyNeuralNetworkCV::getTrainingError);
        type.def_property_readonly("training_epoch", &Falcor::DummyNeuralNetworkCV::getTrainingEpoch);
        type.def_property("diffuse_time", &Falcor::DummyNeuralNetworkCV::getDiffuseTime, &Falcor::DummyNeuralNetworkCV::setDiffuseTime);
        type.def("reloadModel", &Falcor::DummyNeuralNetworkCV::reloadModel);
    }
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("DummyNeuralNetworkCV", kDesc, Falcor::DummyNeuralNetworkCV::create);
    Falcor::ScriptBindings::registerBinding(Falcor::regNNPass);
}

Falcor::DummyNeuralNetworkCV::DummyNeuralNetworkCV() {
    mpConvertNNOutputPass = ComputePass::create(kConvertBufferToTexture);
    mpFinalOutputPass = ComputePass::create(kFinalOutput);
    mpYTrainAssemblyPass = ComputePass::create(kYTrainBufferAssembly);
}

Falcor::DummyNeuralNetworkCV::SharedPtr Falcor::DummyNeuralNetworkCV::create(Falcor::RenderContext* pRenderContext, const Falcor::Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new DummyNeuralNetworkCV);
    return pPass;
}

std::string Falcor::DummyNeuralNetworkCV::getDesc() { return kDesc; }

Falcor::Dictionary Falcor::DummyNeuralNetworkCV::getScriptingDictionary()
{
    return Falcor::Dictionary();
}

Falcor::RenderPassReflection Falcor::DummyNeuralNetworkCV::reflect(const CompileData& compileData)
{
    // Define the required resources here
    Falcor::RenderPassReflection reflector;
    reflector.addInput(kColorInput, "Input FrameBuffer");
    
    
    reflector.addInput(kPredBuffer, "Pred Buffer").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));

    reflector.addInput(kPredPixelID, "Pred Pixel ID").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addInput(kPredThroughput, "Pred Throughput").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));

    reflector.addInput(kPredBufferOutActAddData, "Prediction Buffer Additional Data for Output Activation").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addInternal(kPredictedRadianceMerged, "Prediction Radiance Merge").format(ResourceFormat::RGBA16Float).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));

    reflector.addInput(kXTrainBuffer, "X Train Buffer").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    
    reflector.addInput(kLThpSelfTrainingBuffer, "L and thp Self Train Buffer").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addInput(kXTrainAdditionalBuffer, "Additional Train Buffer").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    
    
    // Mapping of idNN to index in X Buffers
    reflector.addInput(kNNMapPredBufferKeys, "Map Prediction Buffer Keys").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addInput(kNNMapPredBufferValues, "Map Prediction Buffer Values").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    
    
    reflector.addInput(kNNMapTrainBufferKeys, "Map Train Buffer Keys").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    reflector.addInput(kNNMapTrainBufferValues, "Map Train Buffer Values").structuredBuffer(0, 0).bindFlags((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource));
    
    reflector.addOutput(kColorOutput, "Output FrameBuffer Predicted by Neural Network").format(ResourceFormat::RGBA32Float).bindFlags(ResourceBindFlags::UnorderedAccess);
    //reflector.addOutput("dst");
    //reflector.addInput("src");
    return reflector;
}


void Falcor::DummyNeuralNetworkCV::setScene(Falcor::RenderContext* pRenderContext, const Falcor::Scene::SharedPtr& pScene)
{
    mpScene = pScene;
}

#define SYNC_EVERYTHING 0
void Falcor::DummyNeuralNetworkCV::execute(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData)
{
    if (!mpScene)
    {
        return;
    }

    if (!mpFenceForPredictionData) mpFenceForPredictionData = GpuFence::create(true);
    if (!mpFenceOut) mpFenceOut = GpuFence::create(true);

    

    // Wait getting data for predicting

    PROFILE("Start");

    if (!mpCounterPredBuffer) {
        mpCounterPredBuffer = Buffer::create(1 * sizeof(uint32_t), ResourceBindFlags::None, Buffer::CpuAccess::Read);
        mpCounterTrainBuffer= Buffer::create(1 * sizeof(uint32_t), ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    Falcor::Buffer::SharedPtr xTrainBuff = renderData[kXTrainBuffer]->asBuffer();
    
    Falcor::Buffer::SharedPtr xLThpSelfTrainBuff = renderData[kLThpSelfTrainingBuffer]->asBuffer();
    Falcor::Buffer::SharedPtr xXTrainAddBuff = renderData[kXTrainAdditionalBuffer]->asBuffer();
    Falcor::Buffer::SharedPtr xPredBuff = renderData[kPredBuffer]->asBuffer();

    auto t = renderData[kColorInput]->asTexture();
    mScreenDim.x = t->getWidth(0);
    mScreenDim.y = t->getHeight(0);



    pRenderContext->copyBufferRegion(mpCounterTrainBuffer.get(), 0, xTrainBuff->getUAVCounter().get(), 0, 4);
    pRenderContext->copyBufferRegion(mpCounterPredBuffer.get(), 0, xPredBuff->getUAVCounter().get(), 0, 4);

    pRenderContext->flush(false || SYNC_EVERYTHING);
    uint64_t signalValue = mpFenceForPredictionData->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
    mpFenceForPredictionData->syncCpu();


    uint32_t* uavCounters = (uint32_t*)mpCounterTrainBuffer->map(Buffer::MapType::Read);
    uint32_t NumTrainingElements = uavCounters[0];
    mpCounterTrainBuffer->unmap();

    uint32_t* uavCounters2 = (uint32_t*)mpCounterPredBuffer->map(Buffer::MapType::Read);
    uint32_t NumPredictElements = uavCounters2[0];
    mpCounterPredBuffer->unmap();


    auto frameBuffer = renderData[kColorInput]->asTexture();
    uint32_t w = frameBuffer->getWidth();
    uint32_t h = frameBuffer->getHeight();
    uint32_t nElements = w * h*3;
    uint32_t nrcElements = w * h / (8 * 4);


    if (!model)
    {
#if 1
        //std::string dummy_path = "C:/Users/devmi/Documents/Projects/MSUNeuralRendering/ibvfteh-GMLNeuralRendering/Falcor/Source/";
        std::string dummy_path = Falcor::getWorkingDirectory();
        dummy_path = dummy_path.substr(0, dummy_path.find_last_of('\\') + 1);
#else
        const std::string dummy_path = "";
#endif
        std::vector<std::string> configs = {
            dummy_path+kNetworkConfig,
            dummy_path+kNetworkConfig1
        };
        std::cout << dummy_path << std::endl;
        std::cout << dummy_path + kPyramidConfig << std::endl;
        model.reset(new Falcor::NNLaplacianCV("NNLaplacian", configs, dummy_path + kPyramidConfig, 2, nElements+nrcElements, Falcor::AABB(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f)), &m_diffuse_time));
        //model->m_denoising_level = &m_diffuse_time;
    }

    nElements = NumPredictElements;
    nrcElements = 0;
    
    Falcor::Buffer::SharedPtr NNMapPredBufferKeys = renderData[kNNMapPredBufferKeys]->asBuffer();
    Falcor::Buffer::SharedPtr NNMapPredBufferValues = renderData[kNNMapPredBufferValues]->asBuffer();

    Falcor::Buffer::SharedPtr NNMapPredBufferOutActAddData = renderData[kPredBufferOutActAddData]->asBuffer();

    if (predColorBuffer == nullptr) {
        predColorBuffer = Falcor::Buffer::createStructured(3 * 4, w * h*4 + nrcElements, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
    }

    if (w * h * 4 < nElements) {
        throw std::exception("Please resize predColorBuffer");
    }

    Falcor::Buffer::SharedPtr NNMapTrainBufferKeys = renderData[kNNMapTrainBufferKeys]->asBuffer();
    Falcor::Buffer::SharedPtr NNMapTrainBufferValues = renderData[kNNMapTrainBufferValues]->asBuffer();

    // Wait prediction results
    pRenderContext->clearTexture(renderData[kPredictedRadianceMerged]->asTexture().get());

    if (nElements > 0)
    {

        PROFILE("NeuralNetwork Predict Start");
        // Just prediction

        bool predictedSucessfuly = false;
        predictedSucessfuly = model->predict(NumPredictElements+nrcElements, xPredBuff, predColorBuffer, NNMapPredBufferKeys, NNMapPredBufferValues, mpFenceForPredictionData, signalValue, mpFenceOut, NNMapPredBufferOutActAddData);

        if(SYNC_EVERYTHING)
            cudaDeviceSynchronize();


        if (predictedSucessfuly) {
            mpFenceOut->syncGpu(pRenderContext->getLowLevelData()->getCommandQueue());
            if (NumTrainingElements > 0 && model->getTrainFlag()) {
                PROFILE("NeuralNetwork Train Sorting IDs");
                model->fit_sorting(NumTrainingElements, NNMapTrainBufferKeys, NNMapTrainBufferValues);
            }


            {
                PROFILE("Merge predicted CV for all bounces to a intermediate frame buffer");
                mpConvertNNOutputPass["PerFrameCB"]["gElems"] = NumPredictElements;
                // Run mapping predicted data from structure buffer to final texture
                mpConvertNNOutputPass["gPredictedColor"] = predColorBuffer;
                mpConvertNNOutputPass[kPredPixelID] = renderData[kPredPixelID]->asBuffer();
                mpConvertNNOutputPass[kPredThroughput] = renderData[kPredThroughput]->asBuffer();
                mpConvertNNOutputPass[kPredictedRadianceMerged] = renderData[kPredictedRadianceMerged]->asTexture();
                mpConvertNNOutputPass->execute(pRenderContext, Falcor::uint3(NumPredictElements, uint32_t(1), uint32_t(1)));
            }
        }
    }

    {
        PROFILE("Merge estimated radiance and predicted one");
        mpFinalOutputPass[kPredictedRadianceMerged] = renderData[kPredictedRadianceMerged]->asTexture();
        mpFinalOutputPass["gEstimatedRadiance"] = renderData[kColorInput]->asTexture();
        mpFinalOutputPass[kColorOutput] = renderData[kColorOutput]->asTexture();
        mpFinalOutputPass->execute(pRenderContext, Falcor::uint3(w, h, uint32_t(1)));
    }

    if(SYNC_EVERYTHING)
        pRenderContext->flush(true);

   
    if (NumTrainingElements > 0 && model->getTrainFlag()) {
        uint32_t dims = (w / 8) * (h / 4) * 4;

        {
        PROFILE("Assembler Training Data");
        if (YTrainingBuffer == nullptr) {
            YTrainingBuffer = Falcor::Buffer::createStructured(sizeof(Falcor::YData), dims, ((ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)));
        }
        mpYTrainAssemblyPass["PerFrameCB"]["predictDataOffset"] = nElements;

        mpYTrainAssemblyPass["gPredictedRadianceSTYBuffer"] = predColorBuffer;
        mpYTrainAssemblyPass["gLThpSelfTrainingBuffer"] = xLThpSelfTrainBuff;
        mpYTrainAssemblyPass["gXTrainAdditionalBuffer"] = xXTrainAddBuff;
        mpYTrainAssemblyPass["gYTrainBuffer"] = YTrainingBuffer;

        mpYTrainAssemblyPass->execute(pRenderContext, Falcor::uint3(NumTrainingElements, uint32_t(1), uint32_t(1)));


        pRenderContext->flush(false);
        if (!mpFenceForTrainingData) mpFenceForTrainingData = GpuFence::create(true);

        signalValue = mpFenceForTrainingData->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
        }
        // TODO: coordBuffer and colorBuffer generation

        {
            PROFILE("NeuralNetwork Train");

            model->fit(NumTrainingElements, xTrainBuff, YTrainingBuffer, NNMapTrainBufferKeys, NNMapTrainBufferValues, mpFenceForTrainingData, signalValue);
        }
    }


    m_training_epoch = model->getTrainingEpoch();
    m_training_error = model->getTrainingError();
}

void Falcor::DummyNeuralNetworkCV::reloadModel() {
    model->reloadModel();
}

void Falcor::DummyNeuralNetworkCV::renderUI(Falcor::Gui::Widgets& widget)
{
    model->renderUI(widget);
}

int32_t Falcor::DummyNeuralNetworkCV::getTrainingEpoch() const {
    return m_training_epoch;
}

float Falcor::DummyNeuralNetworkCV::getTrainingError() const {
    return m_training_error;
}

float Falcor::DummyNeuralNetworkCV::getDiffuseTime() const {
    return m_diffuse_time;
}


void Falcor::DummyNeuralNetworkCV::setDiffuseTime(float t){
    m_diffuse_time = t;
}

