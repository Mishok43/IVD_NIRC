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
#include "FalcorCUDA.h"

#include "Falcor.h"
#include "NNLaplacianCV.h"

namespace Falcor {
    class SimpleNNModel;




class DummyNeuralNetworkCV : public Falcor::RenderPass
{
public:
    using SharedPtr = std::shared_ptr<DummyNeuralNetworkCV>;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(Falcor::RenderContext* pRenderContext = nullptr, const Falcor::Dictionary& dict = {});

    virtual std::string getDesc() override;
    virtual Falcor::Dictionary getScriptingDictionary() override;
    virtual Falcor::RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(Falcor::RenderContext* pContext, const CompileData& compileData) override {}
    virtual void execute(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData) override;
    virtual void renderUI(Falcor::Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual bool onMouseEvent(const Falcor::MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const Falcor::KeyboardEvent& keyEvent) override { return false; }
    DummyNeuralNetworkCV();


    void reloadModel();
    float getTrainingError() const;
    int32_t getTrainingEpoch() const;
    float getDiffuseTime() const;
    void setDiffuseTime(float t);

    float m_training_error;
    int32_t m_training_epoch;
    float m_diffuse_time;
private:
    Falcor::Buffer::SharedPtr predColorBuffer;
    
    Falcor::Buffer::SharedPtr YTrainingBuffer;
    Falcor::ComputePass::SharedPtr          mpGenerateTrainingDataPass;
    Falcor::ComputePass::SharedPtr          mpConvertNNOutputPass;
    Falcor::ComputePass::SharedPtr          mpFinalOutputPass;
    Falcor::ComputePass::SharedPtr          mpYTrainAssemblyPass;
    bool skipFirstPred = true;

    std::shared_ptr<Falcor::NNLaplacianCV> model;

    Falcor::Buffer::SharedPtr mpCounterTrainBuffer;
    Falcor::Buffer::SharedPtr mpCounterPredBuffer;
    uint2 mScreenDim = { 1920, 1080 };
    Falcor::Scene::SharedPtr mpScene;

    GpuFence::SharedPtr mpFenceForPredictionData;
    GpuFence::SharedPtr mpFenceForTrainingData;
    GpuFence::SharedPtr mpFenceOut;
    bool bInitCudaFence = false;

    
};
}
