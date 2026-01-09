
#pragma once
#include "FalcorCUDA.h"
#include "RenderPasses/Shared/PathTracer/PathTracer.h"

#include "Falcor.h"

namespace Falcor {
    class SimpleNNModel;

    class NNPathGuiding : public PathTracer
    {
    public:
        using SharedPtr = std::shared_ptr<NNPathGuiding>;

        /** Create a new render pass object.
            \param[in] pRenderContext The render context.
            \param[in] dict Dictionary of serialized parameters.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(Falcor::RenderContext* pRenderContext = nullptr, const Falcor::Dictionary& dict = {});

        virtual std::string getDesc() override { return sDesc; }
    
        virtual void execute(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData) override;
        virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
        virtual void renderUI(Gui::Widgets& widget) override;
    
        static const char* sDesc;

    private:
        NNPathGuiding(const Dictionary& dict);
        Falcor::Buffer::SharedPtr predXBuffer;
        Falcor::Buffer::SharedPtr predYBuffer;
        Falcor::Buffer::SharedPtr trainYBuffer;
        Falcor::Texture::SharedPtr guidingLobesTextures;


        
        Falcor::ComputePass::SharedPtr          mpGenerateTrainingDataPass;
        Falcor::ComputePass::SharedPtr          mpConvertNNOutputPass;
        Falcor::ComputePass::SharedPtr          mpPreparePredictDataPass;

        Falcor::SimpleNNModel* model;

        Falcor::Buffer::SharedPtr mpCounterBuffer;
        uint2 mScreenDim = { 1920, 1080 };
        Falcor::Scene::SharedPtr mpScene;

        bool bGuiding= false;

        void recreateVars() override { mTracer.pVars = nullptr; }
        void prepareVars();
        void setTracerData(const RenderData& renderData);

        // Ray tracing program.
        struct
        {
            RtProgram::SharedPtr pProgram;
            RtBindingTable::SharedPtr pBindingTable;
            RtProgramVars::SharedPtr pVars;
            ParameterBlock::SharedPtr pParameterBlock;      ///< ParameterBlock for all data.
        } mTracer;
    };
}
