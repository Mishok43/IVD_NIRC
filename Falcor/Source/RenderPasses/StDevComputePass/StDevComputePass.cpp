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
#include "StDevComputePass.h"


namespace
{
    const char kDesc[] = "Insert pass description here";
    const std::string kComputePass = "RenderPasses/StDevComputePass/stdev_compute.cs.slang";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("StDevComputePass", kDesc, StDevComputePass::create);
}

StDevComputePass::StDevComputePass() {
    mpComputePass = ComputePass::create(kComputePass);
}

StDevComputePass::SharedPtr StDevComputePass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new StDevComputePass);
    return pPass;
}

std::string StDevComputePass::getDesc() { return kDesc; }

Dictionary StDevComputePass::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection StDevComputePass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    //reflector.addOutput("dst", "desc").format(ResourceFormat::R32f);
    reflector.addInput("moment1", "desc");
    reflector.addInput("moment2", "desc");

    return reflector;
}

void StDevComputePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // renderData holds the requested resources
    // auto& pTexture = renderData["src"]->asTexture();
    if (!mEnabled)
        return;

    //// Create parallel reduction helper.
    //if (!mpParallelReduction)
    //{
    //    mpParallelReduction = ComputeParallelReduction::create();
    //    mpReductionResult = Buffer::create(, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    //}

    const uint2 targetDim = renderData.getDefaultTextureDims();
    output = renderData["dst"]->asTexture();

    mpComputePass["gFirstMoment"] = renderData["moment1"]->asTexture();
    mpComputePass["gSecondMoment"] = renderData["moment2"]->asTexture();
    mpComputePass["colorOutput"] = renderData["dst"]->asTexture();
    
    mpComputePass->execute(pRenderContext, Falcor::uint3(targetDim.x, targetDim.y, uint32_t(1)));
}

std::filesystem::path StDevComputePass::getOutputPath() const
{
    auto path = std::filesystem::path(mOutputDir);
    if (!path.is_absolute()) path = std::filesystem::absolute(std::filesystem::path(getExecutableDirectory()) / path);
    return path;
}

void StDevComputePass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Is Enabled", mEnabled);

    std::string folder;

    if (widget.button("Change Folder") && chooseFolderDialog(folder)) {
        std::filesystem::path path(folder);
        if (path.is_absolute())
        {
            // Use relative path to executable directory if possible.
            auto relativePath = path.lexically_relative(getExecutableDirectory());
            if (!relativePath.empty() && relativePath.string().find("..") == std::string::npos) path = relativePath;
        }
        mOutputDir = path.string();
    }
        
    if (widget.button("Save")) {
        auto path = getOutputPath();
        path /= "std.exr";
        output->captureToFile(0, 0, path.string(), Falcor::Bitmap::FileFormat::ExrFile);
    }
}



