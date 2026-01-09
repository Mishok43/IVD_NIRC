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
#include "ImageUIPresenter.h"


namespace
{
    const char kDesc[] = "Insert pass description here";    
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("ImageUIPresenter", kDesc, ImageUIPresenter::create);
}

ImageUIPresenter::SharedPtr ImageUIPresenter::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new ImageUIPresenter);
    return pPass;
}

std::string ImageUIPresenter::getDesc() { return kDesc; }

Dictionary ImageUIPresenter::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection ImageUIPresenter::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addOutput("dst", "lalal");
    //reflector.addInput("src");

    reflector.addInput("src", "real input that is transfered as an output");
    for (uint32_t i = 0; i < 10; i++) {
        reflector.addInput("src" + std::to_string(i), "not used input").flags(RenderPassReflection::Field::Flags::Optional);
    }

    return reflector;
}

void ImageUIPresenter::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpImages.clear();
    if (mpImages.empty()) {
        mpImages.push_back(renderData["src"]->asTexture());

        for (uint32_t i = 0; i < 10; i++) {
            auto res = renderData["src" + std::to_string(i)];
            if (!res)
                break;

            mpImages.push_back(res->asTexture());
        }    
    }

    // renderData holds the requested resources
    // auto& pTexture = renderData["src"]->asTexture();
}

void ImageUIPresenter::renderUI(Gui::Widgets& widget)
{
    for (uint32_t i = 0; i < mpImages.size(); i++) {
        std::string k = "Image " + std::to_string(i);
        widget.image(k.c_str(), mpImages[i], { 200, 200 });
    }


    if (widget.button("Save Images")) {
        auto p = std::filesystem::path(__FILE__).remove_filename();

        for (uint32_t i = 0; i < mpImages.size(); i++) {
            auto p2 = p;

            
            if (mpImages[i]->asTexture()->getFormat() == Falcor::ResourceFormat::RGBA32Float) {
                p2 /= "im" + std::to_string(i) + ".exr";
                mpImages[i]->captureToFile(0, 0, p2.u8string(), Falcor::Bitmap::FileFormat::ExrFile, Falcor::Bitmap::ExportFlags::Uncompressed);
            }
            else {
                p2 /= "im" + std::to_string(i) + ".png";
                mpImages[i]->captureToFile(0, 0, p2.u8string(), Falcor::Bitmap::FileFormat::PngFile);

            }

        }
    }
}
