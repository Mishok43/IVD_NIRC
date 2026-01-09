from falcor import *

def render_graph_DefaultRenderGraph():
    g = RenderGraph('DefaultRenderGraph')
    loadRenderPassLibrary('BSDFViewer.dll')
    loadRenderPassLibrary('ErrorMeasurePass.dll')
    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary('PassLibraryTemplate.dll')
    loadRenderPassLibrary('Antialiasing.dll')
    loadRenderPassLibrary('BlitPass.dll')
    loadRenderPassLibrary('CSM.dll')
    loadRenderPassLibrary('DummyNeuralNetwork.dll')
    loadRenderPassLibrary('DebugPasses.dll')
    loadRenderPassLibrary('DepthPass.dll')
    loadRenderPassLibrary('FLIPPass.dll')
    loadRenderPassLibrary('ForwardLightingPass.dll')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('ImageLoader.dll')
    loadRenderPassLibrary('MegakernelPathTracer.dll')
    loadRenderPassLibrary('WhittedRayTracer.dll')
    loadRenderPassLibrary('MGPTforNRC.dll')
    loadRenderPassLibrary('MGPTInterim.dll')
    loadRenderPassLibrary('MinimalPathTracer.dll')
    loadRenderPassLibrary('SVGFPass.dll')
    loadRenderPassLibrary('OptixDenoiser.dll')
    loadRenderPassLibrary('PixelInspectorPass.dll')
    loadRenderPassLibrary('ToneMapper.dll')
    loadRenderPassLibrary('TestPasses.dll')
    loadRenderPassLibrary('SceneDebugger.dll')
    loadRenderPassLibrary('SimplePostFX.dll')
    loadRenderPassLibrary('TemporalDelayPass.dll')
    loadRenderPassLibrary('SkyBox.dll')
    loadRenderPassLibrary('SSAO.dll')
    loadRenderPassLibrary('Utils.dll')
    loadRenderPassLibrary('ImageUIPresenter.dll')
    loadRenderPassLibrary('DummyPass.dll')
    

    VBufferRT = createPass('VBufferRT', {'samplePattern': SamplePattern(3), 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': CullMode(2), 'useTraceRayInline': False})
    g.addPass(VBufferRT, 'VBufferRT')
    DummyNeuralNetwork = createPass('DummyNeuralNetwork', {'params': PathTracerParams(samplesPerPixel=1, lightSamplesPerVertex=1, maxBounces=5, maxNonSpecularBounces=5, useVBuffer=1, useAlphaTest=1, adjustShadingNormals=0, forceAlphaOne=1, clampSamples=0, clampThreshold=10.0, specularRoughnessThreshold=0.25, useBRDFSampling=1, useNEE=1, useMIS=1, misHeuristic=1, misPowerExponent=2.0, useRussianRoulette=1, probabilityAbsorption=0.20000000298023224, useFixedSeed=0, useNestedDielectrics=1, useLightsInDielectricVolumes=0, disableCaustics=0, rayFootprintMode=0, rayConeMode=2, rayFootprintUseRoughness=0), 'sampleGenerator': 1, 'emissiveSampler': EmissiveLightSamplerType(1), 'uniformSamplerOptions': LightBVHSamplerOptions(buildOptions=LightBVHBuilderOptions(splitHeuristicSelection=SplitHeuristic(2), maxTriangleCountPerLeaf=10, binCount=16, volumeEpsilon=0.0010000000474974513, splitAlongLargest=False, useVolumeOverSA=False, useLeafCreationCost=True, createLeavesASAP=True, allowRefitting=True, usePreintegration=True, useLightingCones=True), useBoundingCone=True, useLightingCone=True, disableNodeFlux=False, useUniformTriangleSampling=True, solidAngleBoundMethod=SolidAngleBoundMethod(3))})
    g.addPass(DummyNeuralNetwork, 'DummyNeuralNetwork')
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass, "ToneMappingPass")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False})
    g.addPass(AccumulatePass, "AccumulateFrontBuffer")
    AccumulatePass2Moment = createPass("AccumulatePass", {'enabled': False, "secondMoment": True, "precisionMode": AccumulatePrecision(0)})
    g.addPass(AccumulatePass2Moment, "AccumulatePass2Moment")

    g.addEdge('VBufferRT.vbuffer', 'DummyNeuralNetwork.vbuffer')
    g.addEdge("DummyNeuralNetwork.colorOutputFinal", "AccumulateFrontBuffer.input")
    g.addEdge("AccumulateFrontBuffer.output", "ToneMappingPass.src")
    ToneMappingPass2 = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass2, "ToneMappingDebug")
    AccumulatePass3 = createPass("AccumulatePass", {'enabled': False, "frameResX": 256, "frameResY": 256})
    g.addPass(AccumulatePass3, "AccumulateDebug")
    g.addEdge("DummyNeuralNetwork.colorOutputDebug", "AccumulateDebug.input")
    g.addEdge("AccumulateDebug.output", "ToneMappingDebug.src")

    AccumulatePass2 = createPass("AccumulatePass", {'enabled': False, "frameResX": 256, "frameResY": 256})
    g.addPass(AccumulatePass2, "AccumulateGroundTruth")
    ToneMappingPass3 = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass3, "ToneMappingDebugGT")

    ToneMappingComposite = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingComposite, "ToneMappingComposite")


    g.addEdge("DummyNeuralNetwork.colorOutputDebugGT", "AccumulateGroundTruth.input")
    g.addEdge("AccumulateGroundTruth.output", "ToneMappingDebugGT.src")
    

    ImageLoader = createPass("ImageLoader")
    g.addPass(ImageLoader, "ImageLoader")

    ImageLoaderHDR = createPass("ImageLoader", {"outputFormat": 27})
    
    g.addPass(ImageLoaderHDR, "ImageLoaderHDR")
 

    ImageLoaderBiased = createPass("ImageLoader", {"outputFormat": 27})
    g.addPass(ImageLoaderBiased, "ImageLoaderBiased")
    g.addEdge("ImageLoaderBiased.dst", "DummyNeuralNetwork.ourEstimations")
    g.addEdge("ImageLoaderHDR.dst", "DummyNeuralNetwork.gtEstimations")


    #g.addEdge("ImageLoaderBiased.dst", "DummyPass.srcDummy")

    ImageLoaderFLIP = createPass("ImageLoader", {"outputFormat": 27})
    g.addPass(ImageLoaderFLIP, "ImageLoaderFLIP")

    ImageLoaderStDev = createPass("ImageLoader", {"outputFormat": 27})
    g.addPass(ImageLoaderStDev, "ImageLoaderStDev")

    FLIPPass = createPass("FLIPPass", {'calculatePerFrameFLIP': True, 'forwardInputToOutput': False, "enabled": False})
    g.addPass(FLIPPass, "FLIPPass")

    g.addEdge('ImageLoader.dst', 'FLIPPass.inputA')
    g.addEdge('ToneMappingPass.dst', 'FLIPPass.inputB')
    
    DummyPass = createPass("DummyPass")
    g.addPass(DummyPass, "DummyPass")

    #g.markOutput("FLIPPass.output")

    RelativeErrorSquared = createPass("ErrorMeasurePass", {'RunningErrorSigma': 0.0, "Enabled": False, "Relative": True})
    g.addPass(RelativeErrorSquared, "RelativeErrorSquared")

    RelativeBiasSquared = createPass("ErrorMeasurePass", {'RunningErrorSigma': 0.0, "Enabled": False, "Relative": True})
    g.addPass(RelativeBiasSquared, "RelativeBiasSquared")

    AccumulatePassFLIP = createPass("AccumulatePass", {'enabled': False})
    g.addPass(AccumulatePassFLIP, "AccumulatePassFLIP")


  


    ImageUIPresenter = createPass("ImageUIPresenter")
    g.addPass(ImageUIPresenter, "ImageUIPresenter")

    ImageUIPresenterHDR = createPass("ImageUIPresenter")
    g.addPass(ImageUIPresenterHDR, "ImageUIPresenterHDR")

    RelativeVariance = createPass("ErrorMeasurePass", {'RunningErrorSigma': 0.0, "Enabled": False, "Variance": True, "Relative": True})
    g.addPass(RelativeVariance, "RelativeVariance")


    ErrorMeasurePassFLIP = createPass("ErrorMeasurePass", {'RunningErrorSigma': 0.0, "SelectedOutputId": 2, "Enabled": False, "SignedError": True})
    g.addPass(ErrorMeasurePassFLIP, "ErrorMeasurePassFLIP")

    ImageUIPresenterFLIP = createPass("ImageUIPresenter")
    g.addPass(ImageUIPresenterFLIP, "ImageUIPresenterFLIP")

    ErrorMeasurePassHDR = createPass("ErrorMeasurePass", {'RunningErrorSigma': 0.0, "Enabled": False, "Relative": True})
    g.addPass(ErrorMeasurePassHDR, "ErrorMeasurePassHDR")

    CompositePass = createPass("Composite", {'mode': 2})
    g.addPass(CompositePass, "CompositePass")



    g.addEdge("ImageLoaderHDR.dst", "ErrorMeasurePassHDR.Reference")
    g.addEdge("DummyNeuralNetwork.colorOutputFinal", "ErrorMeasurePassHDR.Source")
    g.addEdge("ToneMappingDebug.dst", "ImageUIPresenter.src")
    g.addEdge("ToneMappingDebugGT.dst", "ImageUIPresenter.src0")
    g.addEdge("ErrorMeasurePassFLIP.Output", "ImageUIPresenterFLIP.src")
    g.addEdge('ToneMappingComposite.dst', 'ImageUIPresenter.src1')

    g.addEdge("AccumulateDebug.output", "ImageUIPresenterHDR.src")
    g.addEdge("AccumulateGroundTruth.output", "ImageUIPresenterHDR.src0")

    g.addEdge("DummyNeuralNetwork.colorOutputFinal", "AccumulatePass2Moment.input")
    

    g.addEdge("AccumulateFrontBuffer.output", "RelativeVariance.Source")
    g.addEdge("AccumulatePass2Moment.output", "RelativeVariance.Reference")

    g.addEdge("ImageLoader.dst", "RelativeErrorSquared.Reference")
    g.addEdge("ToneMappingPass.dst", "RelativeErrorSquared.Source")

    g.addEdge("ImageLoaderHDR.dst", "RelativeBiasSquared.Reference")
    g.addEdge("AccumulateFrontBuffer.output", "RelativeBiasSquared.Source")



    g.addEdge('ToneMappingPass.dst', 'DummyPass.src')
    g.addEdge('FLIPPass.output', 'AccumulatePassFLIP.input')
    g.addEdge('ImageLoaderFLIP.dst', 'ErrorMeasurePassFLIP.Reference')

    g.addEdge('FLIPPass.output', 'ErrorMeasurePassFLIP.Source')

    g.addEdge('AccumulateDebug.output', 'CompositePass.A')
    g.addEdge('AccumulateGroundTruth.output', 'CompositePass.B')
    g.addEdge('CompositePass.out', 'ToneMappingComposite.src')



    g.addEdge('RelativeErrorSquared.Output', 'DummyPass.srcDummy0')
    g.addEdge('ErrorMeasurePassHDR.Output', 'DummyPass.srcDummy1')
    g.addEdge('DummyNeuralNetwork.colorOutputDebug', 'DummyPass.srcDummy2')
    g.addEdge('DummyNeuralNetwork.colorOutputDebugGT', 'DummyPass.srcDummy3')
    g.addEdge("ImageUIPresenter.dst", "DummyPass.srcDummy4")
    g.addEdge('ErrorMeasurePassFLIP.Output', 'DummyPass.srcDummy5')
    g.addEdge('FLIPPass.output', 'DummyPass.srcDummy6')
    g.addEdge("ImageUIPresenterFLIP.dst", "DummyPass.srcDummy7")
    g.addEdge('AccumulatePassFLIP.output', 'DummyPass.srcDummy8')
    g.addEdge('ToneMappingComposite.dst', 'DummyPass.srcDummy9')
    
    g.addEdge("RelativeVariance.Output", "DummyPass.srcDummy11")
    g.addEdge("ImageUIPresenterHDR.dst", "DummyPass.srcDummy12")
    g.addEdge('RelativeBiasSquared.Output', 'DummyPass.srcDummy13')



    #g.addEdge('DummyNeuralNetwork.gExpectedValueTmp', 'DummyPass.srcDummy1')
    g.markOutput("DummyPass.dst")
    #g.markOutput("ToneMappingPass.dst")


    return g

DefaultRenderGraph = render_graph_DefaultRenderGraph()
try: m.addGraph(DefaultRenderGraph)
except NameError: None
