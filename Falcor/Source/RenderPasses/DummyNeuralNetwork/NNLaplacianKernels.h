#include "NNLaplacianDataTypes.h"

#define NAN_DEBUG 0

struct ACESSettings {
    float a, b, c, d, e;
};

__device__ float aces_tonemap(float x, ACESSettings s) {
    x *= 0.6;
    float r = (x * x * s.a + s.b * x) / (x * x * s.c + x * s.d + s.e);
    if (r <= 0.0) {
        return 0.0;
    }
    if (r > 1.0) {
        return 1.0;
    }
    return r;
}

__device__ cFloat3 aces_tonemap_rgb(cFloat3 c, ACESSettings s) {
    return cFloat3(aces_tonemap(c.x, s), aces_tonemap(c.y, s), aces_tonemap(c.z, s));
}

__device__ float aces_tonemap_der(float x, ACESSettings s, float val) {
    x *= 0.6;
    if (x <= 0.0) {
        return -0.2*0.6;
    }
    else {
        float t = x * (s.c * x + s.d) + s.e;
        return  0.6 * (s.a * x * (s.d * x + 2 * s.e) + s.b * (s.e - s.c * x * x)) / (t * t);
    }
}

__device__ cFloat3 aces_tonemap_der_rgb(cFloat3 x, ACESSettings s, cFloat3 val) {
    return cFloat3(aces_tonemap_der(x.x, s, val.x), aces_tonemap_der(x.y, s, val.y), aces_tonemap_der(x.z, s, val.z));
}


__global__ void assemblyTrainingDataKernel(unsigned int numElements, cFloat3* __restrict__ predictedRadiance, cTrainingAdditionalData* __restrict__ trainSamples, cLThp* __restrict__ lightPaths, cYData* __restrict__ output, cFloat3* __restrict__ resError, cFloat3* __restrict__ varianceLoss, unsigned int* __restrict__ varianceLossMapping, cFloat3* __restrict__ estimations, cFloat3* __restrict__ varianceLossPart, float outScale = 1.0f)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElements) return;

    cTrainingAdditionalData xadd = trainSamples[i];




    cLThp xlthp = lightPaths[xadd.lightPathID];
    // cFloat3 resErrorV;
    // if (resError != nullptr)
    //     resErrorV = resError[xadd.lightPathID];


    cFloat3 surfaceRad = xlthp.L;
    cFloat3 thp = xadd.thp;

    if (predictedRadiance != nullptr) {
        cFloat3 predRadiance = predictedRadiance[xadd.lightPathID];
        surfaceRad.add(predRadiance);
    }
    cFloat3 addv = xadd.att;

    cFloat3 thpDiv = xadd.thpDiv;
    float pdf = xadd.pdf;


#if 0
    thpDiv.add(cFloat3(0.000001));
    surfaceRad.subst(xadd.sub);
    surfaceRad.div(thpDiv);
    thp.div(thpDiv);
#else
    thpDiv = 1.0f;
#endif


#if 0
    surfaceRad.subst(xadd.att);
    addv = cFloat3();
#endif

#if 0
    surfaceRad.subst(xadd.att);
    surfaceRad.clamp(0.0f, 100000.0f);

    cFloat3 divider = xadd.thpDiv;
    divider.add(0.0001);
    surfaceRad.div(divider);
    thp.div(divider);
    addv = cFloat3();
    // if(thpDiv.x == 1.0 && thpDiv.y == 1.0 && thpDiv.z == 1.0)
    //     thpDiv = 0.0;
    // else{
    //     thpDiv = 1.0f;
    //     thp = 1.0f;
    //     surfaceRad = 1.0f;
    //     pdf = 1.0f;
    // }


    surfaceRad = 1.0f;

    //surfaceRad.add(xadd.att);
#else
    //thpDiv = 1.0f;
#endif

    //surfaceRad.subst(xadd.att);
    //surfaceRad.clamp(0.0f, 100000.0f);

    //cFloat3 pathRadiance = (predictedRadiance * xlthp.thp + surfaceRad) / xadd.thp;

    //predRadiance.mult(xlthp.thp);
    //surfaceRad.add(predRadiance);
    //predRadiance.div(xadd.thp);

    //surfaceRad.clamp(0.0f, 100000.0f);
    //surfaceRad.subst(resErrorV);
    //surfaceRad.mult(outScale);
    //predRadiance.mult(xadd.pdf);



    if (varianceLoss != nullptr && varianceLossMapping != nullptr && 0) {
        unsigned int lossIndex = varianceLossMapping[i];
        cFloat3 loss = varianceLoss[lossIndex];
        //addv = surfaceRad;
        surfaceRad.add(loss);
        addv = varianceLossPart[lossIndex];
        if (estimations != nullptr) {
            cFloat3 expectedValue = estimations[lossIndex];
            // addv = expectedValue;
            // addv.mult(-1);
            if (0) {
                ACESSettings s;
                s.a = 2.51;
                s.b = 0.03;
                s.c = 2.43;
                s.d = 0.59;
                s.e = 0.14;

                //surfaceRad.absv();

                float exposure = 1.0f;

                //float exposure = 4.0f;
                surfaceRad.mult(exposure);
                cFloat3 tmp = aces_tonemap_rgb(surfaceRad, s);
                cFloat3 der_tonemap = aces_tonemap_der_rgb(surfaceRad, s, tmp);
                //der_tonemap *= exposure;
                der_tonemap.mult(exposure);
                surfaceRad = tmp;

                //addv = der_tonemap;

                cFloat3 tmpExpectedVal = expectedValue;

                tmpExpectedVal.mult(-exposure);
                //tmpExpectedVal.clamp(0.0f, 10000000.0f);
                tmpExpectedVal = aces_tonemap_rgb(tmpExpectedVal, s);

                surfaceRad.subst(tmpExpectedVal);
                addv = der_tonemap;
            }
            else
                surfaceRad.add(expectedValue);
        }

        //addv = loss;
        //surfaceRad = loss;
        pdf = 0.0f; // just marker of using variance loss
    }
    else if (estimations != nullptr && varianceLossPart != nullptr && 0) {

#if 0
        surfaceRad = estimations[i];

#else
        ACESSettings s;
        s.a = 2.51;
        s.b = 0.03;
        s.c = 2.43;
        s.d = 0.59;
        s.e = 0.14;

        cFloat3 ourEstimator = varianceLossPart[xadd.lightPathID];
        cFloat3 gtEstimator = estimations[i];

        float exposure = 1.0f;
        ourEstimator.mult(exposure);
        gtEstimator.mult(exposure);

#if 0
        cFloat3 ourTMP = aces_tonemap_rgb(ourEstimator, s);
        cFloat3 gtTMP = aces_tonemap_rgb(gtEstimator, s);
#else
        cFloat3 ourTMP = ourEstimator;
        cFloat3 gtTMP = gtEstimator;
#endif
        gtTMP.clamp(0.0f, 1.0f);
        ourTMP.clamp(0.0f, 1.0f);

        ourTMP.sqrtv();
        gtTMP.sqrtv();

        ourTMP.subst(gtTMP);
        float weight = ourTMP.sqr_length();
        pdf = weight / 3.0;
#endif

        //surfaceRad = cFloat3(1.0);
        //surfaceRad.subst(xadd.att);
    }
    else if (estimations != nullptr) {
        // surfaceRad.subst(xadd.att);
        // addv = 0;
        // //surfaceRad.mult(xadd.pdf);
        // surfaceRad.add(estimations[i]);
    }
    else if (resError != nullptr) {
        // surfaceRad.subst(xadd.att);
        // addv = 0;
        // surfaceRad.add(resError[i]);
    }

#if NAN_DEBUG

    if (surfaceRad.isanynan()) {
        surfaceRad.x *= 0.999999;
    }

#endif

    //xadd.thp.mult(xadd.pdf);
    cYData ydata;
    ydata.radiance = surfaceRad;
    ydata.weight_thp = thp;
    ydata.weight_add = addv;
    ydata.pdf = pdf;
    ydata.weight_thp_div = thpDiv;

    output[i] = ydata;
}


__global__ void accumulateRadianceForResidualError(unsigned int numTrainingElements, cFloat3* __restrict__  predictedRadiance, cFloat3* __restrict__ outputRadiance, cResErrorData* __restrict__ samplesMapping, cFloat3* __restrict__ samplesThroughput, float outScaleInv = 1.0f)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTrainingElements) return;

    cResErrorData resData = samplesMapping[i];
    cFloat3 res = resData.estimation;

    if (resData.id >= 0)
    {
        cFloat3 predColor = predictedRadiance[resData.id];
        cFloat3 thoughput = samplesThroughput[resData.id];
        predColor.clamp(0, 100000);
        predColor.mult(outScaleInv);
        predColor.mult(thoughput);
        res.add(predColor);
    }

    outputRadiance[i] = res;
}


__global__ void accumulateRadianceClamp(unsigned int numPredictedElements, cFloat3* __restrict__  predictedRadiance, float* __restrict__ outputRadiance, cFloat3* __restrict__ samplesThroughput, float outScaleInv = 1.0f, bool bClamp = true)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPredictedElements) return;

    cFloat3 predColor = predictedRadiance[i];
    cFloat3 thoughput = samplesThroughput[i];

    if(bClamp && 0)
        predColor.clamp(0, 100000);




    predColor.mult(outScaleInv);
    predColor.mult(thoughput);

#if NAN_DEBUG

    if (predColor.isanynan()) {
        predColor.x *= 0.999999;
    }

#endif

    outputRadiance[i * 3] =  predColor.x;
    outputRadiance[i * 3 + 1] = predColor.y;
    outputRadiance[i * 3 + 2] = predColor.z;
}


__global__ void accumulateRadiance(unsigned int numPredictedElements, cFloat3* __restrict__  predictedRadiance, float* __restrict__ outputRadiance, unsigned int* __restrict__ samplesMapping, cFloat3* __restrict__ samplesThroughput, float outScaleInv = 1.0f)
{
    unsigned int i = blockIdx.x * blockDim.x    + threadIdx.x;
    if (i >= numPredictedElements) return;

    cFloat3 predColor = predictedRadiance[i];
    cFloat3 thoughput = samplesThroughput[i];
    unsigned int outID = samplesMapping[i];

    predColor.clamp(0, 100000);
    //predColor.mult(outScaleInv);
    predColor.mult(thoughput);

#if NAN_DEBUG

    if (thoughput.x == 0.0f && thoughput.y == 0.0f && thoughput.z == 0.0f) {
        predColor = cFloat3();
    }

    if (predColor.isanynan()) {
        predColor.x *= 0.999999;
    }

#endif

#if 1
    atomicAdd(&outputRadiance[outID * 3], predColor.x);
    atomicAdd(&outputRadiance[outID * 3 + 1], predColor.y);
    atomicAdd(&outputRadiance[outID * 3 + 2], predColor.z);
#else
    outputRadiance[outID * 3] =  predColor.x;
    outputRadiance[outID * 3 + 1] = predColor.y;
    outputRadiance[outID * 3 + 2] = predColor.z;
#endif
}



__global__ void accumulateRadianceOnlyPart(unsigned int numPredictedElements, cFloat3* __restrict__  predictedRadiance, float* __restrict__ outputRadiance, unsigned int* __restrict__ samplesMapping, cFloat3* __restrict__ samplesThroughput, float outScaleInv = 1.0f)
{
    unsigned int i = blockIdx.x * blockDim.x    + threadIdx.x;
    if (i >= numPredictedElements) return;

    cFloat3 predColor = predictedRadiance[i];
    cFloat3 thoughput = samplesThroughput[i];

    if(thoughput.x < 0.0 || thoughput.y < 0.0 || thoughput.z < 0.0)
        return;

    unsigned int outID = samplesMapping[i];

    //predColor.clamp(0, 100000);
    //predColor.mult(outScaleInv);
    predColor.mult(thoughput);

#if NAN_DEBUG
    if (thoughput.x == 0.0f && thoughput.y == 0.0f && thoughput.z == 0.0f) {
        predColor = cFloat3();
    }

    if (predColor.isanynan()) {
        predColor.x *= 0.999999;
    }
#endif

#if 1
    atomicAdd(&outputRadiance[outID * 3], predColor.x);
    atomicAdd(&outputRadiance[outID * 3 + 1], predColor.y);
    atomicAdd(&outputRadiance[outID * 3 + 2], predColor.z);
#else
    outputRadiance[outID * 3] =  predColor.x;
    outputRadiance[outID * 3 + 1] = predColor.y;
    outputRadiance[outID * 3 + 2] = predColor.z;
#endif
}
