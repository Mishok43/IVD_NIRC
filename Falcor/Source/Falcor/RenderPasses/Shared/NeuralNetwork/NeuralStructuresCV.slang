#pragma once
#include "Utils/HostDeviceShared.slangh"

#define ONLY_POS 1
#define DIRECT_CACHE 1
BEGIN_NAMESPACE_FALCOR
struct LThp
{
    float3 L;
    float3 thp;
};

struct TrainingAdditionalData
{
    uint lightPathID;
    float3 att;
    float3 thp;
    float3 incidentDir;
    float3 dirL;
    float pdf;
};

struct YData{
    float3 radiance;
    float3 incidentDir;
};  

struct OutputTransformData{
    float3 normal;
    float3 incidentDir;
    float3 viewDir;
    float roughness;
    float pdf;
    float3 diffuseAlbedoInvPi;
    float3 specAlbedo;
    float specFactor;
    float only_cv;
};

struct XInferenceData
{
#if !ONLY_POS
    float3 Position;
    float3 ScatteredDir;
    float3 SurfaceNormal;
    float SurfaceRoughness;
    float3 DiffuseReflectance;
    float3 SpecularRefelctance;
#else
    float3 Position;
#endif
};



END_NAMESPACE_FALCOR
