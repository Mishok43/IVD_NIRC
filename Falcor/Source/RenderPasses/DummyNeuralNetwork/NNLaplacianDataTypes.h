#pragma once


struct cFloat3 {
    __device__ cFloat3(float _x, float _y, float _z) {
        x = _x;
        y = _y;
        z = _z;
    }

    __device__ cFloat3(float a) {
        x = a;
        y = a;
        z = a;
    }


    __device__ cFloat3() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }

    __device__ void sqrtv(){
        x = sqrt(x+0.000001);
        y = sqrt(y+0.000001);
        z = sqrt(z+0.000001);
    }

    __device__ void absv() {
        x = abs(x);
        y = abs(y);
        z = abs(z);
    }

    __device__ bool isanynan() {
        return isnan(x) || isnan(y) || isnan(z);
    }

    __device__  void clamp(float minV, float maxV) {
        if (x < minV) {
            x = minV;
        }
        else if (x > maxV) {
            x = maxV;
        }

        if (y < minV) {
            y = minV;
        }
        else if (y > maxV) {
            y = maxV;
        }

        if (z < minV) {
            z = minV;
        }
        else if (z > maxV) {
            z = maxV;
        }
    };

    __device__ void mult(float s) {
        x *= s;
        y *= s;
        z *= s;
    }

    __device__ void mult(cFloat3 c)
    {
        x *= c.x;
        y *= c.y;
        z *= c.z;
    }

    __device__ void subst(cFloat3 c) {
        x -= c.x;
        y -= c.y;
        z -= c.z;
    }

    __device__ void add(cFloat3 c) {
        x += c.x;
        y += c.y;
        z += c.z;
    }

    __device__ float sqr_length() {
        return x * x + y * y + z * z;
    }

    __device__ void div(cFloat3 c) {
        x /= c.x;
        y /= c.y;
        z /= c.z;
    }

    float x;
    float y;
    float z;
};



struct cResErrorData {
    int id;
    cFloat3 estimation;
};



struct cTrainingAdditionalData {
    unsigned int lightPathID;
    cFloat3 att;
    cFloat3 thp;
    cFloat3 thpDiv;
    cFloat3 sub;
    float pdf;
};

struct cLThp {
    cFloat3 L;
    cFloat3 thp;
};

struct cYData {
    cFloat3 radiance;
    cFloat3 weight_thp;
    cFloat3 weight_add;
    float pdf;
    cFloat3 weight_thp_div;
};

