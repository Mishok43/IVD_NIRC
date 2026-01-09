
#include <tiny-cuda-nn/common.h>
#include <cuda_fp16.h>

template <typename T>
__device__ T clamp(T v, T vMin, T vMax) {
	if (v < vMin)
		return vMin;
	if (v > vMax)
		return vMax;
	return v;
}

template <typename T>
__global__ void relative_l2_luminance_loss(
	const unsigned n_elements,
	const unsigned stride,
	const unsigned dims,
	const float loss_scale,
	const T* __restrict__ predictions,
	const float* __restrict__ targets,
	float* __restrict__ values,
	T* __restrict__ gradients,
	const float* __restrict__ data_pdf = nullptr,
	float* __restrict__ sao_gradients = nullptr
) {
	unsigned i_ = threadIdx.x + blockIdx.x * blockDim.x;
	if (i_ >= n_elements_padded) return;

	unsigned i = i_/3*stride+i_%3;

	const unsigned intra_elem_idx = i % stride;
	const unsigned inter_elem_idx = i / stride;


	//const int gradient_index = (!m_soa) ? i : inter_elem_idx+intra_elem_idx*n_elements;


	const unsigned target_idx = inter_elem_idx * dims + intra_elem_idx;

	const unsigned n_total = n_elements / stride * dims;

	const float prediction = (float)predictions[i];

	float r = clamp((float)predictions[i - intra_elem_idx + 0], 0.0f, 10000.0f);
	float g = clamp((float)predictions[i - intra_elem_idx + 1], 0.0f, 10000.0f);
	float b = clamp((float)predictions[i - intra_elem_idx + 2], 0.0f, 10000.0f);



	const float Fac = 1.0;
#define SQRT 0

	float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
#if !SQRT
	luminance = luminance;
#endif
	//const float prediction_sq_plus_epsilon =sqrt(sqrt(sqrt(sqrt(luminance+0.000001f)+0.0000001f)+ 0.0000001f ) + 0.0000001f )+ 0.001f;
	float prediction_sq_plus_epsilon = luminance + 0.01f;
	
	//const float prediction_sq_plus_epsilon = 1.0f;

	const float pdf = data_pdf ? data_pdf[target_idx] : 1;

#if SQRT
	const float difference = sqrt(clamp(prediction * Fac, 0.0f, 10000.0f) + 0.000001) - sqrt(targets[target_idx] * Fac + 0.000001) / pdf;
#else
	const float difference = prediction - targets[target_idx] / pdf;
#endif

	values[i] = difference * difference / prediction_sq_plus_epsilon / n_total;

	float gradient = 2 * difference / prediction_sq_plus_epsilon;
	gradient = loss_scale * gradient / n_total;

	gradients[i] = (T)gradient;

	if (sao_gradients != nullptr) {
		sao_gradients[inter_elem_idx + intra_elem_idx * n_elements] = gradient;
	}
}


struct ACESSettings {
	float a, b, c, d, e;
};

__device__ float aces_tonemap(float x, ACESSettings s) {
	x *= 0.6;
	float r = (x * x * s.a + s.b * x) / (x * x * s.c + x * s.d + s.e);
	if (r < 0.0) {
		return sqrt(0.0001);
	}
	if (r > 1.0) {
		return 1.0;
	}
	return sqrt(r + 0.0001);
}

__device__ float aces_tonemap_der(float x, ACESSettings s, float val){
	x *= 0.6;
	float t = x*(s.c*x+s.d)+s.e;
	return 	(0.5/sqrt(val))*0.6*(s.a*x*(s.d*x+2*s.e)+s.b*(s.e-s.c*x*x))/(t*t);
}



template <typename T>
__global__ void relative_l2_luminance_loss_lod(
	const unsigned n_elements_padded,
	const unsigned batch_size,
	const unsigned n_elements,
	const int* __restrict__ mapping,
	const unsigned stride,
	const unsigned dims,
	const float loss_scale,
	const T* __restrict__ predictions,
	float* __restrict__ targets,
	float* __restrict__ values,
	T* __restrict__ gradients,
	float* __restrict__ data_pdf = nullptr,
	unsigned LOD = 0,
	float ema = 1.0f,
	bool luminance_weighting = true,
	bool m_custom_weighting = false,
	float luminance_divider = 0.01f,
	float* __restrict__ sao_gradients = nullptr,
	const float integral_coef = 0.5f,
	const float variance_coef = 0.5f,
	const float m_tonemapped = false
) {
	unsigned i_ = threadIdx.x + blockIdx.x * blockDim.x;
	if (i_ >= n_elements_padded) return;
	unsigned i = i_/3*stride+i_%3;
	unsigned realI = i;


	const unsigned intra_elem_idx = i % stride;
	unsigned inter_elem_idx = i / stride;

#if 1
	// Maybe it will work better:
	unsigned target_idx = inter_elem_idx;
	if (mapping != nullptr){
		target_idx = mapping[target_idx] * dims + intra_elem_idx;
		inter_elem_idx = mapping[inter_elem_idx];
	}
	else
		target_idx = inter_elem_idx * dims + intra_elem_idx;

#else
	const unsigned target_idx = inter_elem_idx * dims + intra_elem_idx;
#endif

	const unsigned n_total = n_elements / stride * dims;

#if 1
	float prediction = clamp((float)predictions[i], 0.0f, 10000.0f);

	float r = clamp((float)predictions[i - intra_elem_idx + 0], 0.0f, 10000.0f);
	float g = clamp((float)predictions[i - intra_elem_idx + 1], 0.0f, 10000.0f);
	float b = clamp((float)predictions[i - intra_elem_idx + 2], 0.0f, 10000.0f);
#else

	float prediction = (float)predictions[i];

	float r = abs((float)predictions[i - intra_elem_idx + 0]);
	float g = abs((float)predictions[i - intra_elem_idx + 1]);
	float b = abs((float)predictions[i - intra_elem_idx + 2]);
#endif



	float target_value = targets[target_idx];
	const float pdf = targets[inter_elem_idx * dims + 9];
	const float inv_pdf = 1.0 / pdf;
	 float thp = targets[inter_elem_idx * dims + 3 + intra_elem_idx];
	const float add = targets[inter_elem_idx * dims + 6 + intra_elem_idx];

	float custom_luminance = 0.299f * (r * targets[inter_elem_idx * dims + 3]+targets[inter_elem_idx * dims + 6 + 0]) +
				0.587f * (g * targets[inter_elem_idx * dims + 4]+targets[inter_elem_idx * dims + 6 + 1]) + 
				0.114f * (b * targets[inter_elem_idx * dims + 5]+targets[inter_elem_idx * dims + 6 + 2]);

	float lum_weight = 1.0f;

	#if 1
	lum_weight = 1.0 /(custom_luminance*custom_luminance+0.001);
	prediction *= thp;
	prediction += add;
	#endif

	#if 0
	lum_weight = 1.0 /(custom_luminance*custom_luminance+0.001);
	target_value -= add;
	target_value /= thp;
	thp = 1.0;

	#endif
	
	const float diff =  prediction - target_value ;

	const float integral_val = diff;
	const float integral_diff = integral_val * integral_val;

	const float loss_val = integral_diff*lum_weight;
	float grad_val = 2*(diff*thp)*lum_weight;

	float weight = 1.0f;
	float limit = 300000;
	if(grad_val > limit)
		grad_val = limit;
	if(grad_val < -limit)
		grad_val = -limit;

	// if (luminance_weighting) {
	// 	weight = 1.0 / prediction_sq_plus_epsilon;
	// }
	// else
	// 	if (m_custom_weighting) {
	// 		luminance = 0.299f * (r * targets[inter_elem_idx * dims + 3] + targets[inter_elem_idx * dims + 6]) +
	// 			0.587f * (g * targets[inter_elem_idx * dims + 4] + targets[inter_elem_idx * dims + 7]) +
	// 			0.114f * (b * targets[inter_elem_idx * dims + 5] + targets[inter_elem_idx * dims + 8]);
	// 		weight = 1.0 / (luminance + luminance_divider);
	// 	}


	values[i] = loss_val * weight / n_total;

	float gradient = grad_val * loss_scale/ n_total;
	/*if (isnan(gradient) || abs(gradient) > 1000.0f) {
		gradient = fminf(fmaxf(gradient, -1000.0f), 1000.0f);
	}*/

	// if(intra_elem_idx != 0){
	// 	gradient = 0;
	// }

	gradients[i] = (T)(gradient);

	if (sao_gradients != nullptr) {
		sao_gradients[i / stride + intra_elem_idx * batch_size] = gradient;
	}
}
