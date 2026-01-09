
#include <tiny-cuda-nn/common.h>
#include <cuda_fp16.h>

template <typename T>
__device__ T clamp2(T v, T vMin, T vMax) {
	if (v < vMin)
		return vMin;
	if (v > vMax)
		return vMax;
	return v;
}

template <typename T>
__global__ void cross_entropy_luminance(
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

	///unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned intra_elem_idx = i % stride;
	unsigned inter_elem_idx = i / stride;
	
	unsigned realI = i;

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

	const unsigned n_total = n_elements / stride * 3;

	float prediction = clamp2((float)predictions[i], 0.0f, 1.0f);

	float r = clamp2((float)predictions[i - intra_elem_idx + 0], 0.0f, 1.0f);
	float g = clamp2((float)predictions[i - intra_elem_idx + 1], 0.0f, 1.0f);
	float b = clamp2((float)predictions[i - intra_elem_idx + 2], 0.0f, 1.0f);

	const float Fac = 1.0;

	float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

	float target_value = targets[target_idx];
	const float pdf = targets[inter_elem_idx * dims + 9];
	const float inv_pdf = 1.0 / pdf;
	const float thp = targets[inter_elem_idx * dims + 3 + intra_elem_idx];
	const float add = targets[inter_elem_idx * dims + 6 + intra_elem_idx];


#if 1
	float custom_luminance = 0.299f * (r * targets[inter_elem_idx * dims + 3]+targets[inter_elem_idx * dims + 6 + 0]) +
				0.587f * (g * targets[inter_elem_idx * dims + 4]+targets[inter_elem_idx * dims + 6 + 1]) + 
				0.114f * (b * targets[inter_elem_idx * dims + 5]+targets[inter_elem_idx * dims + 6 + 2]);
	const float prediction_sq_plus_epsilon = 1.0/(custom_luminance*custom_luminance+ 0.01);
#else
	float custom_luminance = 0.299f * (r*targets[inter_elem_idx * dims + 3]) +
				0.587f * (g*targets[inter_elem_idx * dims + 4]) + 
				0.114f * (b*targets[inter_elem_idx * dims + 5]);
	const float prediction_sq_plus_epsilon = 1.0/(custom_luminance*custom_luminance+ 0.01);
#endif

	float add_custom_luminance = 0.299f * (abs(targets[inter_elem_idx * dims + 6])) +
				0.587f * (abs(targets[inter_elem_idx * dims + 7])) + 
				0.114f * (abs(targets[inter_elem_idx * dims + 8]));
	//prediction *= thp;


	

	float value = 0.0f;
	float gradient = 0.0f;

	#if 1
		target_value  = (target_value > 0.0) ? 1.0 : 0.0;
		float eps = 0.000001;

		if(target_value == 1.0f){
			const float factor = -target_value;

			value = factor * logf(prediction+ eps); // Epsilon to prevent NaNs
			gradient = factor / prediction;
		}
		else{
			const float factor = 1.0;
			value = factor * logf((1.0-prediction + eps)); // Epsilon to prevent NaNs
			gradient = factor / (1.0-prediction + eps);
		}
	#else
	
		const float diff = (prediction*thp - target_value);
		const float integral_val = diff;
		const float integral_diff = integral_val * integral_val;
		value = integral_diff;
		gradient = 2*diff;		
	#endif

	float lum_weight = 1.0f;
	#if 1
	if(pdf != 0.0f)
	 	lum_weight = prediction_sq_plus_epsilon;
	#endif

	float weight = 1.0f;
	float mis_weight = 1.0f;
	// if(pdf != 0.0f){
	// 	mis_weight = pdf;
	// }

#if 1
	float we = mis_weight*lum_weight*thp/n_total;
#else
	float we = 1.0f/n_total;
#endif


	value *= we;
	gradient *= we;
	gradient *= loss_scale;
	

	
	float limit = 50000;
	if(gradient > limit)
		gradient = limit;

	if(gradient < -limit)
		gradient = -limit;

	/*if (isnan(gradient) || abs(gradient) > 1000.0f) {
		gradient = fminf(fmaxf(gradient, -1000.0f), 1000.0f);
	}*/

	// if(intra_elem_idx != 0){
	// 	gradient = 0;
	// }

	// if (isnan(gradient))
	// 	gradient = 0;

	values[i] = value;
	gradients[i] = (T)(gradient);

	if (sao_gradients != nullptr) {
		sao_gradients[i / stride + intra_elem_idx * batch_size] = gradient;
	}
}
