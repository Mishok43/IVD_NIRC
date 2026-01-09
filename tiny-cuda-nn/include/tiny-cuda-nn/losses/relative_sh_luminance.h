/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

 /** @file   relative_l2_luminance.h
  *  @author Thomas Müller, NVIDIA
  *  @brief  Hacky implementation of the relative l2 loss based on the LUMINANCE of a six-channel prediction
  */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>


TCNN_NAMESPACE_BEGIN



template <typename T>
__global__ void relative_sh_luminance_loss(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t dims,
	const float loss_scale,
	const T* __restrict__ predictions,
	const float* __restrict__ targets,
	float* __restrict__ values,
	T* __restrict__ gradients,
	const Const4FloatArray bands_weights,
	const float* __restrict__ data_pdf = nullptr,
	float* __restrict__ sao_gradients = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	
	//const int gradient_index = (!m_soa) ? i : inter_elem_idx+intra_elem_idx*n_elements;

	const uint32_t target_dim = 6;
	const uint32_t target_idx_base = i* target_dim;	
	

	float target_rgb[3];
	//const float fac = (4*3.14);
	const float fac = 1.0;
	target_rgb[0] = targets[target_idx_base]*fac;
	target_rgb[1] = targets[target_idx_base+1]*fac;
	target_rgb[2] = targets[target_idx_base+2]*fac;

	
	float x = targets[target_idx_base+3];
	float y = targets[target_idx_base+4];
	float z = targets[target_idx_base+5];

	float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z, xyz=xy*z;

	float* __restrict__ values_cur = values+i* stride;
	T* __restrict__ gradients_cur = gradients+i* stride;

	const uint32_t ireal = i*stride;

	const uint32_t n_total = n_elements * dims;
	
	// A00 rgb coefficians = average radiance in the given position
	const float sh0_inv = 0.28209479177387814f;
	float r = clamp((float)predictions[ireal + 0], 0.0f, 10000.0f)*sh0_inv;
	float g = clamp((float)predictions[ireal + 1], 0.0f, 10000.0f)*sh0_inv;
	float b = clamp((float)predictions[ireal + 2], 0.0f, 10000.0f)*sh0_inv;

	const float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

	const float prediction_sq_plus_epsilon = luminance + 0.01f;

	const float norm_factor = 1.0/(prediction_sq_plus_epsilon*n_total);
	const float grad_norm_factor = loss_scale*norm_factor;
	
	auto res_compute = [&](float sh_projection, float p_r, int id, float grad_weight) {
		float l = (sh_projection*target_rgb[id%3] - p_r);
		float we = grad_weight*grad_norm_factor;
#if 0	
		l *= norm_factor;
		values_cur[id] = abs(l);
		if(l == 0.0f)
			gradients_cur[id] = (T)0.0f;
		else
			gradients_cur[id] = (T)((l > 0.0f) ? -we : we);
#else
		values_cur[id] = l*l*(norm_factor*grad_weight);
		gradients_cur[id] = (T)(-2.0f*l*we);
#endif
	};

	
	// 0-band
	#pragma unroll
	for(int k=0; k<3; k++)
		res_compute(0.28209479177387814f, predictions[ireal+k],0+k, bands_weights.a0);

	if(dims > 3){
		// 1-band
		#pragma unroll
		for(int k=0; k<3; k++){
			res_compute(-0.48860251190291987f*y, predictions[ireal+3+k], 3+k, bands_weights.a1);
			res_compute(0.48860251190291987f*z, predictions[ireal+6+k], 6+k, bands_weights.a1);
			res_compute(-0.48860251190291987f*x, predictions[ireal+9+k], 9+k, bands_weights.a1);
		}
	}

	if(dims > 12){
		// 2-band
		#pragma unroll
		for(int k=0; k<3; k++){
			res_compute(1.0925484305920792f*xy, predictions[ireal+12+k], 12+k, bands_weights.a2);
			res_compute(-1.0925484305920792f*yz, predictions[ireal+15+k], 15+k, bands_weights.a2);
			res_compute(0.94617469575755997f*z2 - 0.31539156525251999f, predictions[ireal+18+k], 18+k, bands_weights.a2);
			res_compute(-1.0925484305920792f*xz, predictions[ireal+21+k], 21+k, bands_weights.a2);
			res_compute(0.54627421529603959f*x2 - 0.54627421529603959f*y2, predictions[ireal+24+k], 24+k, bands_weights.a2);
		}
	}

	#pragma unroll
	for (int k = 0; k < stride-dims; k++) {
		values_cur[k+dims] = 0.0f;
		gradients_cur[k+dims] = (T)0.0f;
	}
}


// template <typename T>
// __global__ void relative_sh_luminance_loss_lod(
// 	const uint32_t n_elements_padded,
// 	const uint32_t batch_size,
// 	const uint32_t n_elements,
// 	const int* __restrict__ mapping,
// 	const uint32_t stride,
// 	const uint32_t dims,
// 	const float loss_scale,
// 	const T* __restrict__ predictions,
// 	 float* __restrict__ targets,
// 	float* __restrict__ values,
// 	T* __restrict__ gradients,
// 	 float* __restrict__ data_pdf = nullptr,
// 	uint32_t LOD = 0,
// 	float ema = 1.0f,
// 	bool luminance_weighting = true,
// 	float* __restrict__ sao_gradients = nullptr
// ) {

// 	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
// 	uint32_t realI = i;
// 	if (i >= n_elements_padded) return;
// 	if (i >= n_elements) {
// 		values[i] = 0;
// 		gradients[i] = 0;
// 		return;
// 	}

	
// 	const uint32_t intra_elem_idx = i % stride;
// 	const uint32_t inter_elem_idx = i / stride;
// 	if (intra_elem_idx >= dims) {
// 		values[i] = 0;
// 		gradients[i] = 0;
// 		return;
// 	}

// #if 1
// 	// Maybe it will work better:
// 	uint32_t target_idx = inter_elem_idx;
// if(mapping != nullptr)
// 	target_idx = mapping[target_idx]*dims+intra_elem_idx;
// else
// 	target_idx = inter_elem_idx * dims + intra_elem_idx;

// #else
// 	const uint32_t target_idx = inter_elem_idx * dims + intra_elem_idx;
// #endif

// 	const uint32_t n_total = n_elements / stride * dims;

// 	float prediction = (float)predictions[i];

// 	float r = (float)predictions[i - intra_elem_idx + 0];
// 	float g = (float)predictions[i - intra_elem_idx + 1];
// 	float b = (float)predictions[i - intra_elem_idx + 2];


// 	const float Fac = 1.0;
// #define SQRT 0

// 	if(LOD == 0){
// 		r = clamp(r, 0.0f, 10000.0f);
// 		g = clamp(g, 0.0f, 10000.0f);
// 		b = clamp(b, 0.0f, 10000.0f);
// 	}

// 	float luminance = 0.299f * r + 0.587f * g + 0.114f * b;

// #if !SQRT
// 	luminance = luminance;
// #endif
// 	//const float prediction_sq_plus_epsilon =sqrt(sqrt(sqrt(sqrt(luminance+0.000001f)+0.0000001f)+ 0.0000001f ) + 0.0000001f )+ 0.001f;

	
// 	if (LOD == 0) {
// 		data_pdf[target_idx] = luminance;
		
// 	}
// 	else {
// 		luminance += data_pdf[target_idx];
// 		luminance = clamp(luminance, 0.0f, 1000000.0f);
// 	}
	
// 	const float prediction_sq_plus_epsilon =luminance + 0.01f;

// 	const float pdf =  1;

// 	const float target_value = targets[target_idx];

// #define EMA 0

// #if EMA
// 	prediction = prediction*ema+(1.0-ema)*target_value;
// #endif

// #if SQRT
// 	const float difference = sqrt(clamp(prediction*Fac, 0.0f, 10000.0f) + 0.000001) - sqrt(target_value*Fac + 0.000001) / pdf;
// #else
// 	const float difference = prediction - target_value / pdf;
// #endif

// 	if(luminance_weighting)
// 		values[i] = difference * difference / prediction_sq_plus_epsilon / n_total;
// 	else
// 		values[i] = difference * difference;

// 	float gradient = loss_scale * (difference / prediction_sq_plus_epsilon)/n_total;
// 	/*if (isnan(gradient) || abs(gradient) > 1000.0f) {
// 		gradient = fminf(fmaxf(gradient, -1000.0f), 1000.0f);
// 	}*/


// 	gradients[i] = (T)(gradient);
// 	if(LOD == 0)
// 		targets[target_idx] = -difference;
	
// 	if(sao_gradients != nullptr){
// 		sao_gradients[inter_elem_idx+intra_elem_idx*batch_size] = gradient;
// 	}
// }


template <typename T>
class RelativeSHLuminanceLoss : public Loss<T> {
public:
	RelativeSHLuminanceLoss(const json& lossData){
		m_lod = 0;
		m_ema = 1.0f;
		m_luminance_weighting = true;
		for (uint32_t i = 0; i < 4; i++)
			grad_bands_weights[i] = 1.0;

		update_hyperparams(lossData);
	}

	RelativeSHLuminanceLoss(){
		m_lod = 0;
		m_ema = 1.0f;
		m_luminance_weighting = true;

		for (uint32_t i = 0; i < 4; i++)
			grad_bands_weights[i] = 1.0;
	}

	void evaluate(
		cudaStream_t stream,
		const uint32_t stride,
		const uint32_t dims,
		const float loss_scale,
		const GPUMatrix<T>& prediction,
		const GPUMatrix<float>& target,
		GPUMatrix<float>& values,
		GPUMatrix<T>& gradients,
		const GPUMatrix<float>* data_pdf = nullptr,
		GPUMatrix<float>* sao_gradients=nullptr) const override {
		if (prediction.n() != target.n()) {
			throw std::runtime_error(std::string("Prediction and target don't have matching batch size ") + std::to_string(prediction.n()) + "!=" + std::to_string(target.n()));
		}

		if (prediction.m() != stride) {
			throw std::runtime_error(std::string("Prediction does not have appropriate dimensions ") + std::to_string(prediction.m()) + "!=" + std::to_string(stride));
		}

		if (target.m() != dims) {
			throw std::runtime_error(std::string("Target does not have appropriate dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(dims));
		}

		linear_kernel(relative_sh_luminance_loss<T>, 0, stream,
			prediction.n(),
			stride,
			dims,
			loss_scale,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data(),
			Const4FloatArray(grad_bands_weights),
			data_pdf ? data_pdf->data() : nullptr,
			sao_gradients ? sao_gradients->data() : nullptr
		);
	}
	
	void evaluate(
		cudaStream_t stream,
		const uint32_t stride,
		const uint32_t dims,
		uint32_t batch_size,
		int* mapping,
		const float loss_scale,
		const GPUMatrix<T>& prediction,
		GPUMatrix<float>& target,
		GPUMatrix<float>& values,
		GPUMatrix<T>& gradients,
		const GPUMatrix<float>* data_pdf = nullptr,
		GPUMatrix<float>* sao_gradients=nullptr) const override {
		/*if (prediction.n() != batch_size) {
			throw std::runtime_error(std::string("Prediction and mapping don't have matching batch size ") + std::to_string(prediction.n()) + "!=" + std::to_string(target.n()));
		}*/

		if (prediction.m() != stride) {
			throw std::runtime_error(std::string("Prediction does not have appropriate dimensions ") + std::to_string(prediction.m()) + "!=" + std::to_string(stride));
		}


		linear_kernel(relative_sh_luminance_loss<T>, 0, stream,
			prediction.n(),
			stride,
			dims,
			loss_scale,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data(),
			Const4FloatArray(grad_bands_weights),
			data_pdf ? data_pdf->data() : nullptr,
			sao_gradients ? sao_gradients->data() : nullptr
		);

		//if (target.m() != dims) {
		//	throw std::runtime_error(std::string("Target does not have appropriate dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(dims));
		//}

		// linear_kernel(relative_sh_luminance_loss_lod<T>, 0, stream,
		// 	prediction.n_elements(),
		// 	batch_size,
		// 	batch_size* prediction.m(),
		// 	mapping,
		// 	stride,
		// 	dims,
		// 	loss_scale,
		// 	prediction.data(),
		// 	target.data(),
		// 	values.data(),
		// 	gradients.data(),
		// 	data_pdf ? reinterpret_cast<float*>(data_pdf->data_mutable()) : nullptr,
		// 	m_lod,
		// 	m_ema,
		// 	m_luminance_weighting,
		// 	sao_gradients ? sao_gradients->data() : nullptr
		// );
	}

	void update_hyperparams(const json& params) override { 
		if (params.contains("lod")) {
			m_lod = params["lod"];
		}

		if (params.contains("ema")) {
			m_ema = params["ema"];
		}

		if (params.contains("luminance_weighting")) {
			m_luminance_weighting = params["luminance_weighting"];
		}

		if (params.contains("grad_bands_weights")) {
			for(uint32_t i=0; i < 4; i++)
		 		grad_bands_weights[i] = params["grad_bands_weights"][i];
		}
		else{
			for (uint32_t i = 0; i < 4; i++)
				grad_bands_weights[i] = 1.0;
		}
	}

	void update_band_weight(uint32_t i_band, float weight) {
		grad_bands_weights[i_band] = weight;
	}

	float get_band_weight(uint32_t i_band) const {
		return grad_bands_weights[i_band];
	}
protected:
	uint32_t m_lod;
	float m_ema;
	bool m_luminance_weighting;

	float grad_bands_weights[4];
};

TCNN_NAMESPACE_END
