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
#include "tiny-cuda-nn/jitify_wrapper.h"


#ifdef _DEBUG
#include "relative_l2_luminance_kernel.h"
#endif



//#pragma comment(lib, "Dbghelp")
//#define JITIFY_PRINT_ALL 1
//#include "jitify.hpp"
//
//#pragma comment(lib, "cudart")
//#pragma comment(lib, "nvrtc")
//
//// These live in _BootstrapUtils.cpp since they use Falcor includes / namespace,
////    which does not appear to play nice with the CUDA includes / namespace.
//extern void logFatal(std::string str);
//extern void logError(std::string str);
//extern void logOptixWarning(unsigned int level, const char* tag, const char* message, void*);

TCNN_NAMESPACE_BEGIN


template <typename T>
class RelativeL2LuminanceLoss : public Loss<T> {
public:
	RelativeL2LuminanceLoss(const json& lossData){
		m_lod = 0;
		m_ema = 1.0f;
		m_luminance_divider = 0.01f;
		m_luminance_weighting = false;
		m_custom_weighting = false;
		m_lum_weight = 0.5f;
		m_var_weight = 0.5f;
		m_tonemapped = false;
		update_hyperparams(lossData);
	}

	RelativeL2LuminanceLoss(){
		m_lod = 0;
		m_luminance_divider = 0.01f;
		m_ema = 1.0f;
		m_luminance_weighting = false;
		m_custom_weighting = false;
		m_tonemapped = false;
		m_lum_weight = 0.5f;
		m_var_weight = 0.5f;
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

		// linear_kernel(relative_l2_luminance_loss<T>, 0, stream,
		// 	prediction.n_elements(),
		// 	stride,
		// 	dims,
		// 	loss_scale,
		// 	prediction.data(),
		// 	target.data(),
		// 	values.data(),
		// 	gradients.data(),
		// 	data_pdf ? data_pdf->data() : nullptr,
		// 	sao_gradients ? sao_gradients->data() : nullptr
		// );
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

		//if (target.m() != dims+ m_custom_weighting) {
		//	throw std::runtime_error(std::string("Target does not have appropriate dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(dims+ m_custom_weighting));
		//}

		if (true) {
			CUDA_CHECK_THROW(cudaMemsetAsync(gradients.data(), 0, sizeof(T)* prediction.n_elements(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(values.data(), 0, sizeof(float) * values.n_elements(), stream));
		}

#if _DEBUG
		linear_kernel(relative_l2_luminance_loss_lod<T>, 0, stream,
#else
		JitCacheManager::getInstance().runKernelLinearTemplate(LOCAL_DIR, "relative_l2_luminance_kernel.h", "relative_l2_luminance_loss_lod", 0, stream, KernelTemplate<T>(),
#endif
			prediction.n_elements()/stride*3,
			batch_size,
			batch_size * prediction.m(),
			mapping,
			stride,
			dims + 10,
			loss_scale,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data(),
			data_pdf ? reinterpret_cast<float*>(data_pdf->data_mutable()) : nullptr,
			m_lod,
			m_ema,
			m_luminance_weighting,
			m_custom_weighting,
			m_luminance_divider,
			sao_gradients ? sao_gradients->data() : nullptr,
			m_lum_weight,
			m_var_weight,
			m_tonemapped
		);

		if (m_validate_gradients) {
			if (!m_validation_buffer)
			{
				void** t = (void**)(&m_validation_buffer);
				CUDA_CHECK_THROW(cudaMalloc(t, sizeof(int)));
			}
			CUDA_CHECK_THROW(cudaMemsetAsync(m_validation_buffer, 0, sizeof(int), stream));

#if _DEBUG
			linear_kernel(validate_gradients<T>, 0, stream,
#else
			JitCacheManager::getInstance().runKernelLinearTemplate(LOCAL_DIR, "relative_l2_luminance_kernel.h", "validate_gradients", 0, stream, KernelTemplate<T>(),
#endif
				prediction.n_elements(),
				gradients.data(),
				m_validation_buffer
			);

			int sum;
			CUDA_CHECK_THROW(cudaMemcpyAsync(&sum, m_validation_buffer, sizeof(int), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			std::cout << "validation: " << sum << std::endl;
		}
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

		if (params.contains("custom_weighting")) {
			m_custom_weighting = params["custom_weighting"];
		}

		if (params.contains("luminance_divider")) {
			m_luminance_divider = params["luminance_divider"];
		}


		if (params.contains("var_weight")) {
			m_var_weight = params["var_weight"];
		}

		if (params.contains("lum_weight")) {
			m_lum_weight = params["lum_weight"];
		}

		if (params.contains("tonemapped")) {
			m_tonemapped = params["tonemapped"];
		}
	}

protected:
	uint32_t m_lod;
	float m_ema;
	bool m_luminance_weighting;
	bool  m_custom_weighting;
	float m_luminance_divider;
	float m_var_weight;
	float m_lum_weight;
	bool m_tonemapped;

	bool m_validate_gradients = false;
	bool m_clear_gradients = false;
	int* m_validation_buffer = nullptr;
};

TCNN_NAMESPACE_END
