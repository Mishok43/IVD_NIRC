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

/** @file   adam.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Implementation of the adam optimizer with support for
 *          the AdaBound paper: https://openreview.net/pdf?id=Bkg3g2R9FX
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <cuda_fp16.h>


__device__ inline float weight_decay(float relative_weight_decay, float absolute_weight_decay, float weight) {
	// Relative weight decay is closely related to l2 regularization, whereas absolute weight decay corresponds to l1 regularization
	return (1 - relative_weight_decay) * weight - copysignf(absolute_weight_decay, weight);
}

template <typename T>
__global__ void adam_step(
	const unsigned int n_elements,
	const unsigned int n_matrix_weights,
	const float relative_weight_decay,
	const float absolute_weight_decay,
	const float loss_scale,
	float learning_rate,
	const float non_matrix_learning_rate_factor,
	const bool optimize_matrix_params,
	const bool optimize_non_matrix_params,
	 float beta1,
	 float beta2,
	const float epsilon,
	const float lower_lr_bound,
	const float upper_lr_bound,
	const float l2_reg,
	float* __restrict__ weights_full_precision,
	T* __restrict__ weights,
	const T* __restrict__ gradients,
	float* __restrict__ first_moments,
	float* __restrict__ second_moments,
	unsigned int* __restrict__ param_steps
) {
	
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float gradient = (float)gradients[i] / loss_scale;

	// static const float clip_value = 1000.0f;
	// if (isnan(gradient) || abs(gradient) > 1000.0f) {
	// 	gradient = fminf(fmaxf(gradient, -clip_value), clip_value);
	// }


	if (i >= n_matrix_weights) {
		if (!optimize_non_matrix_params || gradient == 0) {
			return;
		}
	}
	else {
		if (!optimize_matrix_params) {
			return;
		}
	}

	const float weight_fp = weights_full_precision[i];

	if (i < n_matrix_weights) {
		// No L2 reg for non-matrix params
		gradient += l2_reg * weight_fp;
	}

	const float gradient_sq = gradient * gradient;

	float first_moment = first_moments[i] = beta1 * first_moments[i] + (1 - beta1) * gradient;
	const float second_moment = second_moments[i] = beta2 * second_moments[i] + (1 - beta2) * gradient_sq;

	if (i >= n_matrix_weights) {
		// Potentially different learning rate for non-matrix params
		learning_rate *= non_matrix_learning_rate_factor;
	}


	// Debiasing. Since some parameters might see fewer steps than others, they each need their own step counter.
	const unsigned int current_step = ++param_steps[i];

#if 1
	learning_rate *= sqrtf(1 - powf(beta2, (float)current_step)) / (1 - powf(beta1, (float)current_step));
#endif

	// Follow AdaBound paradigm
	const float effective_learning_rate = fminf(fmaxf(learning_rate / (sqrtf(second_moment) + epsilon), lower_lr_bound), upper_lr_bound);

	const float decayed_weight = weight_decay(relative_weight_decay * learning_rate, absolute_weight_decay * learning_rate, weight_fp);
	const float new_weight = decayed_weight - effective_learning_rate * first_moment;

	weights_full_precision[i] = new_weight;
	weights[i] = (T)new_weight;
}
