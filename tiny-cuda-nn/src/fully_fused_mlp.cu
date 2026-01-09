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

/** @file   fully_fused_mlp.cu
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Fully fused CUDA implementation of a multi-layer perceptron. Supports online training
 *          and simultaneous inference.
 */

#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <tiny-cuda-nn/cutlass_matmul.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/sh.h>
#include <mma.h>



#define FAST_COMPILATION 1

TCNN_NAMESPACE_BEGIN


struct SeparateInputeCuda5 {
	const __half* data[5];
	const uint32_t* mapping;
	bool is_mapped[5];
	int  elements_per_channel[5];
	int  offset[5];
	int  num_elements;
};

struct InstantOutput {
	const uint32_t* mapping;
	const float3* thp;
	float* output;
};

__constant__ SeparateInputeCuda5 c_input;
__constant__ InstantOutput c_output;

//#define FUSED_SHADING 0

void check_shmem_error(cudaError_t error) {
	if (error != cudaSuccess) {
		throw std::runtime_error{"FullyFusedMLP: insufficient shared memory available on the GPU. Reduce `n_neurons` or use `CutlassMLP` (better compatibility but slower) instead."};
	}
}


template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, bool BACKWARD=false, bool FUSED_SHADING = false, uint32_t MIN_SH_BAND = 0, uint32_t NUM_SH_BAND = 0>
__device__ void threadblock_layer(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer,
 OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const OUT_T* __restrict__ activation_aux = nullptr,
 OutputShadingStructure outShadeData = OutputShadingStructure(), Activation output_activation = Activation::None, uint32_t output_padded_size=0) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch.
	//           Can be forward activations or backward activations, depending on caller.
	// weights_this_layer points to the weight matrix of the current layer.
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// activation_aux points to additional arguments that the activation function may depend on. Points to the hidden forward activations when computing backward activations.


	bool bLastLayer = output_activation != Activation::None;

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// If we're performing the backward pass, weights must be loaded in transposed form, which
	// is achieved by interpreting the memory in row_major instead of col_major order.
	using weights_layout_t = std::conditional_t<BACKWARD, wmma::row_major, wmma::col_major>;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, weights_layout_t> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;

	__syncthreads();


	bool fused_output = !BACKWARD && FUSED_SHADING && output_activation == Activation::SH;
	bool execution = !fused_output || output_padded_size > weights_col;
	if (execution)
	{
		// Load N_BLOCKS chunks of weights from global memory into registers.
#pragma unroll
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			if (BACKWARD) {
				// If we're performing the backward pass, additional index swizzling is needed to
				// load the weights in transposed form.
				wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i * WIDTH + weights_col, WIDTH);
			}
			else {
				wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16 * i + weights_col * WIDTH, WIDTH);
			}
		}

#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			wmma::fill_fragment(result_frag[l], 0.0f);

#pragma unroll
			for (uint32_t i = 0; i < N_BLOCKS; ++i) {
				// Load a chunk of intermediate activations from shared memory and multiply with chunk of weights
				wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), WIDTH + SKEW);
				wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
			}

			// Activation
			if (BACKWARD) {
				if (activation != Activation::None && activation_aux != nullptr) {
					// Load the temporary forward matrix for the relu transfer
					wmma::load_matrix_sync(act_frag, activation_aux + weights_col + (threadIdx.z + l * BLOCK_DIM_Z) * 16 * WIDTH, WIDTH);
					warp_activation_backward<__half>(activation, result_frag[l], act_frag, result_frag[l]);
				}
			}
			else {
				if (!bLastLayer)
					warp_activation<__half>(activation, result_frag[l], result_frag[l]);
			}
		}
	}

	__syncthreads();

	if (execution) {
#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			wmma::store_matrix_sync(act_shmem + weights_col + (threadIdx.z + l * BLOCK_DIM_Z) * 16 * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);

		}
	}

	__syncthreads();
	if (fused_output)
	{

		// Every wi is responsible for 32 elements of batch. 

		const uint32_t elem_offset = WIDTH + SKEW;
		const uint32_t elem_idx = (16*N_ITERS* BLOCK_DIM_Z)*blockIdx.x+ wi*32+li;
		const uint32_t shmem_offset = (16*2 * (threadIdx.z + wi * BLOCK_DIM_Z)) * elem_offset;
		half* data_ptr = act_shmem + shmem_offset + li * elem_offset;
		if (elem_idx < outShadeData.final_output_nelements) {
			SHTransformData sd = readSHTransformData(outShadeData.output_trasnform_data, elem_idx);
			float res_r, res_g, res_b;

			compute_sh_cv_templated<MIN_SH_BAND, NUM_SH_BAND>(data_ptr, sd, res_r, res_g, res_b, outShadeData.bands_shade_weights);

			outShadeData.final_output[3 * elem_idx] = res_r;
			outShadeData.final_output[3 * elem_idx + 1] = res_g;
			outShadeData.final_output[3 * elem_idx + 2] = res_b;
		}
	}
	else {
		if (out_intermediate_threadblock_this_layer != nullptr) {
			__syncthreads();

			#pragma unroll
			for (int l = 0; l < N_ITERS; ++l) {
				*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
			}
		}
	}
}

__device__ float2 oct_wrap2(float2 v)
{
	float2 res;
	res.x = (1 - abs(v.y)) * (v.x >= 0.0f ? 1.f : -1.f);
	res.y = (1 - abs(v.x)) * (v.y >= 0.0f ? 1.f : -1.f);
	return res;
}

__device__ float3 normalize2(float3 v) {
	float invlen = 1.0/sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	v.x *= invlen;
	v.y *= invlen;
	v.z *= invlen;
	return v;
}

__device__ float3 oct_to_ndir_snorm2(float2 p)
{
	float3 n;
	n.x = p.x;
	n.y = p.y;
	n.z = 1.0 - abs(p.x) - abs(p.y);
	if (n.z < 0.0) {
		float2 tmp = oct_wrap2(make_float2(n.x, n.y));
		n.x = tmp.x;
		n.y = tmp.y;
	}

    return normalize2(n);
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, bool INFERENCE, bool INPUT_SEPARATE=false>
__device__ void threadblock_load_input_static(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, uint32_t m_sh_offset) {
	// act_shmem will be filled by the thread block's chunk of input_threadblock

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	if(!INPUT_SEPARATE){

		#pragma unroll
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)] = *(int4*)&input_threadblock[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH];
		}
	
		__syncthreads();
	}



	const uint32_t elem_offset = WIDTH + SKEW;
	const uint32_t elem_idx = (16*N_ITERS* BLOCK_DIM_Z)*blockIdx.x+ wi*32+li;
	const uint32_t shmem_offset = (16*2 * (threadIdx.z + wi * BLOCK_DIM_Z)) * elem_offset;
	half* data_ptr = act_shmem + shmem_offset + li * elem_offset;
	if (INFERENCE) {
		if(INPUT_SEPARATE){
			uint32_t g_offset = 0;
			if (elem_idx < c_input.num_elements) {
				uint32_t mapped_id = c_input.mapping[elem_idx];

				bool bSpecialSH = c_input.elements_per_channel[4] == 2;

				
				for(uint32_t j = 0; j < 4+ !bSpecialSH; j++) {
					uint32_t index = (c_input.is_mapped[j]) ? mapped_id : elem_idx;
					uint32_t offset = c_input.elements_per_channel[j];
					const __half* data_input = &(c_input.data[j][index * offset]);

					uint32_t stride = c_input.offset[j];
					for (uint32_t k = 0; k < offset; k++) {
						data_ptr[stride + k] = data_input[k];
					}
				}

					g_offset = c_input.offset[4];
					if (bSpecialSH) {
						uint32_t offset = c_input.elements_per_channel[4];


						const __half* data_input = &(c_input.data[4][elem_idx * offset]);

						uint32_t of = g_offset;
						float2 v2;
						v2.x = float(data_input[0]);
						v2.y = float(data_input[1]);
						float3 v = oct_to_ndir_snorm2(v2);

						float x = v.x;
						float y = v.y;
						float z = v.z;


						float xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z, xyz = xy * z;
						float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
						float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;

						data_ptr[0 + g_offset] = 0.28209479177387814f;                          // 1/(2*sqrt(pi))

						data_ptr[1 + g_offset] = -0.48860251190291987f * y;                               // -sqrt(3)*y/(2*sqrt(pi))
						data_ptr[2 + g_offset] = 0.48860251190291987f * z;                                // sqrt(3)*z/(2*sqrt(pi))
						data_ptr[3 + g_offset] = -0.48860251190291987f * x;                               // -sqrt(3)*x/(2*sqrt(pi))

						data_ptr[4 + g_offset] = 1.0925484305920792f * xy;                                // sqrt(15)*xy/(2*sqrt(pi))
						data_ptr[5 + g_offset] = -1.0925484305920792f * yz;                               // -sqrt(15)*yz/(2*sqrt(pi))
						data_ptr[6 + g_offset] = (0.94617469575755997f * z2 - 0.31539156525251999f);                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
						data_ptr[7 + g_offset] = -1.0925484305920792f * xz;                               // -sqrt(15)*xz/(2*sqrt(pi))
						data_ptr[8 + g_offset] = (0.54627421529603959f * x2 - 0.54627421529603959f * y2);                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))

						data_ptr[9 + g_offset] = 0.59004358992664352f * y * (-3.0f * x2 + y2);                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
						data_ptr[10 + g_offset] = 2.8906114426405538f * xy * z;                             // sqrt(105)*xy*z/(2*sqrt(pi))
						data_ptr[11 + g_offset] = 0.45704579946446572f * y * (1.0f - 5.0f * z2);                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
						data_ptr[12 + g_offset] = 0.3731763325901154f * z * (5.0f * z2 - 3.0f);                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
						data_ptr[13 + g_offset] = 0.45704579946446572f * x * (1.0f - 5.0f * z2);                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
						data_ptr[14 + g_offset] = 1.4453057213202769f * z * (x2 - y2);                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
						data_ptr[15 + g_offset] = 0.59004358992664352f * x * (-x2 + 3.0f * y2);                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))

						data_ptr[16 + g_offset] = 2.5033429417967046f * xy * (x2 - y2);                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
						data_ptr[17 + g_offset] = 1.7701307697799304f * yz * (-3.0f * x2 + y2);                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
						data_ptr[18 + g_offset] = 0.94617469575756008f * xy * (7.0f * z2 - 1.0f);                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
						data_ptr[19 + g_offset] = 0.66904654355728921f * yz * (3.0f - 7.0f * z2);                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
						data_ptr[20 + g_offset] = (-3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f);                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
						data_ptr[21 + g_offset] = (0.66904654355728921f * xz * (3.0f - 7.0f * z2));                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
						data_ptr[22 + g_offset] = 0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f);                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
						data_ptr[23 + g_offset] = 1.7701307697799304f * xz * (-x2 + 3.0f * y2);                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
						data_ptr[24 + g_offset] = (-3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4);                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))

						g_offset += 25;
					}

			}

			for(uint32_t j=g_offset; j < WIDTH; j++){
				data_ptr[j] = 1.0f;
			}
		}
		else{
			if (m_sh_offset != 128) {
				const uint32_t offset = m_sh_offset;

				for (uint32_t j = m_sh_offset + 25; j < WIDTH; j++) {
					data_ptr[j] = 1.0;
				}

				data_ptr += offset;
				float2 v2;
				v2.x = float(data_ptr[0]);
				v2.y = float(data_ptr[1]);
				float3 v = oct_to_ndir_snorm2(v2);

				float x = v.x;
				float y = v.y;
				float z = v.z;


				float xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z, xyz = xy * z;
				float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
				float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;

				data_ptr[0] = 0.28209479177387814f;                          // 1/(2*sqrt(pi))

				data_ptr[1] = -0.48860251190291987f * y;                               // -sqrt(3)*y/(2*sqrt(pi))
				data_ptr[2] = 0.48860251190291987f * z;                                // sqrt(3)*z/(2*sqrt(pi))
				data_ptr[3] = -0.48860251190291987f * x;                               // -sqrt(3)*x/(2*sqrt(pi))

				data_ptr[4] = 1.0925484305920792f * xy;                                // sqrt(15)*xy/(2*sqrt(pi))
				data_ptr[5] = -1.0925484305920792f * yz;                               // -sqrt(15)*yz/(2*sqrt(pi))
				data_ptr[6] = (0.94617469575755997f * z2 - 0.31539156525251999f);                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
				data_ptr[7] = -1.0925484305920792f * xz;                               // -sqrt(15)*xz/(2*sqrt(pi))
				data_ptr[8] = (0.54627421529603959f * x2 - 0.54627421529603959f * y2);                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))

				data_ptr[9] = 0.59004358992664352f * y * (-3.0f * x2 + y2);                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
				data_ptr[10] = 2.8906114426405538f * xy * z;                             // sqrt(105)*xy*z/(2*sqrt(pi))
				data_ptr[11] = 0.45704579946446572f * y * (1.0f - 5.0f * z2);                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
				data_ptr[12] = 0.3731763325901154f * z * (5.0f * z2 - 3.0f);                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
				data_ptr[13] = 0.45704579946446572f * x * (1.0f - 5.0f * z2);                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
				data_ptr[14] = 1.4453057213202769f * z * (x2 - y2);                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
				data_ptr[15] = 0.59004358992664352f * x * (-x2 + 3.0f * y2);                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))

				data_ptr[16] = 2.5033429417967046f * xy * (x2 - y2);                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
				data_ptr[17] = 1.7701307697799304f * yz * (-3.0f * x2 + y2);                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
				data_ptr[18] = 0.94617469575756008f * xy * (7.0f * z2 - 1.0f);                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
				data_ptr[19] = 0.66904654355728921f * yz * (3.0f - 7.0f * z2);                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
				data_ptr[20] = (-3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f);                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
				data_ptr[21] = (0.66904654355728921f * xz * (3.0f - 7.0f * z2));                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
				data_ptr[22] = 0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f);                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
				data_ptr[23] = 1.7701307697799304f * xz * (-x2 + 3.0f * y2);                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
				data_ptr[24] = (-3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4);                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))

				
			}
		}

		__syncthreads();
	}

}


template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, Activation ACTIVATION, typename OUTPUT_LAYOUT>
__global__ void kernel_mlp_fused_backward(const __half* __restrict__ dL_doutput, const __half* __restrict__ weights, __half* __restrict__ out_intermediate, const __half* __restrict__ forward, __half* __restrict__ dL_dinput, const __half* __restrict__ weights_first_layer, const uint32_t batch_size, const uint32_t out_width, const uint32_t n_hidden_matmuls, const uint32_t m_sh_offset) {
	// `dL_doutput` points to the input matrix of the backward pass, i.e. the loss gradients. Assumed to be 16 neurons wide.
	// `weights` points to the weight matrices (contiguous in memory).
	// `out_intermediate` points to the memory where backpropagated activation gradients should be written.
	// `forward` points to the memory where the intermediate activations of the forward pass are located. (needed for activation backprop)

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")
	const uint32_t bi = blockIdx.x;	 // block index

	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// A skew is applied to the matrix storage to avoid bank conflicts.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	// Multipying one 16-row chunk of intermediate activations with the weight matrix requires all warps of the block.
	// Thus, each block computes exactly one 16-row chunk of the next layer's intermediate activations.
	const uint32_t elem_idx_base = 16 * bi * N_ITERS * BLOCK_DIM_Z;
	const uint32_t elem_idx = elem_idx_base + 16 * threadIdx.z;

	const uint32_t layer_stride = WIDTH * WIDTH;
	const uint32_t output_stride = WIDTH * batch_size;

	// Backprop through last layer
	if (out_width <= 16) {
		using namespace nvcuda;

		// Fragments in registers
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, OUTPUT_LAYOUT> act_frag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> weights_frag;
		wmma::fragment<wmma::accumulator, 16, 16, 16, __half> result_frag[N_ITERS];

		// Load the relevant chunk of the last layer's weight matrix from global memory into registers
		const uint32_t weights_col = 16 * wi;

		wmma::load_matrix_sync(weights_frag, weights + layer_stride * n_hidden_matmuls + weights_col, WIDTH);

		#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			wmma::fill_fragment(result_frag[l], 0.0f);

			// Load a chunk of output gradients from shared memory and multiply with previously loaded weights
			if (std::is_same<OUTPUT_LAYOUT, wmma::row_major>::value) {
				wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * (threadIdx.z + l * BLOCK_DIM_Z)) * 16, 16);
			} else {
				wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16 * (threadIdx.z + l * BLOCK_DIM_Z)), batch_size);
			}

			// NOTE: activation transfer of the _output_ activation is expected to be done _prior_ to calling this kernel
			//       in a separate pass, because the tranfered activation gradient is also needed to compute the weight
			//       gradient of the last weight matrix (see backward()).
			wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);

			// Load the temporary forward matrix for the relu transfer
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> forward_frag;
			wmma::load_matrix_sync(forward_frag, forward + output_stride * n_hidden_matmuls + weights_col + (elem_idx + l * BLOCK_DIM_Z * 16) * WIDTH, WIDTH);

			warp_activation_backward<__half>(ACTIVATION, result_frag[l], forward_frag, result_frag[l]);
		}

		__syncthreads();

		#pragma unroll
		for (int l = 0; l < N_ITERS; ++l) {
			wmma::store_matrix_sync(act_shmem + weights_col + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
		}

		__syncthreads();

		#pragma unroll
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&out_intermediate[lane_offset + (row + elem_idx + i * BLOCK_DIM_Z * 16) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
		}
	} else {
		// If the output width is larger than 16, we will have used CUTLASS for backpropping through the last layer.
		// Load the resulting gradients.
		threadblock_load_input_static<WIDTH, BLOCK_DIM_Z, N_ITERS, false>(act_shmem, out_intermediate + elem_idx * WIDTH, m_sh_offset);
	}

	// Backprop through hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, true>(ACTIVATION, act_shmem, weights + layer_stride * (n_hidden_matmuls - k - 1), out_intermediate + output_stride * (k + 1) + elem_idx_base * WIDTH, forward + output_stride * (n_hidden_matmuls - k - 1) + elem_idx_base * WIDTH);
	}

	// Compute loss gradients w.r.t. input if desired.
	// THIS CODE ASSUMES THAT THE INPUT WIDTH IS THE SAME AS THE NETWORK WIDTH.
	// DON'T PASS A NON-NULL dL_dinput IF THIS REQUIREMENT IS NOT MET.
	if (dL_dinput != nullptr) {
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, true>(Activation::None, act_shmem, weights_first_layer, dL_dinput + elem_idx_base * WIDTH);
	}
}

template <int WIDTH, typename T, Activation ACTIVATION>
std::enable_if_t<!std::is_same<__half, T>::value> mlp_fused_backward(
	cudaStream_t stream,
	const GPUMatrix<T, RM>& weights_first_layer,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrix<T>& temporaries,
	const GPUMatrix<T>& forward,
	GPUMatrix<T>* dL_dinput,
	const uint32_t n_hidden_matmuls
) {
	throw std::runtime_error{"The fully fused backward pass only supports __half precision."};
}

template <int WIDTH, typename T, Activation ACTIVATION>
std::enable_if_t<std::is_same<__half, T>::value> mlp_fused_backward(
	cudaStream_t stream,
	const GPUMatrix<T, RM>& weights_first_layer,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrix<T>& temporaries,
	const GPUMatrix<T>& forward,
	GPUMatrix<T>* dL_dinput,
	const uint32_t n_hidden_matmuls,
	const uint32_t m_sh_offset
) {
	const uint32_t batch_size = dL_doutput.cols();
	const uint32_t out_width = dL_doutput.rows();
	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	if (forward.cols() != batch_size) {
		throw std::runtime_error{"Batch size of matrices dL_doutput and temporaries doesn't match."};
	}

	const int N_ITERS = WIDTH >= 256 ? 2 : 8;
	const uint32_t BLOCK_DIM_Z = 1;

	if (batch_size % (16 * N_ITERS * BLOCK_DIM_Z) != 0) {
		throw std::runtime_error{"Batch size must be a multiple of " + std::to_string(16 * N_ITERS * BLOCK_DIM_Z) + "."};
	}

	const dim3 threads = { 32u, N_BLOCKS, BLOCK_DIM_Z }; // 32 threads = 1 warp, 8 warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

	uint32_t n_elems_per_block = 16 * BLOCK_DIM_Z * N_ITERS;
	uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

	int shmem_size = sizeof(__half) * ((16 * BLOCK_DIM_Z * N_ITERS) * (WIDTH + SKEW)); // WIDTH rows of input and 16 * threads.z rows of weights
	const dim3 blocks = { n_blocks, 1u, 1u };

	// The kernels operate with transposed layouts compared with the MLP code
	if (dL_doutput.layout() == RM) {
		check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, ACTIVATION, nvcuda::wmma::col_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
		kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, ACTIVATION, nvcuda::wmma::col_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), batch_size, out_width, n_hidden_matmuls, m_sh_offset);
	} else {
		check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, ACTIVATION, nvcuda::wmma::row_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
		kernel_mlp_fused_backward<WIDTH, BLOCK_DIM_Z, N_ITERS, ACTIVATION, nvcuda::wmma::row_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), batch_size, out_width, n_hidden_matmuls, m_sh_offset);
	}
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T>
__device__ void threadblock_input_layer_forward_dynamic(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const uint32_t in_width) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// input_threadblock points to the thread block's chunk of the input batch in global memory
	// weights_this_layer points to the weight matrix of the current layer
	// out_intermediate_threadblock_this_layer points to the location where intermediate activations produced by the thread block should be written to.
	//                  Can be nullptr if nothing should be written.
	// in_width is the dynamic width of the input layer

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t INPUT_SKEW = 8;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	const uint32_t weights_col = 16 * wi;

	__half* __restrict__ weights_shmem = act_shmem + BLOCK_DIM_Z * 16 * (in_width + INPUT_SKEW);

	// Load input weight matrix (fits completely into shared memory)
	// Each thread can load 8 fp16 elements (16 bytes) at once; we have N_BLOCKS*BLOCK_DIM_Z warps
	const uint32_t n_elems_per_load = N_BLOCKS * 32 * BLOCK_DIM_Z * 8;
	const uint32_t thread_elem_idx = (li + wi * 32 + threadIdx.z * N_BLOCKS * 32) * 8;

	const uint32_t n_elems_b = WIDTH * in_width;

	#pragma unroll
	for (uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
		const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
		*(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights_this_layer[idx];
	}

	const uint32_t n_tensor_ops = in_width / 16;

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		// Load chunk of inputs into shmem.
		// This is faster than loading it from gmem directly, even though it is only used once.
		// (Possibly due to latency hiding through staging.)
		const uint32_t n_elems_a = BLOCK_DIM_Z * 16 * in_width;

		#pragma unroll
		for (uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
			const uint32_t idx_skewed = idx + idx / in_width * INPUT_SKEW;
			*(int4*)&act_shmem[idx_skewed] = *(int4*)&input_threadblock[l * n_elems_a + idx];
		}

		__syncthreads();

		wmma::fill_fragment(result_frag[l], 0.0f);
		#pragma unroll
		for (uint32_t i = 0; i < n_tensor_ops; ++i) {
			// Load chunk of inputs and weights from shared memory and multiply them
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * threadIdx.z) * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
			wmma::load_matrix_sync(weights_frag, weights_shmem + 16 * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
			wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
		}

		__syncthreads();

		warp_activation<__half>(activation, result_frag[l], result_frag[l]);
	}

	#pragma unroll
	for (int l = 0; l < N_ITERS; ++l) {
		wmma::store_matrix_sync(act_shmem + weights_col + (16 * (threadIdx.z + l * BLOCK_DIM_Z)) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, wmma::mem_row_major);
	}

	if (out_intermediate_threadblock_this_layer != nullptr) {
		__syncthreads();

		#pragma unroll
		for (int i = 0; i < N_ITERS; ++i) {
			*(int4*)&out_intermediate_threadblock_this_layer[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
		}
	}
}




__device__ __half reluhalf(__half v){
	if(v < __half(0.0f))
		return __half(0.0f);
	return v;
}


template <bool INFERENCE, int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T=float, bool FUSED_SHADING = false, uint32_t MIN_SH_BAND=0, uint32_t NUM_SH_BAND=0, bool FUSED_OUTPUT=false>
__device__ void threadblock_last_layer_forward(Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out, const uint32_t batch_size, const nvcuda::wmma::layout_t output_layout, OutputShadingStructure outShadeData=OutputShadingStructure(), Activation output_activation=Activation::None) {
	// act_shmem contains the intermediate activations (shared memory) of the thread block's chunk of the batch
	// weights_this_layer points to the weight matrix of the current layer
	// out points to the location where the result produced by the thread block should be written to.
	//   Can be nullptr if nothing should be written.

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;
	constexpr uint32_t N_BLOCKS = WIDTH / 16;

	using namespace nvcuda;

	// Fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> act_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag[N_BLOCKS];
	wmma::fragment<wmma::accumulator, 16, 16, 16, OUT_T> result_frag;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	__half* __restrict__ weights_shmem = act_shmem + N_ITERS * BLOCK_DIM_Z * 16 * (WIDTH + SKEW);

	const uint32_t weights_row = (8 * li) % WIDTH;
	const uint32_t weights_col = (8 * li + 8 * 32 * wi) / WIDTH;

	// Load weight matrix into shared memory for the last multiplication.
	// Loading into shared memory as opposed to directly into registers is faster
	// because unlike in the previous layers, each warp uses the same entries of the weight matrix.
	if (threadIdx.z == 0) {
		*(int4*)&weights_shmem[weights_row + weights_col * (WIDTH + SKEW)] = *(int4*)&weights_this_layer[weights_row + weights_col * WIDTH];
	}

	__syncthreads();

	#pragma unroll
	for (uint32_t i = 0; i < N_BLOCKS; ++i)
		wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16 * i, WIDTH + SKEW);

	const bool isUniformActivation = activation != Activation::VMF && activation != Activation::SH;

	const uint32_t elem_offset = WIDTH + SKEW;

	// Perform last layer by parallelizing over iters
	for (uint32_t idx = wi; idx < N_ITERS; idx += N_BLOCKS) {
		wmma::fill_fragment(result_frag, 0.0f);
		#pragma unroll
		for (uint32_t i = 0; i < N_BLOCKS; ++i) {
			// Load a chunk of intermediate activations from shared memory and multiply with chunk of the weight matrix
			wmma::load_matrix_sync(act_frag, act_shmem + 16 * i + (16 * (threadIdx.z + idx * BLOCK_DIM_Z)) * (WIDTH + SKEW), WIDTH + SKEW);
			wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
		}

		//Doesnt neeed for nonuniform as we have a separate one
		if (isUniformActivation) {
			warp_activation<__half>(activation, result_frag, result_frag);
		}

		if (INFERENCE && ((output_activation	== Activation::SH && FUSED_SHADING) || FUSED_OUTPUT)) {
			if (output_layout == wmma::mem_row_major) {
				// Just push output neurons in the same place where we had the first 16 intermediate activations for the last hidden layer. Reusage of shared memory! 
				wmma::store_matrix_sync(act_shmem + (16 * (threadIdx.z + idx * BLOCK_DIM_Z)) * elem_offset, result_frag, elem_offset, output_layout);
			}
			else {
				wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16, result_frag, batch_size, output_layout);
			}
		}
		else{
			if (output_layout == wmma::mem_row_major) {
				wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16 * 16, result_frag, 16, output_layout);
			} else {
				wmma::store_matrix_sync(out + (threadIdx.z + idx * BLOCK_DIM_Z) * 16, result_frag, batch_size, output_layout);
			}
		}
	}


	if (INFERENCE && output_activation == Activation::SH && FUSED_SHADING) {
		const uint32_t g_id = wi + (li > 15) * N_BLOCKS;
		const uint32_t li_i = li % 16;
		// group_offset + offset inside subbatch (128 = 16x8) + offset inside subsubbatch
		const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS * BLOCK_DIM_Z + g_id * 16 + li_i;
		const uint32_t shmem_offset = (16 * (threadIdx.z + g_id * BLOCK_DIM_Z)) * elem_offset;
		half* data_ptr = act_shmem + shmem_offset + li_i * elem_offset;
		if (elem_idx < outShadeData.final_output_nelements) {

			
			SHTransformData sh = readSHTransformData(outShadeData.output_trasnform_data, elem_idx);
			float res_r, res_g, res_b;

			
			compute_sh_cv_templated<MIN_SH_BAND, NUM_SH_BAND>(data_ptr, sh, res_r, res_g, res_b, outShadeData.bands_shade_weights);
#if 1
			outShadeData.final_output[3 * elem_idx] = res_r;
			outShadeData.final_output[3 * elem_idx+1] = res_g;
			outShadeData.final_output[3 * elem_idx+2] = res_b;
#else
			outShadeData.final_output[3 * elem_idx] = 1.0f;
			outShadeData.final_output[3 * elem_idx+1] = 0.0f;
			outShadeData.final_output[3 * elem_idx+2] = 0.0f;
#endif


		}
	}

	#if 1
	if (INFERENCE && FUSED_OUTPUT) {
		const uint32_t g_id = wi + (li > 15) * N_BLOCKS;
		const uint32_t li_i = li % 16;
		// group_offset + offset inside subbatch (128 = 16x8) + offset inside subsubbatch
		const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS * BLOCK_DIM_Z + g_id * 16 + li_i;
		const uint32_t shmem_offset = (16 * (threadIdx.z + g_id * BLOCK_DIM_Z)) * elem_offset;
		half* data_ptr = act_shmem + shmem_offset + li_i * elem_offset;
		if (elem_idx < c_input.num_elements) {
			float3 res;
			float3 thp = c_output.thp[elem_idx];
			int scatter_id = c_output.mapping[elem_idx];


#if 1
			res.x = float(reluhalf(data_ptr[0])) * thp.x;
			res.y = float(reluhalf(data_ptr[1])) * thp.y;
			res.z = float(reluhalf(data_ptr[2])) * thp.z;

			atomicAdd(&(c_output.output[scatter_id*3]), res.x);
			atomicAdd(&(c_output.output[scatter_id*3+1]), res.y);
			atomicAdd(&(c_output.output[scatter_id*3+2]), res.z);
#else
			c_output.output[scatter_id * 3] = float(reluhalf(data_ptr[0]));
			c_output.output[scatter_id * 3+1] = float(reluhalf(data_ptr[1]));
			c_output.output[scatter_id * 3+2] = float(reluhalf(data_ptr[2]));


			c_output.output[scatter_id * 3] = 1.0f;
			c_output.output[scatter_id * 3 + 1] = 1.0f;
			c_output.output[scatter_id * 3 + 2] = 1.0f;
#endif
		}
	}

	#endif
}

template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS>
__device__ void threadblock_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_threadblock) {
	// output_threadblock will be filled by the thread block's act_shmem

	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")

	const uint32_t lane_offset = (8 * li) % WIDTH;
	const uint32_t row = (8 * li + wi * 8 * 32) / WIDTH;

	__syncthreads();

	#pragma unroll
	for (int i = 0; i < N_ITERS; ++i) {
		*(int4*)&output_threadblock[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * WIDTH] = *(int4*)&act_shmem[lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW)];
	}

	// lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)) * (WIDTH + SKEW) to lane_offset + (row + 16 * (threadIdx.z + i * BLOCK_DIM_Z)

}



template <int WIDTH, int BLOCK_DIM_Z, int N_ITERS, typename OUT_T, Activation ACTIVATION, bool INFERENCE, bool FUSED_SHADING = false, uint32_t MIN_SH_BAND = 0, uint32_t NUM_SH_BAND = 0, bool INPUT_SEPARATE=false, bool FUSED_OUTPUT=false>
__global__ void kernel_mlp_fused(const Activation output_activation, const __half* __restrict__ input, const __half* __restrict__ weights, OUT_T* __restrict__ out_intermediate, OUT_T* __restrict__ out, const uint32_t batch_size, const uint32_t in_width, const uint32_t out_width, const uint32_t n_hidden_matmuls, const uint32_t m_sh_offset, const nvcuda::wmma::layout_t output_layout = nvcuda::wmma::mem_row_major, OutputShadingStructure outShadeData = OutputShadingStructure()) {
	// `input` points to the input matrix. Can be any width.
	// `weights` points to the weight matrices (contiguous in memory).
	// `out_intermediate` points to the memory where intermediate activations should be written. When performing inference, a value of nullptr is expected (intermediate results are not written).
	// `out` points to the memory where the network output should be written. (Output width is assumed to be 16 neurons.)

	// Commented out due to isolated strange side-effects on Windows
	// if (INFERENCE) {
	// 	assert(out_intermediate == nullptr);
	// } else {
	// 	assert(out_intermediate);
	// }

	// Shared memory contains the intermediate activations of blockDim.y*16 elements.
	// In some cases, it also contains the weight matrix for the first and last layer.
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	// Each block computes exactly one 16-element chunk of the batch.
	const uint32_t elem_idx = 16 * blockIdx.x * N_ITERS * BLOCK_DIM_Z;

	// First layer
	if (in_width == WIDTH) {
		// If the input has the same width as the network, we can simply use the network's regular layer routine (with static size)
		// instead of using the slower dynamic input layer routine.
		threadblock_load_input_static<WIDTH, BLOCK_DIM_Z, N_ITERS, INFERENCE, INPUT_SEPARATE>(act_shmem, input + elem_idx * WIDTH, m_sh_offset);
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
	} else {
		threadblock_input_layer_forward_dynamic<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width);
	}

	const uint32_t first_layer_size = WIDTH * in_width;
	const uint32_t layer_stride = WIDTH * WIDTH;
	const uint32_t output_stride = WIDTH * batch_size;



	// Hidden layers
	for (uint32_t k = 0; k < n_hidden_matmuls; ++k) {
		threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_layer_size + layer_stride * k, !INFERENCE ? (out_intermediate + output_stride * (k + 1) + elem_idx * WIDTH) : nullptr);
	}

	

	if (out_width > 16) {
	
		// In the forward pass, intermediate activations are already written out.
		if (INFERENCE) {
			if (FUSED_SHADING) {
				threadblock_layer<WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, false, FUSED_SHADING, MIN_SH_BAND, NUM_SH_BAND>(ACTIVATION, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, nullptr, nullptr, outShadeData, output_activation, out_width);
			}
			else {
				threadblock_write_output_static<WIDTH, BLOCK_DIM_Z, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
			}
		}
	} else if (out) {
		// Last layer
		if (output_layout == nvcuda::wmma::mem_row_major) {
			threadblock_last_layer_forward<INFERENCE, WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, FUSED_SHADING, MIN_SH_BAND, NUM_SH_BAND, FUSED_OUTPUT>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, out + elem_idx * 16, 16, output_layout, outShadeData, output_activation);
		} else {	
			threadblock_last_layer_forward<INFERENCE, WIDTH, BLOCK_DIM_Z, N_ITERS, OUT_T, false, 0, 0, FUSED_OUTPUT>(output_activation, act_shmem, weights + first_layer_size + layer_stride * n_hidden_matmuls, out + elem_idx, batch_size, output_layout);
		}
	}
}

template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE>
std::enable_if_t<!std::is_same<__half, T>::value> mlp_fused_forward(
	cudaStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrix<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_hidden_layers
) {
	throw std::runtime_error{"The fully fused forward pass only supports __half precision."};
}


template <int WIDTH, typename T, Activation ACTIVATION, bool INFERENCE, bool FUSED_SHADING = false, uint32_t MIN_SH_BAND = 0, uint32_t NUM_SH_BAND = 0>
std::enable_if_t<std::is_same<__half, T>::value> mlp_fused_forward(
	cudaStream_t stream,
	Activation output_activation,
	const GPUMatrix<T, RM>& weights,
	const NeuralInputData<T>& input,
	GPUMatrix<T>& output_intermediate,
	GPUMatrixDynamic<T>* output,
	const uint32_t n_hidden_layers,
	const uint32_t m_sh_offset,
	FusedOutputData outShadeData = FusedOutputData()
) {
	const uint32_t batch_size = input.cols();
	const uint32_t in_width = (input.is_mono()) ? input.rows() : WIDTH;
	const bool bFusedOutput = outShadeData.arr0 != nullptr;
	constexpr uint32_t SKEW = WIDTH % 16 == 0 ? 8 : 0; // <- always going to be 8 as we only support multiple-of-16 widths
	constexpr uint32_t INPUT_SKEW = 8; // <- likewise with inputs
	constexpr uint32_t N_BLOCK_ROWS = WIDTH / 16;

	static_assert(WIDTH % 16 == 0, "Width must be a multiply of 16.");
	if (input.is_mono() && in_width % 16 != 0) {
		throw std::runtime_error{"Inputs must have a multiple-of-16 elements."};
	}

	if (weights.rows() != WIDTH) {
		throw std::runtime_error{"The fully fused forward pass only works with WIDTH-sized matrices."};
	}

	if (weights.cols() % 16 != 0) {
		throw std::runtime_error{std::string("weights must have a multiple-of-16 number of columns. ") + std::to_string(weights.cols())};
	}

	if (!bFusedOutput && !INFERENCE && input.is_mono()&& output_intermediate.cols() != batch_size) {
		throw std::runtime_error{"Batch size of inputs and output_intermediate doesn't match."};
	}

	if (input.is_mono() && output && output->cols() != batch_size) {
		throw std::runtime_error{"Batch size of inputs and outputs doesn't match."};
	}

	const int N_ITERS = WIDTH >= 256 ? 2 : 8;
	const uint32_t BLOCK_DIM_Z = (INFERENCE && WIDTH == 128) ? 2 : 1;

	if (batch_size % (16 * N_ITERS * BLOCK_DIM_Z) != 0) {
		throw std::runtime_error{"Batch size must be a multiple of " + std::to_string(16 * N_ITERS * BLOCK_DIM_Z) + "."};
	}

	// 32, 4, 1
	const dim3 threads = { 32u, N_BLOCK_ROWS, BLOCK_DIM_Z }; // 32 threads = 1 warp, N_BLOCK_ROWS warps per block for 16 rows, up to 2x 8 warps can share input (does not help vs. 1)

	uint32_t n_elems_per_block = 16 * BLOCK_DIM_Z * N_ITERS;
	uint32_t n_blocks = div_round_up(batch_size, n_elems_per_block);

	size_t shmem_size = sizeof(__half) * (16 + 16 * BLOCK_DIM_Z * N_ITERS) * (WIDTH + SKEW); // 16*WIDTH rows of weights (for the last layer; others are in registers only) + 16*WIDTH*BLOCK_DIM_Z*N_ITERS rows of intermediate activations
	if (input.is_mono() && in_width != WIDTH) {
		// If the input width is dynamic, the input weight matrix as well as part of the input will live in extra shared memory
		shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16 * BLOCK_DIM_Z) * (in_width + INPUT_SKEW));
	}

	const dim3 blocks = { n_blocks, 1u, 1u };


	if(!input.is_mono()){


		SeparateInputeCuda5 inputData;
		uint32_t offset =0; 
		for(uint32_t i=0; i < 5; i++){
			inputData.data[i] = (__half*)input.data[i].data();
			inputData.is_mapped[i] = input.mappings[i] != nullptr;
			if(inputData.is_mapped[i])
				inputData.mapping = (uint32_t*)input.mappings[i];
			else
				inputData.num_elements = input.data[i].cols();

			inputData.elements_per_channel[i] = input.data[i].rows();
			inputData.offset[i] = offset; 
			offset += inputData.elements_per_channel[i];
		}

		cudaMemcpyToSymbol(c_input, &inputData,  sizeof(SeparateInputeCuda5));




		if(!bFusedOutput){
			check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE, FUSED_SHADING, MIN_SH_BAND, NUM_SH_BAND, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));

			kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE, FUSED_SHADING , MIN_SH_BAND , NUM_SH_BAND, true, false><<<blocks, threads, shmem_size, stream>>>(
				output_activation,
				input.data[0].data(),
				weights.data(),
				output_intermediate.data(),
				output ? output->data() : nullptr,
				batch_size,
				in_width,
				output ? output->rows() : 0,
				n_hidden_layers,
				m_sh_offset,
				output && output->layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major, // The kernels operate with transposed layouts compared with the MLP code
				OutputShadingStructure()
			);
		}
		else{
			InstantOutput outData;
			outData.thp = reinterpret_cast<float3*>(outShadeData.arr0);
			outData.mapping = reinterpret_cast<uint32_t*>(outShadeData.arr1);
			outData.output = outShadeData.arr2;
			cudaMemcpyToSymbol(c_output, &outData,  sizeof(InstantOutput));

			check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE, FUSED_SHADING, MIN_SH_BAND, NUM_SH_BAND, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));

			kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE, FUSED_SHADING , MIN_SH_BAND , NUM_SH_BAND, true, true><<<blocks, threads, shmem_size, stream>>>(
				output_activation,
				input.data[0].data(),
				weights.data(),
				output_intermediate.data(),
				output ? output->data() : nullptr,
				batch_size,
				in_width,
				output ? output->rows() : 0,
				n_hidden_layers,
				m_sh_offset,
				output && output->layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major, // The kernels operate with transposed layouts compared with the MLP code
				OutputShadingStructure()
			);
		}
	}
	else{

		check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE, FUSED_SHADING, MIN_SH_BAND, NUM_SH_BAND>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size));

		kernel_mlp_fused<WIDTH, BLOCK_DIM_Z, N_ITERS, __half, ACTIVATION, INFERENCE, FUSED_SHADING , MIN_SH_BAND , NUM_SH_BAND ><<<blocks, threads, shmem_size, stream>>>(
			output_activation,
			input.data[0].data(),
			weights.data(),
			output_intermediate.data(),
			output ? output->data() : nullptr,
			batch_size,
			in_width,
			output ? output->rows() : 0,
			n_hidden_layers,
			m_sh_offset,
			output && output->layout() == RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major, // The kernels operate with transposed layouts compared with the MLP code
			OutputShadingStructure()
		);
	}
}


template <typename T, int WIDTH>
FullyFusedMLP<T, WIDTH>::FullyFusedMLP(
	uint32_t input_width,
	uint32_t output_width,
	uint32_t n_hidden_layers,
	bool use_feedback_alignment,
	Activation activation,
	Activation output_activation,
	bool fused_out_transform,
	uint32_t min_sh,
	uint32_t num_sh
) :
m_input_width{input_width},
m_network_width{WIDTH},
m_output_width{output_width},
m_n_hidden_layers{n_hidden_layers},
m_use_feedback_alignment{use_feedback_alignment},
m_activation{activation},
m_output_activation{output_activation},
m_fused_out_transform{fused_out_transform},
m_min_sh_band{min_sh},
m_num_sh_band{num_sh}
{
	if (m_n_hidden_layers <= 0) {
		throw std::runtime_error("FullyFusedMLP requires at least 1 hidden layer (3 layers in total).");
	}


	bands_shade_weights.resize(4);
	for(uint32_t i=0; i < 4; i++)
		bands_shade_weights[i] = 1.0f;

	m_n_hidden_matmuls = n_hidden_layers-1;


	m_padded_output_width = next_multiple(m_output_width, tensorcore_width);

	// Create matrices related to weights
	m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_input_width);
	m_weight_matrices_backward.emplace_back(nullptr, m_network_width, m_input_width);
	m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_input_width);
	m_gradient_matrices.emplace_back(nullptr, m_network_width, m_input_width);

	for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
		m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
		m_weight_matrices_inference.emplace_back(nullptr, m_network_width, m_network_width);
		m_weight_matrices_backward.emplace_back(nullptr, m_network_width, m_network_width);
		m_weight_matrices_full_precision.emplace_back(nullptr, m_network_width, m_network_width);
		m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);
	}

	m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_weight_matrices_inference.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_weight_matrices_backward.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_weight_matrices_full_precision.emplace_back(nullptr, m_padded_output_width, m_network_width);
	m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);

	// Determine total number of memory entries and set it
	m_total_n_params = 0;
	for (const auto& m : m_weight_matrices) {
		m_total_n_params += m.n_elements();
	}

	// Buffers to keep data from the forward and backward pass
	m_forward_tmp.resize(m_n_hidden_layers);
	m_backward_tmp.resize(m_n_hidden_layers);

	// 1 stream per matmul
	m_training_splitk_streams.resize(m_n_hidden_layers + 1);
	m_training_splitk_events.resize(m_n_hidden_layers + 1);

	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		CUDA_CHECK_THROW(cudaStreamCreate(&m_training_splitk_streams[i]));
		CUDA_CHECK_THROW(cudaEventCreate(&m_training_splitk_events[i]));
	}
}

template <typename T, int WIDTH>
FullyFusedMLP<T, WIDTH>::~FullyFusedMLP() {
	for (size_t i = 0; i < m_training_splitk_streams.size(); ++i) {
		cutlass_free_workspace(m_training_splitk_streams[i]);

		CUDA_CHECK_PRINT(cudaEventDestroy(m_training_splitk_events[i]));
		CUDA_CHECK_PRINT(cudaStreamDestroy(m_training_splitk_streams[i]));
	}
}



template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::inference(cudaStream_t stream, const GPUMatrix<T>& input, GPUMatrix<float>& output, GPUMatrix<float>* outTransformData) {
	NeuralInputData<T> temp;
	temp.addInput(input);

	inference_mixed_precision(stream, temp, m_inference_output_tmp);

	//const uint32_t n_elements = (uint32_t)output.n_elements()
	const uint32_t n_elements = (uint32_t)output.n_elements();
	if(m_output_activation == Activation::VMF)
		apply_vmf<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream >> > (n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data());
	if(m_output_activation == Activation::SH){
		//compute_sh_cv_store<T><<<n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_output_width, m_inference_output_tmp.data(), outTransformData->data(), output.data());
	}
	else{
		trim_and_cast<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data(), output.data());
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::inference(cudaStream_t stream, NeuralInputData<T>& input, GPUMatrix<float>& output,
					uint32_t size, int* mapping, float* bbStart, float* bbEnd, FusedOutputData outShadeData) {
	/*OutputShadingStructure outShadeData = OutputShadingStructure(1);

	outShadeData.output_trasnform_data = outTransformData->data();
	outShadeData.final_output = output.data();
	outShadeData.final_output_nelements = size;
	outShadeData.bands_shade_weights = Const4FloatArray(bands_shade_weights.data());*/
	outShadeData.arr2 = reinterpret_cast<float*>(output.data());
	inference_mixed_precision(stream, input, m_inference_output_tmp, true, outShadeData);

	//const uint32_t n_elements = (uint32_t)output.n_elements();
	if(outShadeData.arr0 != nullptr){
		return;
	}


	if(false){
		const uint32_t n_elements = (uint32_t)output.n_elements();
		if( m_output_activation == Activation::VMF)
			apply_vmf<T> << <n_blocks_linear(n_elements), n_threads_linear, 0, stream >> > (n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data());
		if(m_output_activation == Activation::SH)
			apply_vmf<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream >> > (n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data());
		trim_and_cast<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_padded_output_width, m_output_width, m_inference_output_tmp.data(), output.data());
	}
	else{
		if(m_output_activation != Activation::SH)
			trim_and_cast<T><<<n_blocks_linear(size*m_output_width), n_threads_linear, 0, stream>>>(size* m_output_width, m_padded_output_width, m_output_width, m_inference_output_tmp.data(), output.data(), mapping, bbStart, bbEnd);
		else if(!m_fused_out_transform){
			/*	switch (m_min_sh_band) {
					case 0:
						if(m_num_sh_band == 1)
							compute_sh_cv_store<0, 1, T> << <n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_inference_output_tmp.data(), outShadeData);
						else if(m_num_sh_band == 2)
							compute_sh_cv_store<0, 2, T> << <n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_inference_output_tmp.data(), outShadeData);
						else if (m_num_sh_band == 3)
							compute_sh_cv_store<0, 3, T> << <n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_inference_output_tmp.data(), outShadeData);
						break;
					case 1:
						if (m_num_sh_band == 1)
							compute_sh_cv_store<1, 1, T> << <n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_inference_output_tmp.data(), outShadeData);
						else if (m_num_sh_band == 2)
							compute_sh_cv_store<1, 2, T> << <n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_inference_output_tmp.data(), outShadeData);
						break;
					case 2:
						compute_sh_cv_store<2, 1, T> << <n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_inference_output_tmp.data(), outShadeData);
						break;
					case 3:
						compute_sh_cv_store<3, 1, T> << <n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_inference_output_tmp.data(), outShadeData);
						break;
					case 4:
						compute_sh_cv_store<4, 1, T> << <n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_inference_output_tmp.data(), outShadeData);
						break;
					case 5:
						compute_sh_cv_store<5, 1, T> << <n_blocks_linear(output.n()), n_threads_linear, 0, stream >> > (output.n(), m_padded_output_width, m_inference_output_tmp.data(), outShadeData);
						break;
				};*/
		}
	}


	
}

template <typename CutlassLayer, MatrixLayout input_layout, typename T>
void compute_inference_layer(
	cudaStream_t stream,
	Activation activation,
	const GPUMatrix<T, RM>& weights,
	const GPUMatrix<T, input_layout>& input,
	GPUMatrixDynamic<T>& output
) {
	fc_multiply<CutlassLayer>(stream, weights, input, output, activation);
}


#define mlp_fused_forward_macro(WIDTH, T, ACT, INFERENCE, FUSED_SHADING, MIN_SH, NUM_SH) mlp_fused_forward<WIDTH, T, ACT, INFERENCE, FUSED_SHADING, MIN_SH, NUM_SH>(stream, m_output_activation, input_weight_matrix(weight_usage), input_data, m_inference_tmp, &output, m_n_hidden_matmuls, m_sh_offset, outShadeData)

//#define mlp_fused_forward_fused_transform_macro(WIDTH, T, ACT) if(m_min_sh_band == 0){ \
//	if (m_num_sh_band == 1) \
//		mlp_fused_forward_macro(WIDTH, T, ACT, true, true, 0, 1); \
//	else if (m_num_sh_band == 2) \
//		mlp_fused_forward_macro(WIDTH, T, ACT, true, true, 0, 2); \
//	else if (m_num_sh_band == 3) \
//		mlp_fused_forward_macro(WIDTH, T, ACT, true, true, 0, 3); \
//	else \
//		throw std::runtime_error{ "Doesn't support more than 3 bands after the zero one" }; \
//} \
//else if (m_min_sh_band == 1) { \
//	if (m_num_sh_band == 1) \ 
//		mlp_fused_forward_macro(WIDTH, T, ACT, true, true, 1, 1); \
//	else if (m_num_sh_band == 2) \
//		mlp_fused_forward_macro(WIDTH, T, ACT, true, true, 1, 2); \
//	else \
//		throw std::runtime_error{ "Doesn't support more than 2 bands after the first one" }; \
//} \
//switch (m_min_sh_band) { \
//case 0: break; \ 
//case 1: break; \
//case 2: mlp_fused_forward_macro(WIDTH, T, ACT, true, true, 2, 1); break; \
//case 3: mlp_fused_forward_macro(WIDTH, T, ACT, true, true, 3, 1); break; \
//case 4: mlp_fused_forward_macro(WIDTH, T, ACT, true, true, 4, 1); break; \
//case 5: mlp_fused_forward_macro(WIDTH, T, ACT, true, true, 5, 1); break; \
//} 

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::inference_mixed_precision(cudaStream_t stream, NeuralInputData<T>& input_data, GPUMatrixDynamic<T>& output, bool use_inference_matrices, FusedOutputData outShadeData) {
	// Various error checks

	if (input_data.is_mono() && input_data.data[0].m() != m_input_width ) {
		throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input_data.m()) + "!=" + std::to_string(m_input_width));
	}

	if (&output != &m_inference_output_tmp && output.m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output has incorrect width: ") + std::to_string(output.m()) + "!=" + std::to_string(m_output_width));
	}

	if (&output != &m_inference_output_tmp && input_data.is_mono() && input_data.data[0].n() != output.n()) {
		throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input_data.n()) + "!=" + std::to_string(output.n()));
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	const bool bFusedOuput = outShadeData.arr0 != nullptr;

	uint32_t batch_size = input_data.n();
	if(!bFusedOuput && m_inference_output_tmp.n() != batch_size)
		allocate_inference_buffers(batch_size, true);


	const WeightUsage weight_usage = use_inference_matrices ? WeightUsage::Inference : WeightUsage::Forward;

	if(m_min_sh_band > 5)
		throw std::runtime_error{ "Doesn't support more than 6 bands" };

	// ASSUMPTION: weight matrices are contiguous in memory
	switch (m_activation) {
	case Activation::None:
	{
		if (!m_fused_out_transform) {
			mlp_fused_forward_macro(WIDTH, T, Activation::None, true, false, 0, 0);
		}
		else {
			if(m_min_sh_band == 0){
				if (m_num_sh_band == 1) 
					mlp_fused_forward_macro(WIDTH, T, Activation::None, true, true, 0, 1);
				else if (m_num_sh_band == 2) 
					mlp_fused_forward_macro(WIDTH, T, Activation::None, true, true, 0, 2);
				else if (m_num_sh_band == 3) 
					mlp_fused_forward_macro(WIDTH, T, Activation::None, true, true, 0, 3);
				else 
					throw std::runtime_error{ "Doesn't support more than 3 bands after the zero one" }; 
			} 
			else if (m_min_sh_band == 1) { 
				if (m_num_sh_band == 1) 
					mlp_fused_forward_macro(WIDTH, T, Activation::None, true, true, 1, 1);
				else if (m_num_sh_band == 2) 
					mlp_fused_forward_macro(WIDTH, T, Activation::None, true, true, 1, 2);
				else 
					throw std::runtime_error{ "Doesn't support more than 2 bands after the first one" }; 
				} 
			switch (m_min_sh_band) {			
				case 0: break; 
				case 1: break; 
				case 2: mlp_fused_forward_macro(WIDTH, T, Activation::None, true, true, 2, 1); break;
				case 3: mlp_fused_forward_macro(WIDTH, T, Activation::None, true, true, 3, 1); break;
				case 4: mlp_fused_forward_macro(WIDTH, T, Activation::None, true, true, 4, 1); break;
				case 5: mlp_fused_forward_macro(WIDTH, T, Activation::None, true, true, 5, 1); break;
			}
		}

		//mlp_fused_forward_fused_transform_macro(WIDTH, T, Activation::None)
		break;
	}

	case Activation::ReLU:
	{
		if (!m_fused_out_transform) {
			mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, false, 0, 0);
		}
		else {
			if (m_min_sh_band == 0) {
				if (m_num_sh_band == 1)
					mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, true, 0, 1);
				else if (m_num_sh_band == 2)
					mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, true, 0, 2);
				else if (m_num_sh_band == 3)
					mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, true, 0, 3);
				else
					throw std::runtime_error{ "Doesn't support more than 3 bands after the zero one" };
			}
			else if (m_min_sh_band == 1) {
				if (m_num_sh_band == 1)
					mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, true, 1, 1);
				else if (m_num_sh_band == 2)
					mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, true, 1, 2);
				else
					throw std::runtime_error{ "Doesn't support more than 2 bands after the first one" };
			}
			switch (m_min_sh_band) {
			case 0: break;
			case 1: break;
			case 2: mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, true, 2, 1); break;
			case 3: mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, true, 3, 1); break;
			case 4: mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, true, 4, 1); break;
			case 5: mlp_fused_forward_macro(WIDTH, T, Activation::ReLU, true, true, 5, 1); break;
			}

		}
		break;
	}

#if !FAST_COMPILATION
	case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential, true>(stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls, m_sh_offset, outShadeData); break;
	case Activation::Sigmoid:     mlp_fused_forward<WIDTH, T, Activation::Sigmoid, true>(    stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls,  m_sh_offset, outShadeData); break;*/
	case Activation::Squareplus:  mlp_fused_forward<WIDTH, T, Activation::Squareplus, true>( stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls,  m_sh_offset, outShadeData); break;
	case Activation::Softplus:    mlp_fused_forward<WIDTH, T, Activation::Softplus, true>(   stream, m_output_activation, input_weight_matrix(weight_usage), input, m_inference_tmp, &output, m_n_hidden_matmuls,  m_sh_offset, outShadeData); break;
#endif

	default: throw std::runtime_error{"Unsupported activation."};
	}

	// If we have more than 16 output dimensions, these will be taken care of by CUTLASS rather than
	// the fully fused kernel (which will have written out the second-to-last layer activations).
	if (m_output_width > 16) {
		compute_inference_layer<LastLayer>(stream, (m_output_activation == Activation::SH) ? Activation::None : m_output_activation, output_weight_matrix(weight_usage), m_inference_tmp, output);
	}
}
template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::forward(cudaStream_t stream, NeuralInputData<T>& input, uint32_t size, int* mapping, float* bbStart, float* bbEnd, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_matrices = false, bool prepare_input_gradients = false) {
	throw std::runtime_error(std::string("We shouldn't be here...") ); 
	//forward(stream, input, output, use_inference_matrices, prepare_input_gradients);
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::forward(cudaStream_t stream, const GPUMatrix<T>& input, GPUMatrixDynamic<T>* output, bool use_inference_matrices, bool prepare_input_gradients) {
	// Various error checks
	if (input.m() != m_input_width) {
		throw std::runtime_error(std::string("Input has incorrect width: ") + std::to_string(input.m()) + "!=" + std::to_string(m_input_width));
	}

	if (output && output->m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output has incorrect width (must be padded): ") + std::to_string(output->m()) + "!=" + std::to_string(m_padded_output_width));
	}

	if (output && input.n() != output->n()) {
		throw std::runtime_error(std::string("Input and output don't have matching batch size: ") + std::to_string(input.n()) + "!=" + std::to_string(output->n()));
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = input.n();
	if(m_forward_tmp.front().n() != batch_size)
		allocate_forward_buffers(batch_size, true);

	const WeightUsage weight_usage = use_inference_matrices ? WeightUsage::Inference : WeightUsage::Forward;

	NeuralInputData<T> temp;
	temp.addInput(input);

	// ASSUMPTION: weight matrices & forward_tmp matrices are contiguous in memory
	switch (m_activation) {
		case Activation::None:        mlp_fused_forward<WIDTH, T, Activation::None, false>(       stream, m_output_activation, input_weight_matrix(weight_usage), temp, m_forward_tmp.at(0), output, m_n_hidden_matmuls, m_sh_offset); break;
		case Activation::Exponential: mlp_fused_forward<WIDTH, T, Activation::Exponential, false>(stream, m_output_activation, input_weight_matrix(weight_usage), temp, m_forward_tmp.at(0), output, m_n_hidden_matmuls, m_sh_offset); break;
		case Activation::Sigmoid:     mlp_fused_forward<WIDTH, T, Activation::Sigmoid, false>(    stream, m_output_activation, input_weight_matrix(weight_usage), temp, m_forward_tmp.at(0), output, m_n_hidden_matmuls, m_sh_offset); break;
		case Activation::ReLU:        mlp_fused_forward<WIDTH, T, Activation::ReLU, false>(       stream, m_output_activation, input_weight_matrix(weight_usage), temp, m_forward_tmp.at(0), output, m_n_hidden_matmuls, m_sh_offset); break;
		case Activation::Squareplus:  mlp_fused_forward<WIDTH, T, Activation::Squareplus, false>( stream, m_output_activation, input_weight_matrix(weight_usage), temp, m_forward_tmp.at(0), output, m_n_hidden_matmuls, m_sh_offset); break;
		case Activation::Softplus:    mlp_fused_forward<WIDTH, T, Activation::Softplus, false>(   stream, m_output_activation, input_weight_matrix(weight_usage), temp, m_forward_tmp.at(0), output, m_n_hidden_matmuls, m_sh_offset); break;
		case Activation::VMF:         mlp_fused_forward<WIDTH, T, Activation::VMF, false>(        stream, m_output_activation, input_weight_matrix(weight_usage), temp, m_forward_tmp.at(0), output, m_n_hidden_matmuls, m_sh_offset); break;
		case Activation::SH:         mlp_fused_forward<WIDTH, T, Activation::SH, false>(        stream, Activation::None, input_weight_matrix(weight_usage), temp, m_forward_tmp.at(0), output, m_n_hidden_matmuls, m_sh_offset); break;
		default: throw std::runtime_error{"Unsupported activation."};
	}

	// If we have more than 16 output dimensions, these will be taken care of by CUTLASS rather than
	// the fully fused kernel (which will have written out the second-to-last layer activations).
	if (output && m_output_width > 16) {
		compute_inference_layer<LastLayer>(stream, (m_output_activation == Activation::SH) ? Activation::None : m_output_activation, output_weight_matrix(weight_usage), m_forward_tmp.back(), *output);
	}

	const uint32_t n_elements = (uint32_t)output->n_elements();

	
	if (m_output_activation == Activation::VMF) {
		apply_vmf<T> << <n_blocks_linear(n_elements), n_threads_linear, 0, stream >> > (n_elements, m_padded_output_width, m_output_width, output->data());
	}
}


template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::backward(
	cudaStream_t stream,
	const GPUMatrix<T>& input,
	const GPUMatrixDynamic<T>& output,
	const GPUMatrixDynamic<T>& dL_doutput,
	GPUMatrix<T>* dL_dinput,
	bool use_inference_matrices,
	bool compute_param_gradients
) {
	if (dL_doutput.m() != m_padded_output_width) {
		throw std::runtime_error(std::string("Output gradients have incorrect width (must be padded): ") + std::to_string(dL_doutput.m()) + "!=" + std::to_string(m_padded_output_width));
	}

	// Make sure our temporary buffers have the correct size for the given batch size
	uint32_t batch_size = dL_doutput.n();
	if (m_backward_tmp.front().n() != batch_size) {
		allocate_backward_buffers(batch_size);
	}

	// Compute transfer of output activation in-place... it's treated specially for performance reasons
	if (m_output_activation != Activation::None) {
		if(m_output_activation == Activation::VMF){
			const uint32_t n_elements = (uint32_t)output.n_elements();
			vmf_transfer_output<T><<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, output.data(), dL_doutput.data(), m_backward_output_tmp.data(), m_padded_output_width, VMF_COUNT*5);
		}
		else
			activation_backward_output_gpu(stream, dL_doutput.n_elements(), (m_output_activation == Activation::SH) ? Activation::None : m_output_activation, output.data(), dL_doutput.data(), m_backward_output_tmp.data());
	}
	// Backprop
	// - weight_gradient.T = activation * output_gradient.T
	// - input_gradient = weights.T * output_gradient
	// - RELU: pre_activation_gradinet = post_activation_gradient if val > 0 else 0

	const WeightUsage weight_usage = use_inference_matrices ? WeightUsage::Inference : WeightUsage::Backward;

	//uint32_t old_forward_size = m_forward_tmp[0].cols();
	//// CRUNCH
	//for (uint32_t i = 0; i < m_forward_tmp.size(); i++)
	//	m_forward_tmp[i].set_size_const(m_forward_tmp[i].rows(), batch_size);


	{
		int split_k_factor = batch_size / std::min((uint32_t)(1 << 12), batch_size);

		m_backward_output_tmp.set_layout(dL_doutput.layout());
		const GPUMatrixDynamic<T>& tmp_dL_doutput = (m_output_activation == Activation::None || m_output_activation == Activation::SH)  ? dL_doutput : m_backward_output_tmp;

		uint32_t tmp_idx = m_n_hidden_matmuls;
		uint32_t backward_tmp_idx = 0;

		if (compute_param_gradients) {
			// Output layer
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);

			// Compute weight gradients
			fc_multiply_split_k<LastLayerK>(m_training_splitk_streams.at(backward_tmp_idx), tmp_dL_doutput, m_forward_tmp.at(tmp_idx).transposed(), output_gradient_matrix(), split_k_factor);

			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		// If the output width is larger than 16 dims, we use cutlass to backpropagate through the last layer
		// rather than fusing it with our kernel.
		if (m_output_width > 16) {
			fc_multiply<FullLayer>(stream, output_weight_matrix(weight_usage).transposed(), tmp_dL_doutput, m_forward_tmp.at(tmp_idx), m_backward_tmp.at(backward_tmp_idx), m_activation, true);
		}

		// ASSUMPTION: weight matrices & forward_tmp matrices are contiguous in memory
		auto dL_dinput_fused = input.m() == m_forward_tmp.at(0).m() ? dL_dinput : nullptr; // Only let the fully fused kernel compute gradients w.r.t. the input, if the input layer has the same size as the other layers

		switch (m_activation) {
			case Activation::None:        mlp_fused_backward<WIDTH, T, Activation::None>(       stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls, m_sh_offset); break;
			case Activation::Exponential: mlp_fused_backward<WIDTH, T, Activation::Exponential>(stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls, m_sh_offset); break;
			case Activation::Sigmoid:     mlp_fused_backward<WIDTH, T, Activation::Sigmoid>(    stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls, m_sh_offset); break;
			case Activation::ReLU:        mlp_fused_backward<WIDTH, T, Activation::ReLU>(       stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls, m_sh_offset); break;
			case Activation::Squareplus:  mlp_fused_backward<WIDTH, T, Activation::Squareplus>( stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls, m_sh_offset); break;
			case Activation::Softplus:    mlp_fused_backward<WIDTH, T, Activation::Softplus>(   stream, input_weight_matrix(weight_usage), weight_matrix_at(weight_usage, 0), tmp_dL_doutput, m_backward_tmp.at(backward_tmp_idx), m_forward_tmp.at(0), dL_dinput_fused, m_n_hidden_matmuls, m_sh_offset); break;
			default: throw std::runtime_error{"Unsupported activation."};
		}

		tmp_idx -= 1;
		++backward_tmp_idx;

		// layers
		for (uint32_t i = 0; i < m_n_hidden_matmuls; ++i) {
			uint32_t matrix_idx = m_n_hidden_matmuls - i - 1;

			if (compute_param_gradients) {
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
				cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
				fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), m_backward_tmp.at(backward_tmp_idx-1), m_forward_tmp.at(tmp_idx).transposed(), gradient_matrix_at(matrix_idx), split_k_factor);
				cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
			}

			tmp_idx -= 1;
			++backward_tmp_idx;
		}

		if (compute_param_gradients) {
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), stream);
			cudaStreamWaitEvent(m_training_splitk_streams.at(backward_tmp_idx), m_training_splitk_events.at(backward_tmp_idx), 0);
			fc_multiply_split_k<FullLayerK>(m_training_splitk_streams.at(backward_tmp_idx), m_backward_tmp.at(backward_tmp_idx-1), input.transposed(), input_gradient_matrix(), split_k_factor);
			cudaEventRecord(m_training_splitk_events.at(backward_tmp_idx), m_training_splitk_streams.at(backward_tmp_idx));
		}

		// If requested and if the fully fused kernel didn't already take care of it, compute sensitivity of loss w.r.t. inputs
		if (dL_dinput && input.m() != m_forward_tmp.at(0).m()) {
			// TODO: optimization opportunity to only compute sensitivity w.r.t selected SUBSET of inputs. Useful for NFs, where conditional dims stay the same.
			fc_multiply<FullLayer>(stream, input_weight_matrix(weight_usage).transposed(), m_backward_tmp.at(backward_tmp_idx-1), *dL_dinput);
		}
	}

	//for (uint32_t i = 0; i < m_forward_tmp.size(); i++)
	//	m_forward_tmp[i].set_size_const(m_forward_tmp[i].rows(), old_forward_size);

	if (compute_param_gradients) {
		// All the per-layer split-k matrix multiplications summing over
		// the batch are computed in parallel streams to the actual
		// backpropagation. Here, we need to wait for all of these to complete.
		for (auto& event : m_training_splitk_events) {
			cudaStreamWaitEvent(stream, event, 0);
		}
	}
}



template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::allocate_inference_buffers(uint32_t batch_size, bool bAllocate=true) {
	//m_inference_tmp.set_size(m_network_width, batch_size);
	m_inference_output_tmp.set_size(m_padded_output_width, batch_size);

	if(bAllocate){
		GPUMatrixBase::allocate_shared_memory(
			m_inference_buffer,
			{
				//&m_inference_tmp,
				&m_inference_output_tmp,
			}
		);
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::allocate_forward_buffers(uint32_t batch_size, bool bAllocate=true) {
	for (size_t i = 0; i < m_forward_tmp.size(); ++i) {
		m_forward_tmp[i].set_size(m_network_width, batch_size);
	}

	if(bAllocate){
		GPUMatrixBase::allocate_shared_memory(m_forward_buffer, m_forward_tmp);
	}
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::allocate_backward_buffers(uint32_t batch_size) {
	std::vector<GPUMatrixBase*> matrix_pointers = {&m_backward_output_tmp};

	m_backward_output_tmp.set_size(m_padded_output_width, batch_size);
	for (uint32_t i = 0; i < (uint32_t)m_backward_tmp.size(); ++i) {
		m_backward_tmp[i].set_size(m_network_width, batch_size);
		matrix_pointers.emplace_back(&m_backward_tmp[i]);
	}

	GPUMatrixBase::allocate_shared_memory(m_backward_buffer, matrix_pointers);
}

template <typename T, int WIDTH>
void FullyFusedMLP<T, WIDTH>::initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale) {
	size_t current_pos = 0;
	for (size_t i = 0; i < m_weight_matrices.size(); ++i) {
		m_weight_matrices[i].set_data(params + current_pos);
		m_weight_matrices_inference[i].set_data(inference_params + current_pos);
		m_weight_matrices_backward[i].set_data((m_use_feedback_alignment ? backward_params : params) + current_pos);
		m_weight_matrices_full_precision[i].set_data(params_full_precision + current_pos);
		m_gradient_matrices[i].set_data(gradients + current_pos);

		current_pos += m_weight_matrices[i].n_elements();
	}

	for (size_t i = 0; i < m_weight_matrices_full_precision.size(); ++i) {
		if (m_activation == Activation::Sine) {
			if (i == 0) {
				m_weight_matrices_full_precision[i].initialize_siren_uniform_first(rnd, scale);
			} else {
				m_weight_matrices_full_precision[i].initialize_siren_uniform(rnd, scale);
			}
		} else if (m_use_feedback_alignment) {
			m_weight_matrices_full_precision[i].initialize_fa_uniform_forward(rnd, scale);
		} else {
			m_weight_matrices_full_precision[i].initialize_xavier_uniform(rnd, scale);
		}
	}

	// Initialize backward params for feedback alignment
	if (m_use_feedback_alignment) {
		for (size_t i = 0; i < m_weight_matrices_backward.size(); ++i) {
			m_weight_matrices_backward[i].initialize_fa_uniform_backward(rnd, scale);
		}
	}
}



template class FullyFusedMLP<network_precision_t, 64>;

#if !FAST_COMPILATION
template class FullyFusedMLP<network_precision_t, 128>;
template class FullyFusedMLP<network_precision_t, 256>;
template class FullyFusedMLP<network_precision_t, 32>;
template class FullyFusedMLP<network_precision_t, 16>;
#endif

TCNN_NAMESPACE_END
