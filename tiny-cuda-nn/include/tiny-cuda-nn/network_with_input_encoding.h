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

/** @file   network_with_input_encoding.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  A model that includes its encoding
 */

#pragma once

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>
#include "tiny-cuda-nn/networks/fully_fused_mlp.h"


TCNN_NAMESPACE_BEGIN

template <typename T>
class NetworkWithInputEncoding : public Network<float, T> {
public:
	int trainID = 0;
	NetworkWithInputEncoding(std::shared_ptr<Encoding<T>> encoding, uint32_t n_output_dims, const json& network, const json& encoding_json, bool bSeparateInput=true, bool bFusedOutput=true) : m_encoding{encoding} {
		uint32_t alignment = network.contains("otype") && (equals_case_insensitive(network["otype"], "FullyFusedMLP") || equals_case_insensitive(network["otype"], "MegakernelMLP")) ? 16u : 8u;
		m_num_neurons = network["n_neurons"];
		encoding->set_alignment(m_num_neurons);

		json local_network_config = network;
		local_network_config["n_input_dims"] = m_encoding->num_encoded_dims();
		local_network_config["n_output_dims"] = n_output_dims;

		m_network.reset(create_network<T>(local_network_config));

		bSeparateEncodingBuffers = bSeparateInput;
		

		const json::array_t& nested = encoding_json["nested"];
		m_forward_separate_encoding_output.bSeparate = true;

		bool bMoveForward = false;
		for (size_t i = 0; i < nested.size(); ++i) {
			std::string en_type = nested[i]["otype"];
			if (en_type != "SphericalHarmonics")
				continue;


			bMoveForward = nested[i]["move_forward"];
			break;
		}

		if (m_num_neurons == 64) {
			tcnn::FullyFusedMLP<T, 64>* ff_mlp_network = dynamic_cast<tcnn::FullyFusedMLP<T, 64>*>(m_network.get());
			if (bMoveForward) {
				std::vector<uint32_t> data_offsets = m_encoding->offsets();
				ff_mlp_network->setSHEncoderOffset(data_offsets[data_offsets.size() - 2]);
			}
			else {
				ff_mlp_network->setSHEncoderOffset(128);
			}
		}
		if (m_num_neurons == 128) {
			tcnn::FullyFusedMLP<T, 128>* ff_mlp_network = dynamic_cast<tcnn::FullyFusedMLP<T, 128>*>(m_network.get());
			if (bMoveForward) {
				std::vector<uint32_t> data_offsets = m_encoding->offsets();
				ff_mlp_network->setSHEncoderOffset(data_offsets[data_offsets.size() - 2]);
			}
			else {
				ff_mlp_network->setSHEncoderOffset(128);
			}
		}

	}

	NetworkWithInputEncoding(uint32_t n_dims_to_encode, uint32_t n_output_dims, const json& encoding, const json& network, bool bSepateInput=true)
	: NetworkWithInputEncoding{std::shared_ptr<Encoding<T>>{create_encoding<T>(n_dims_to_encode, encoding)}, n_output_dims, network, encoding, bSepateInput } {
	}

	virtual ~NetworkWithInputEncoding() { }

	void inference(cudaStream_t stream, const GPUMatrix<float>& input, GPUMatrix<float>& output, GPUMatrix<float>* outTransformData = nullptr) override {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_inference_network_input.n() < batch_size) {
			allocate_inference_buffers(batch_size);
		}

		m_encoding->encode(
			stream,
			input.n(),
			{input.data(), input.m()},
			{m_inference_network_input.data(),
			m_inference_network_input.m()},
			nullptr,
			true
		);
		m_network->inference(stream, m_inference_network_input, output, outTransformData);
	}
	
	
	void inference(cudaStream_t stream, NeuralInputData<float>& input, GPUMatrix<float>& output,
					uint32_t size, int* mapping, float* bbStart, float* bbEnd, FusedOutputData out_data = FusedOutputData()) override
	{
		int32_t paddedSize = (size + 255) / 256* 256;

		bool bSeparateEncoding = bSeparateEncodingBuffers;	
		if(!bSeparateEncoding && m_inference_network_input.n() != paddedSize)
			allocate_inference_buffers(paddedSize, true);

		m_encoding->encode(
			stream,
			size,
			input,
			(bSeparateEncoding) ? &m_forward_separate_encoding_output : nullptr,
			(bSeparateEncoding) ? PitchedPtr<T>() : PitchedPtr<T>(m_inference_network_input.data(), m_inference_network_input.m()),
			nullptr,
			true,
			mapping,
			bbStart,
			bbEnd,
			trainID
		);

		NeuralInputData<T> temp;
		temp.addInput(m_inference_network_input);

		m_network->inference(stream, (bSeparateEncoding) ? m_forward_separate_encoding_output : temp, output, size, mapping, bbStart, bbEnd, out_data);
	}

	void inference_mixed_precision(cudaStream_t stream, NeuralInputData<float>& input, GPUMatrixDynamic<T>& output, bool use_inference_matrices = true, FusedOutputData data = FusedOutputData()) override {
		assert(0);	
		throw std::exception();

		//// Make sure our teporary buffers have the correct size for the given batch size
		//uint32_t batch_size = input.n();
		//if (m_inference_network_input.n() != batch_size) {
		//	allocate_inference_buffers(batch_size);
		//}

		//m_encoding->encode(stream, input.n(), {input.data(), input.m()}, {m_inference_network_input.data(), m_inference_network_input.m()}, nullptr, use_inference_matrices);
		//m_network->inference_mixed_precision(stream, m_inference_network_input, output, use_inference_matrices, outShadeData);
	}

	uint32_t num_encoded_dims() const {
		return m_encoding->num_encoded_dims();
	}

	void forward(
		cudaStream_t stream, 
		NeuralInputData<float>& input,
		uint32_t size, 
		int* mapping, 
		float* bbStart, 
		float* bbEnd,
		GPUMatrixDynamic<T>* output = nullptr, 
		bool use_inference_matrices = false, 
		bool prepare_input_gradients = false
	) override
	{
		// Make sure our teporary buffers have the correct size for the given batch size

		bool bSeparateEncoding = bSeparateEncodingBuffers && use_inference_matrices;

		if(!bSeparateEncoding){
			if(m_forward_network_input.n() != (size+255)/ 256 * 256)
				allocate_forward_buffers((size + 255) / 256 * 256, true);
		}
		
		m_encoding->encode(
			stream,
			size,
			input,
			nullptr,
			{m_forward_network_input.data(), m_forward_network_input.m()},
			prepare_input_gradients ? m_forward_encoding_forward_gradient.data() : nullptr,
			use_inference_matrices,
			mapping,
			bbStart,
			bbEnd,
			trainID
		);


		uint32_t offsetsize = (size + 257) / 256 * 256 - size;
		if(offsetsize != 0)
			CUDA_CHECK_THROW(cudaMemsetAsync(m_forward_network_input.data()+ m_forward_network_input.m() * size, 0, m_forward_network_input.m() * offsetsize* sizeof(T), stream));
			
		m_network->forward(stream, m_forward_network_input, output, use_inference_matrices, prepare_input_gradients);

		m_cached_batch_size = size;
	}
	
	void forward(cudaStream_t stream, const GPUMatrix<float>& input, GPUMatrixDynamic<T>* output = nullptr, bool use_inference_matrices = false, bool prepare_input_gradients = false) override {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_forward_network_input.n() != batch_size) {
			allocate_forward_buffers(batch_size);
		}

		m_encoding->encode(
			stream,
			input.n(),
			{input.data(), input.m()},
			{m_forward_network_input.data(), m_forward_network_input.m()},
			prepare_input_gradients ? m_forward_encoding_forward_gradient.data() : nullptr,
			use_inference_matrices
		);
		m_network->forward(stream, m_forward_network_input, output, use_inference_matrices, prepare_input_gradients);
	}

	void backward(
		cudaStream_t stream,
		NeuralInputData<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrix<float>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) override {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();


		// CRUUUNCH
		uint32_t crunch_size = (batch_size + 255) / 256 * 256;
		if (crunch_size == 0)
			return;

		const uint32_t oldSize = m_forward_network_input.cols();
		batch_size = m_forward_network_input.cols();



		if (m_backward_dL_dnetwork_input.n() != batch_size) {
			allocate_backward_buffers(batch_size);
		}

		//m_forward_network_input.set_size_const(m_forward_network_input.rows(), batch_size);
		//output.set_size_const(output.rows(), batch_size);
		//dL_doutput.set_size_const(dL_doutput.rows(), batch_size);

		GPUMatrix<T>* dL_dnetwork_input = nullptr;
		if (m_encoding->n_params() > 0 || dL_dinput) {
			dL_dnetwork_input = &m_backward_dL_dnetwork_input;
		}

		m_network->backward(stream, m_forward_network_input, output, dL_doutput, dL_dnetwork_input, use_inference_matrices, compute_param_gradients);
		if (dL_dnetwork_input) {
			m_encoding->backward(
				stream,
				m_cached_batch_size,
				{ dL_dnetwork_input->data(), dL_dnetwork_input->m() },
				dL_dinput ? m_forward_encoding_forward_gradient.data() : nullptr,
				dL_dinput ? PitchedPtr<float>{dL_dinput->data(), dL_dinput->m()} : PitchedPtr<float>{},
				input
			);
		}

		/*	m_forward_network_input.set_size_const(m_forward_network_input.rows(), oldSize);
			output.set_size_const(output.rows(), oldSize);
			dL_doutput.set_size_const(dL_doutput.rows(), oldSize);*/

	}

	void backward(
		cudaStream_t stream,
		const GPUMatrix<float>& input,
		const GPUMatrixDynamic<T>& output,
		const GPUMatrixDynamic<T>& dL_doutput,
		GPUMatrix<float>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) override {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();


		// CRUUUNCH
		uint32_t crunch_size = (batch_size+255) / 256 * 256;
		if (crunch_size == 0)
			return;

		const uint32_t oldSize = m_forward_network_input.cols();
		batch_size = m_forward_network_input.cols();
		
		

		if (m_backward_dL_dnetwork_input.n() != batch_size) {
			allocate_backward_buffers(batch_size);
		}

		//m_forward_network_input.set_size_const(m_forward_network_input.rows(), batch_size);
		//output.set_size_const(output.rows(), batch_size);
		//dL_doutput.set_size_const(dL_doutput.rows(), batch_size);

		GPUMatrix<T>* dL_dnetwork_input = nullptr;
		if (m_encoding->n_params() > 0 || dL_dinput) {
			dL_dnetwork_input = &m_backward_dL_dnetwork_input;
		}

		m_network->backward(stream, m_forward_network_input, output, dL_doutput, dL_dnetwork_input, use_inference_matrices, compute_param_gradients);
		if (dL_dnetwork_input) {
			m_encoding->backward(
				stream,
				m_cached_batch_size,
				{dL_dnetwork_input->data(), dL_dnetwork_input->m()},
				dL_dinput ? m_forward_encoding_forward_gradient.data() : nullptr,
				dL_dinput ? PitchedPtr<float>{dL_dinput->data(), dL_dinput->m()} : PitchedPtr<float>{},
				{input.data(), input.m()}
			);
		}

	/*	m_forward_network_input.set_size_const(m_forward_network_input.rows(), oldSize);
		output.set_size_const(output.rows(), oldSize);
		dL_doutput.set_size_const(dL_doutput.rows(), oldSize);*/

	}

	void initialize_params(pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		m_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_network->n_params();

		m_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_encoding->n_params();
	}

	size_t n_params() const override {
		return m_encoding->n_params() + m_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return m_network->padded_output_width();
	}


	uint32_t output_width() const override {
		return m_network->output_width();
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		return m_network->layer_sizes();
	}

	uint32_t width(uint32_t layer) const override {
		return layer == 0 ? m_encoding->num_encoded_dims() : m_network->width(layer - 1);
	}

	uint32_t num_forward_activations() const override {
		return m_network->num_forward_activations() + 1;
	}

	const T* forward_activations(uint32_t layer) const override {
		return layer == 0 ? m_forward_network_input.data() : m_network->forward_activations(layer - 1);
	}

	const std::shared_ptr<Encoding<T>>& encoding() const {
		return m_encoding.get();
	}
	void* getDebugInput() override {
		return &m_forward_network_input;
	}

	Network<T>* getNetwork() {
		return m_network.get();
	}

private:
	void allocate_inference_buffers(uint32_t s, bool bAllocate=true) {
		m_inference_network_input.set_size(m_encoding->num_encoded_dims(), s);

		if(bAllocate)
			GPUMatrixBase::allocate_shared_memory(
				m_inference_buffer,
				{
					&m_inference_network_input,
				}
			);
	}

	void allocate_forward_buffers(uint32_t s, bool bAllocate=true) {
		m_forward_network_input.set_size(m_encoding->num_encoded_dims(), s);
		m_forward_encoding_forward_gradient.set_size(m_encoding->num_forward_gradient_dims(), s);

		if(bAllocate)
			GPUMatrixBase::allocate_shared_memory(
				m_forward_buffer,
				{
					&m_forward_network_input,
					&m_forward_encoding_forward_gradient,
				}
			);
	}

	

	void allocate_backward_buffers(uint32_t s, bool bAllocate=true) {
		m_backward_dL_dnetwork_input.set_size(m_encoding->num_encoded_dims(), s);

		if(bAllocate)
			GPUMatrixBase::allocate_shared_memory(
				m_backward_buffer,
				{
					&m_backward_dL_dnetwork_input,
				}
			);
	}

	bool bSeparateEncodingBuffers = true;
	bool bFusedOutput = true;

	
private:
	std::shared_ptr<Network<T>> m_network;
	std::shared_ptr<Encoding<T>> m_encoding;

	// Temporary buffers to hold inference data
	GPUMemory<char> m_inference_buffer;
	GPUMatrix<T> m_inference_network_input;

	// Temporary buffers to hold forward data
	GPUMemory<char> m_forward_buffer;
	
	GPUMatrix<T> m_forward_network_input;
	std::vector<GPUMatrix<T>> m_forward_network_input_separate;
	GPUMatrix<float> m_forward_encoding_forward_gradient; // Only needed when computing input gradients

	NeuralInputData<T> m_forward_separate_encoding_output;


	// Temporary buffers to hold backward data
	GPUMemory<char> m_backward_buffer;
	GPUMatrix<T> m_backward_dL_dnetwork_input;

	uint32_t m_cached_batch_size;
	uint32_t m_num_neurons;
};

TCNN_NAMESPACE_END
