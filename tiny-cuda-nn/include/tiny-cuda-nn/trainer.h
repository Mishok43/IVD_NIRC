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

/** @file   trainer.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Class that performs training of a differentiable cuda object, given an optimizer and a loss.
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>


#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/loss.h>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <tiny-cuda-nn/gpu_memory_json.h>

#include <tiny-cuda-nn/cnpy.h>
#include <iostream>
#include <random>
#include <cusparse.h>
#include <cublas_v2.h>
#include <tiny-cuda-nn/helper_cuda.h>

#pragma comment(lib, "cuda")
#pragma comment(lib, "cublas")
#pragma comment(lib, "cusparse")

TCNN_NAMESPACE_BEGIN


template<typename T>
__global__ void gradient_gathering(const uint32_t num_elements, const uint32_t dims, const uint32_t stride, const float* __restrict__ sao_gradients, T* __restrict__ gradients, const float gradients_blending=0.0f)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	const uint32_t num_gradients = num_elements / dims;

	const uint32_t intra_elem_idx = i / num_gradients;
	const uint32_t inter_elem_idx = i % num_gradients;

	float old_grad = gradients[inter_elem_idx * stride + intra_elem_idx];
	float k = 1.0 - gradients_blending;

	gradients[inter_elem_idx * stride + intra_elem_idx] = (T)(sao_gradients[i]*k+gradients_blending*old_grad);
}


__global__ void r1_div_x(float* r1, float* r0, float* b) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid == 0) {
		b[0] = r1[0] / r0[0];
	}
}

__global__ void a_minus(float* a, float* na) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid == 0) {
		na[0] = -(a[0]);
	}
}

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class Trainer : public ObjectWithMutableHyperparams {
public:
	Trainer(std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model, std::shared_ptr<Optimizer<PARAMS_T>> optimizer, std::shared_ptr<Loss<COMPUTE_T>> loss, uint32_t seed = 1337, float perturbation_sigma = 0)
	: m_model{model}, m_optimizer{optimizer}, m_loss{loss}, m_perturbation_sigma{perturbation_sigma} {
		std::seed_seq seq{seed};
		std::vector<uint32_t> seeds(2);
		seq.generate(std::begin(seeds), std::end(seeds));
		m_rng = pcg32{seeds.front()};
		initialize_params();
	}

	virtual ~Trainer() {}

	void set_loss(std::shared_ptr<Loss<COMPUTE_T>> loss) {
		if (!loss) {
			throw std::runtime_error{"Trainer: may not set loss to nullptr"};
		}
		m_loss = loss;
	}

	void initialize_params() {
		training_step_count = 0;
		size_t n_params = m_model->n_params();
#ifdef TCNN_VERBOSE_MEMORY_ALLOCS
		std::cout << "Trainer: Initializing " << n_params << " params and resetting training." << std::endl;
#endif

		m_params_buffer.resize(sizeof(PARAMS_T) * n_params * 3 + sizeof(float) * n_params * 1);
		m_params_buffer.memset(0);

		m_params_full_precision = (float*)(m_params_buffer.data());
		m_params                = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
		m_params_backward       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params);
		m_param_gradients       = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params * 2);

		// Allocate auxiliary optimizer buffers
		m_optimizer->allocate(m_model);

		// Use the optimizer's custom params for inference, if they exist.
		m_params_inference = m_optimizer->custom_weights();
		if (m_params_inference == nullptr) {
			m_params_inference = m_params;
		}

		m_model->initialize_params(
			m_rng,
			m_params_full_precision,
			m_params,
			m_params_inference,
			m_params_backward,
			m_param_gradients
		);

		// initialize_params is only expected to initialize m_params_full_precision. Cast and copy these over!
		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params=m_params] __device__ (size_t i) {
			params[i] = (PARAMS_T)params_fp[i];
		});
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void allocate_training_buffers(uint32_t padded_output_width, uint32_t batch_size, bool bAllocate=true) {
		m_perturbation.set_size(padded_output_width, batch_size);
		m_perturbed_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_gradient_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_tmp.set_size(padded_output_width, batch_size);
		m_sao_gradients.set_size(batch_size, m_model->output_width());

		if(bAllocate)
		GPUMatrixBase::allocate_shared_memory(
			m_training_buffer,
			{
				&m_perturbation,
				&m_perturbed_training_prediction_tmp,
				&m_training_prediction_tmp,
				&m_training_loss_gradient_tmp,
				&m_training_loss_tmp,
				& m_sao_gradients
			}
		);
	}

	void set_training_buffers(uint32_t padded_output_width, uint32_t batch_size) {
		m_perturbation.set_size(padded_output_width, batch_size);
		m_perturbed_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_prediction_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_gradient_tmp.set_size(padded_output_width, batch_size);
		m_training_loss_tmp.set_size(padded_output_width, batch_size);
	}

	const GPUMatrix<COMPUTE_T>& forward(cudaStream_t stream, const GPUMatrix<T>& input) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_training_prediction_tmp.n() != batch_size) {
			allocate_training_buffers(m_model->padded_output_width(), batch_size);
		}

		m_model->forward(stream, input, &m_training_prediction_tmp);
		return m_training_prediction_tmp;
	}

	const GPUMatrix<COMPUTE_T>& forward(cudaStream_t stream, NeuralInputData<T>& input,
										uint32_t padded_size, int* mapping, float* bbStart, float* bbEnd)
	{
		// Make sure our teporary buffers have the correct size for the given batch size
		if(m_training_prediction_tmp.n() != (padded_size+255)/256*256)
		allocate_training_buffers(m_model->padded_output_width(), (padded_size + 255) / 256 * 256, true);

		reinterpret_cast<NetworkWithInputEncoding<T>*>(m_model.get())->trainID = training_step_count;
		m_model->forward(stream, input, padded_size, mapping, bbStart, bbEnd, &m_training_prediction_tmp);
		return m_training_prediction_tmp;
	}

	const GPUMatrix<COMPUTE_T>& forward(const GPUMatrix<T>& input) {
		return forward(nullptr, input);
	}

	const GPUMatrix<float>& evaluate_loss(cudaStream_t stream, const float loss_scale, const GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr, float* loss_value = nullptr) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = target.n();
		if (m_training_prediction_tmp.n() != batch_size) {
			throw std::runtime_error{"Trainer: you must call `forward` before calling `evaluate_loss`"};
		}

		if (m_perturbation_sigma > 0) {
			const uint32_t n_elements = m_perturbation.n_elements();
			generate_random_logistic<float>(stream, m_rng, n_elements, m_perturbation.data(), 0.0f, m_perturbation_sigma);
			add<<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_training_prediction_tmp.data(), m_perturbation.data(), m_perturbed_training_prediction_tmp.data());
		}

		auto& loss_input = m_perturbation_sigma > 0 ? m_perturbed_training_prediction_tmp : m_training_prediction_tmp;

		m_loss->evaluate(
			stream,
			m_model->padded_output_width(),
			m_model->output_width(),
			loss_scale,
			loss_input,
			target,
			m_training_loss_tmp,
			m_training_loss_gradient_tmp,
			data_pdf

		);

		if (loss_value) {
			*loss_value = reduce_sum(m_training_loss_tmp.data(), m_training_loss_tmp.n_elements(), stream);
		}

		return m_training_loss_tmp;
	}

	const GPUMatrix<float>& evaluate_loss(
		cudaStream_t stream,
		const float loss_scale,
		GPUMatrix<float>& target,
		uint32_t batch_size, int* mapping,
		const GPUMatrix<float>* data_pdf = nullptr,
		float* loss_value = nullptr,
		GPUMatrix<float>* sao_gradients = nullptr
	)
	{
		// Make sure our teporary buffers have the correct size for the given batch size
		/*if (m_training_prediction_tmp.n() <= batch_size) {
			throw std::runtime_error{"Trainer: you must call `forward` before calling `evaluate_loss`"};
		}*/

		if (m_perturbation_sigma > 0) {
			const uint32_t n_elements = m_perturbation.n_elements();
			generate_random_logistic<float>(stream, m_rng, n_elements, m_perturbation.data(), 0.0f, m_perturbation_sigma);
			add<<<n_blocks_linear(n_elements), n_threads_linear, 0, stream>>>(n_elements, m_training_prediction_tmp.data(), m_perturbation.data(), m_perturbed_training_prediction_tmp.data());
		}

		auto& loss_input = m_perturbation_sigma > 0 ? m_perturbed_training_prediction_tmp : m_training_prediction_tmp;

		m_loss->evaluate(
			stream,
			m_model->padded_output_width(),
			m_model->output_width(),
			batch_size, mapping,
			loss_scale,
			loss_input,
			target,
			m_training_loss_tmp,
			m_training_loss_gradient_tmp,
			data_pdf,
			sao_gradients
		);

		if (loss_value) {
			*loss_value = reduce_sum(m_training_loss_tmp.data(), m_training_loss_tmp.n_elements(), stream);
		}

		return m_training_loss_tmp;
	}

	void evaluate_loss(const float loss_scale, GPUMatrix<float>& target, const GPUMatrix<float>* data_pdf = nullptr, float* loss_value = nullptr) {
		evaluate_loss(nullptr, loss_scale, target, data_pdf, loss_value);
	}

	void backward(cudaStream_t stream, NeuralInputData<T>& input) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		/*	if (m_training_prediction_tmp.n() != batch_size) {
				throw std::runtime_error{"Trainer: you must call `forward` and `evaluate_loss` before calling `backward`"};
			}*/
		reinterpret_cast<NetworkWithInputEncoding<T>*>(m_model.get())->trainID = training_step_count;
		m_model->backward(stream, input, m_training_prediction_tmp, m_training_loss_gradient_tmp);
	}

	void backward(cudaStream_t stream, const GPUMatrix<T>& input) {
		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
	/*	if (m_training_prediction_tmp.n() != batch_size) {
			throw std::runtime_error{"Trainer: you must call `forward` and `evaluate_loss` before calling `backward`"};
		}*/
		reinterpret_cast<NetworkWithInputEncoding<T>*>(m_model.get())->trainID = training_step_count;
		m_model->backward(stream, input, m_training_prediction_tmp, m_training_loss_gradient_tmp);
	}
	void backward(NeuralInputData<T>& input) {
		backward(nullptr, input);

	}



	void backward(const GPUMatrix<T>& input) {
		backward(nullptr, input);
	}

	void optimizer_step(cudaStream_t stream, float loss_scale) {
		m_optimizer->step(stream, loss_scale, m_params_full_precision, m_params, m_param_gradients);
	}

	void optimizer_step(float loss_scale) {
		optimizer_step(nullptr, loss_scale);
	}

	void training_step(
		cudaStream_t stream,
		 GPUMatrix<T>& input,
		GPUMatrix<float>& target,
		float* loss_value = nullptr,
		const GPUMatrix<float>* data_pdf = nullptr
	) {
		if (input.n() != target.n()) {
			throw std::runtime_error(std::string("Input and target don't have matching batch size ") + std::to_string(input.n()) + "!=" + std::to_string(target.n()));
		}

		// Because of VMF with different size of training data and model output we need to comment it
		/*if (target.m() != m_model->output_width()) {
			throw std::runtime_error(std::string("Target does not have the correct number of dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(m_model->output_width()));
		}*/

		// Make sure our teporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		bool did_allocate = false;
		if (m_training_prediction_tmp.n() != batch_size) {
			allocate_training_buffers(m_model->padded_output_width(), batch_size);
			did_allocate = true;
		}


		static const float loss_scale = 128;

		m_graph.capture_and_execute(stream, did_allocate, [&]() {
			forward(stream, input);
			evaluate_loss(stream, loss_scale, target, data_pdf);
			//backward(stream, input);
		});

		//optimizer_step(stream, loss_scale);

		if (loss_value) {
			*loss_value = reduce_sum(m_training_loss_tmp.data(), m_training_loss_tmp.n_elements(), stream);
		}
	}


	void allocate_diffusion_buffers(uint32_t temp_buffer_size, uint32_t num_vertices, bool bAllocate = true) {
		m_diffusion_temp_buffer.set_size(temp_buffer_size, 1);
		m_vec_ax.set_size(num_vertices, 1);
		m_vec_p.set_size(num_vertices, 1);
		m_vec_x.set_size(num_vertices, 1);
		m_scalars.set_size(10, 1);

		if (bAllocate)
			GPUMatrixBase::allocate_shared_memory(
				m_diffusion_buffer,
				{
					&m_diffusion_temp_buffer,
					&m_vec_ax,
					&m_vec_p,
					&m_vec_x,
					&m_scalars
				}
		);
	}

	void diffuse_gradients(cudaStream_t stream, int numVertices, uint32_t* numNonZeroElements,
		float* laplace_csrValA, int* laplace_csrRowPtrA, int* laplace_csrColIndA,
		GPUMatrix<float> sao_gradients, cudaEvent_t* eventLaplaceFinishMatrixBuild,
		cudaEvent_t* eventLaplaceReceivedNNZE, uint32_t denoising_max_iter=1024) {
		const uint32_t grad_dim = sao_gradients.cols();

		if (!bDiffusionIsSet) {
			/* Get handle to the CUBLAS context */
			checkCudaErrors(cublasCreate(&cublasHandle));
			checkCudaErrors(cusparseCreate(&cusparseHandle));

			checkCudaErrors(cusparseSetStream(cusparseHandle, stream));
			checkCudaErrors(cublasSetStream(cublasHandle, stream));

			bDiffusionIsSet = true;
			checkCudaErrors(cudaStreamCreate(&streamForGraph));

			const uint32_t CUBLAS_MEMORY_SIZE = 10000000;

			CUDA_CHECK_THROW(cudaMalloc(&cublas_memory, CUBLAS_MEMORY_SIZE));

			cublasSetWorkspace(cublasHandle, cublas_memory, CUBLAS_MEMORY_SIZE);
		}


		float alpha, beta, alpham1;
		alpha = 1.0;
		alpham1 = -1.0;
		beta = 0.0;
		float r1;

		cusparseSpMatDescr_t matA = NULL;
		checkCudaErrors(cusparseCreateCsr(&matA, numVertices, numVertices, *numNonZeroElements, laplace_csrRowPtrA, laplace_csrColIndA, laplace_csrValA, CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

#if 0
		{
			std::vector<float> cpu_laplace_csrValA(*numNonZeroElements);
			std::vector<int> cpu_laplace_csrColIndA(*numNonZeroElements);
			std::vector<int> cpu_laplace_row_offset(numVertices + 1);


			CUDA_CHECK_THROW(cudaMemcpy(cpu_laplace_csrValA.data(), laplace_csrValA, (*numNonZeroElements) * 4, cudaMemcpyDeviceToHost));
			CUDA_CHECK_THROW(cudaMemcpy(cpu_laplace_csrColIndA.data(), laplace_csrColIndA, (*numNonZeroElements) * 4, cudaMemcpyDeviceToHost));
			CUDA_CHECK_THROW(cudaMemcpy(cpu_laplace_row_offset.data(), laplace_csrRowPtrA, (numVertices + 1) * 4, cudaMemcpyDeviceToHost));
			cudaDeviceSynchronize();

			/*cpu_matrix.resize(numVertices * numVertices);
			float* dense_matrix{ nullptr };
			CUDA_CHECK_THROW(cudaMalloc(&dense_matrix, 4*numVertices*numVertices+1000));
			cusparseDnMatDescr_t matB = NULL;
			cudaDeviceSynchronize();

			checkCudaErrors(cusparseCreateDnMat(&matB, numVertices, numVertices, numVertices, dense_matrix, CUDA_R_32F, CUSPARSE_ORDER_ROW));
			cudaDeviceSynchronize();

			size_t temp_buffer_size = 0;
			checkCudaErrors(cusparseSparseToDense_bufferSize(cusparseHandle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT,&temp_buffer_size));
			cudaDeviceSynchronize();

			void* temp_buffer{ nullptr };
			CUDA_CHECK_THROW(cudaMalloc(&temp_buffer, 4000));
			cudaDeviceSynchronize();

			checkCudaErrors(cusparseSparseToDense(cusparseHandle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &temp_buffer));
			cudaDeviceSynchronize();

			CUDA_CHECK_THROW(cudaMemcpyAsync(cpu_matrix.data(), dense_matrix, numVertices * numVertices *4, cudaMemcpyDeviceToHost, stream));

			cudaStreamSynchronize(stream);
			cudaDeviceSynchronize();*/
			// Note: Debug paths removed for privacy - uncomment and set your own paths if needed
			//cnpy::npy_save("path/to/val.npy", cpu_laplace_csrValA.data(), { cpu_laplace_csrValA.size() }, "w");
			//cnpy::npy_save("path/to/colinds.npy", cpu_laplace_csrColIndA.data(), { cpu_laplace_csrColIndA.size() }, "w");
			//cnpy::npy_save("path/to/rowoffsets.npy", cpu_laplace_row_offset.data(), { cpu_laplace_row_offset.size() }, "w");
			//exit(0);
		}
#endif

		cusparseDnVecDescr_t vecx = NULL;
		checkCudaErrors(cusparseCreateDnVec(&vecx, numVertices, sao_gradients.data(), CUDA_R_32F));

		cusparseDnVecDescr_t vecAx = NULL;
		checkCudaErrors(cusparseCreateDnVec(&vecAx, numVertices, sao_gradients.data(), CUDA_R_32F));

		cusparseDnVecDescr_t vecp = NULL;
		checkCudaErrors(cusparseCreateDnVec(&vecp, numVertices, sao_gradients.data(), CUDA_R_32F));

		bool did_allocate = false;
		/* Allocate workspace for cuSPARSE */
		size_t bufferSize = 0;
		checkCudaErrors(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,&beta, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));

		checkCudaErrors(cusparseDestroyDnVec(vecx));
		checkCudaErrors(cusparseDestroyDnVec(vecAx));
		checkCudaErrors(cusparseDestroyDnVec(vecp));

		if (bufferSize > m_diffusion_temp_buffer.rows() || numVertices > m_vec_ax.rows())
		{
			did_allocate = true;
			allocate_diffusion_buffers(bufferSize*2, numVertices*2);
		}

		vecx = NULL;
		vecAx = NULL;
		vecp = NULL;
		checkCudaErrors(cusparseCreateDnVec(&vecx, numVertices, m_vec_x.data(), CUDA_R_32F));
		checkCudaErrors(cusparseCreateDnVec(&vecAx, numVertices, m_vec_ax.data(), CUDA_R_32F));
		checkCudaErrors(cusparseCreateDnVec(&vecp, numVertices, m_vec_p.data(), CUDA_R_32F));

		float* d_r1 = m_scalars.data();
		float* d_r0 = m_scalars.data()+1;
		float* d_b = m_scalars.data()+2;
		float* d_dot = m_scalars.data() + 3;
		float* d_a = m_scalars.data() + 4;
		float* d_na = m_scalars.data() + 5;

		for (uint32_t i = 0; i < grad_dim; i++) {
			const uint32_t gradients_offset = (i) * numVertices;
			/* Wrap raw data into cuSPARSE generic API objects */

			float* r = sao_gradients.data() + gradients_offset;


			// copy right hand side to x as zero-guess
			//CUDA_CHECK_THROW(cudaMemcpyAsync(m_vec_x.data(), r, numVertices * 4, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_vec_x.data(), 0, numVertices * 4, stream));

			/* Begin CG */
			// Ax = A*x
			checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));
			checkCudaErrors(cusparseSpMV(
				cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
				&beta, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, m_diffusion_temp_buffer.data()));

			// r = r-1*Ax
			checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));
			checkCudaErrors(cublasSaxpy(cublasHandle, numVertices, &alpham1, m_vec_ax.data(), 1, r, 1));

			// error = r x r
			checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));
			checkCudaErrors(cublasSdot(cublasHandle, numVertices, r, 1, r, 1, d_r1));
			CUDA_CHECK_THROW(cudaMemcpyAsync((float*)&r1, d_r1, sizeof(float), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			const float tol = 1e-5f;
			const int max_iter = denoising_max_iter;

			int k = 1;

			bool bGraphCreated = false;

			checkCudaErrors(cublasScopy(cublasHandle, numVertices, r, 1, m_vec_p.data(), 1));
			while (r1 > tol * tol && k <= max_iter)
			//while (k <= max_iter)

			{
				if (k == 1 && 1) {
					cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
					//// Ax = A*p
					checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));
					checkCudaErrors(cusparseSpMV(
						cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
						vecp, &beta, vecAx, CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, m_diffusion_temp_buffer.data()));

					//// dot = p*(A*p)
					checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));
					checkCudaErrors(cublasSdot(cublasHandle, numVertices, m_vec_p.data(), 1, m_vec_ax.data(), 1, d_dot));

					r1_div_x << <1, 1, 0, stream >> > (d_r1, d_dot, d_a);
					//// x=x+p*a
					checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));
					checkCudaErrors(cublasSaxpy(cublasHandle, numVertices, d_a, m_vec_p.data(), 1, m_vec_x.data(), 1));

					a_minus << <1, 1, 0, stream >> > (d_a, d_na);
					//// r=r-a*(A*p)
					checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));
					checkCudaErrors(cublasSaxpy(cublasHandle, numVertices, d_na, m_vec_ax.data(), 1, r, 1));
					CUDA_CHECK_THROW(cudaMemcpyAsync(d_r0, d_r1, sizeof(float), cudaMemcpyDeviceToDevice, stream));
					//// r1=r x r
					checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));
					cublasSdot(cublasHandle, numVertices, r, 1, r, 1, d_r1);

					r1_div_x << <1, 1, 0, stream >> > (d_r1, d_r0, d_b);
					//// p=b*p
					checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE));
					checkCudaErrors(cublasSscal(cublasHandle, numVertices, d_b, m_vec_p.data(), 1));
					//// p=r+p
					checkCudaErrors(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));
					checkCudaErrors(cublasSaxpy(cublasHandle, numVertices, &alpha, r, 1, m_vec_p.data(), 1));
					CUDA_CHECK_THROW(cudaMemcpyAsync((float*)&r1, d_r1, sizeof(float), cudaMemcpyDeviceToHost, stream));

					cudaStreamEndCapture(stream, &graph);
					cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

					bGraphCreated = true;
				}

				if (bGraphCreated) {
					cudaGraphLaunch(instance, stream);
				}
				CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

				did_allocate = false;



				k++;
			}

			printf("gradient_dim = %1d, last iteration = %3d, residual = %e\n", i,  k, sqrt(r1));
			// copy final x as resulting gradient
			CUDA_CHECK_THROW(cudaMemcpyAsync(r, m_vec_x.data(), numVertices * 4, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));

			if (bGraphCreated) {
				checkCudaErrors(cudaGraphExecDestroy(instance));
				checkCudaErrors(cudaGraphDestroy(graph));
			}


		}

		checkCudaErrors(cusparseDestroySpMat(matA));
		checkCudaErrors(cusparseDestroyDnVec(vecx));
		checkCudaErrors(cusparseDestroyDnVec(vecAx));
		checkCudaErrors(cusparseDestroyDnVec(vecp));
	}




	void training_step(
		cudaStream_t stream,
		NeuralInputData<T>& input,
		GPUMatrix<float>& target,
		uint32_t batch_size, int* mapping, float* bbStart, float* bbEnd,
		bool bGraph,
		float* loss_value = nullptr,
		const GPUMatrix<float>* data_pdf = nullptr,
		bool bTrain = true,
		bool bGradientDiffusion = false,
		float *laplace_csrValA = nullptr,
		int *laplace_csrRowPtrA = nullptr,
		int *laplace_csrColIndA = nullptr,
		uint32_t* numNonZeroElements=nullptr,
		cudaEvent_t* eventLaplaceFinishMatrixBuild = nullptr,
		cudaEvent_t* eventLaplaceReceivedNNZE = nullptr,
		uint32_t denoising_max_iter=1000,
		float diffusion_gradients_blending=0.0f
	) {
		if (!mapping && input.n() != target.n()) {
			throw std::runtime_error(std::string("Input and target don't have matching size") + std::to_string(input.n()) + "!=" + std::to_string(target.n()));
		}

		const uint32_t paddedSize = (batch_size+255) / 256 * 256;
		const uint32_t oldInputSize = input.n();
		input.set_size(input.m(), paddedSize);
		target.set_size(target.m(), paddedSize);
			// Because of VMF with different size of training data and model output we need to comment it
		/*if (target.m() != m_model->output_width()) {
			throw std::runtime_error(std::string("Target does not have the correct number of dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(m_model->output_width()));
		}*/

		// Make sure our teporary buffers have the correct size for the given batch size
		bool did_allocate = m_training_prediction_tmp.n() != paddedSize;
		if(did_allocate)
		allocate_training_buffers(m_model->padded_output_width(), paddedSize, did_allocate);

		static const float loss_scale = 128;
		if (!bGradientDiffusion) {
			if (bGraph) {
				m_graph.capture_and_execute(stream, did_allocate, [&]() {
					forward(stream, input, batch_size, mapping, bbStart, bbEnd);
					evaluate_loss(stream, loss_scale, target, batch_size, mapping, data_pdf, nullptr, (bGradientDiffusion) ? &m_sao_gradients : nullptr);
					backward(stream, input);
				});
			}
			else {
				forward(stream, input, batch_size, mapping, bbStart, bbEnd);
				evaluate_loss(stream, loss_scale, target, batch_size, mapping, data_pdf, nullptr, (bGradientDiffusion) ? &m_sao_gradients : nullptr);
				backward(stream, input);
			}
		}
		else {
			forward(stream, input, batch_size, mapping, bbStart, bbEnd);
			evaluate_loss(stream, loss_scale, target, batch_size, mapping, data_pdf, nullptr, (bGradientDiffusion) ? &m_sao_gradients : nullptr);

			CUDA_CHECK_THROW(cudaEventSynchronize(*eventLaplaceReceivedNNZE));
			CUDA_CHECK_THROW(cudaStreamWaitEvent(stream, *eventLaplaceFinishMatrixBuild));
			diffuse_gradients(stream, batch_size, numNonZeroElements, laplace_csrValA, laplace_csrRowPtrA, laplace_csrColIndA, m_sao_gradients, eventLaplaceFinishMatrixBuild, eventLaplaceReceivedNNZE, denoising_max_iter);
			tcnn::linear_kernel(gradient_gathering<COMPUTE_T>, 0, stream, batch_size * m_sao_gradients.cols(), m_sao_gradients.cols(), m_model->padded_output_width(), m_sao_gradients.data(), m_training_loss_gradient_tmp.data(), diffusion_gradients_blending);
			backward(stream, input);
		}


		if(bTrain)
			optimizer_step(stream, loss_scale);

		input.set_size(input.m(), oldInputSize);
		target.set_size(target.m(), oldInputSize);
		if (loss_value) {
			reduce_sum_async(m_training_loss_tmp.data(), m_training_loss_tmp.n_elements(), stream, loss_value);
		}

		training_step_count += 1;

		/*if (isnan(*loss_value)) {
			std::cout << "NNNAANAN";
		}*/

	/*	input.set_size(input.m(), batch_size);
		target.set_size(target.m(), batch_size);*/
	}

	void training_step(
		 GPUMatrix<T>& input,
		const GPUMatrix<float>& target,
		float* loss_value = nullptr,
		const GPUMatrix<float>* data_pdf = nullptr
	) {
		training_step(nullptr, input, target, loss_value, data_pdf);
	}

	void update_hyperparams(const json& params) override {
		m_optimizer->update_hyperparams(params.value("optimizer", json::object()));
		m_loss->update_hyperparams(params.value("loss", json::object()));
	}

	float* params() {
		return m_params_full_precision;
	}

	void set_params_full_precision(const float* params_cpu, size_t n_params) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set params because CPU buffer has the wrong size."};
		}
		CUDA_CHECK_THROW(cudaMemcpy(m_params_full_precision, params_cpu, sizeof(float)*n_params, cudaMemcpyHostToDevice));

		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params_inference=m_params_inference] __device__ (size_t i) {
			params_inference[i] = (PARAMS_T)params_fp[i];
		});

		CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T)*n_params, cudaMemcpyDeviceToDevice));
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void set_params(const PARAMS_T* params_cpu, size_t n_params) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set params because CPU buffer has the wrong size."};
		}
		CUDA_CHECK_THROW(cudaMemcpy(m_params_inference, params_cpu, sizeof(PARAMS_T)*n_params, cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T)*n_params, cudaMemcpyDeviceToDevice));

		parallel_for_gpu(n_params, [params_fp=m_params_full_precision, params_inference=m_params_inference] __device__ (size_t i) {
			params_fp[i] = (float)params_inference[i];
		});

		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model() {
		return m_model;
	}

	json serialize(bool serialize_optimizer = false) {
		size_t n_params = m_model->n_params();

		json data;
		data["n_params"] = n_params;
		data["params_binary"] = gpu_memory_to_json_binary(m_params_inference, sizeof(PARAMS_T)*n_params);

		if (serialize_optimizer) {
			data["optimizer"] = m_optimizer->serialize();
		}

		return data;
	}

	void deserialize(const json& data) {
		json::binary_t params_binary = data["params_binary"];
		set_params((PARAMS_T*)params_binary.data(), params_binary.size()/sizeof(PARAMS_T));

		if (data.contains("optimizer")) {
			m_optimizer->deserialize(data["optimizer"]);
		}

		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

private:
	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> m_model;
	std::shared_ptr<Optimizer<PARAMS_T>> m_optimizer;
	std::shared_ptr<Loss<COMPUTE_T>> m_loss;

	CudaGraph m_graph;
	CudaGraph m_graph_2;

	bool bDiffusionIsSet = false;
	GPUMemory<char> m_params_buffer;

	float* m_params_full_precision = nullptr;
	PARAMS_T* m_params_inference = nullptr;
	PARAMS_T* m_params = nullptr;
	PARAMS_T* m_params_backward = nullptr; // Used for wonky things like feedback alignment
	PARAMS_T* m_param_gradients = nullptr;

	float m_perturbation_sigma;
	cusparseHandle_t cusparseHandle = 0;
	GPUMemory<char> m_training_buffer;
	GPUMemory<char> m_diffusion_buffer;
	cusparseSpMatDescr_t matA = NULL;

	GPUMatrix<float> m_perturbation;
	GPUMatrix<COMPUTE_T> m_perturbed_training_prediction_tmp;
	GPUMatrix<COMPUTE_T> m_training_prediction_tmp;
	GPUMatrix<COMPUTE_T> m_training_loss_gradient_tmp;

	GPUMatrix<float> m_training_loss_tmp;
	cublasHandle_t cublasHandle = 0;
	GPUMatrix<float> m_sao_gradients;

	GPUMatrix<char> m_diffusion_temp_buffer;
	GPUMatrix<float> m_vec_ax;
	GPUMatrix<float> m_vec_x;
	GPUMatrix<float> m_vec_p;
	GPUMatrix<float> m_scalars;
	std::vector<float> cpu_matrix;
	pcg32 m_rng;
	uint32_t training_step_count = 0;
	tcnn::CudaGraph m_diffuse_graph;
	cudaStream_t streamForGraph;
	void* cublas_memory;
	cudaGraph_t graph;
	cudaGraphExec_t instance;

};

TCNN_NAMESPACE_END
