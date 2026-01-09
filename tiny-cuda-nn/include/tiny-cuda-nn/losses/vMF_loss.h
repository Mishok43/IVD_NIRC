#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>


TCNN_NAMESPACE_BEGIN


#define VMF_COUNT 1
#define PI 3.14159265359
#define ONE_OVER_PI 0.31830988618


__device__ float clampGradient(float gradient) {
	return clamp(gradient, -1000000.0f, 1000000.0f);

}


template <typename T>
__global__ void vMF_loss(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t dims,
	const float loss_scale,
	const T* __restrict__ predictions,
	const float* __restrict__ targets,
	float* __restrict__ values,
	T* __restrict__ gradients,
	const float* __restrict__ data_pdf = nullptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t target_idx = 4 * i;
	const uint32_t pred_idx = i * stride;

	const float w1 = targets[target_idx + 0];
	const float w2 = targets[target_idx + 1];
	const float w3 = targets[target_idx + 2];
	const float L = targets[target_idx + 3];

	float prediction = 0.0f;

	float mu1;
	float mu2;
	float mu3;
	float kappa;

	float exp2k;
	float C2;
	float rmu_length;
	float dot_prod;
	float pdf2_prob;
	float pdf2_prob_nk;
	float A;

	float pdf2_prob_cache[VMF_COUNT];
	float da[VMF_COUNT];

	for (int j = 0; j < VMF_COUNT; j++)
	{
		int joffset = j * 5;
		mu1 = predictions[pred_idx + joffset + 0];
		mu2 = predictions[pred_idx + joffset + 1];
		mu3 = predictions[pred_idx + joffset + 2];
		kappa = predictions[pred_idx + joffset + 3];
		A = predictions[pred_idx + joffset + 4];

		exp2k = __expf(-2.0f * kappa);
		C2 = ONE_OVER_PI / (1.0f - exp2k+0.001); // expf - выше качество

		rmu_length = rsqrtf(mu1 * mu1 + mu2 * mu2 + mu3 * mu3+0.001);
		dot_prod = (mu1 * w1 + mu2 * w2 + mu3 * w3) * rmu_length - 1.0f;

		pdf2_prob_nk = C2 * __expf(kappa * dot_prod) * A;
		pdf2_prob = pdf2_prob_nk * kappa;

		prediction += 0.5 * pdf2_prob;

		// wolfram
		// d(A*p*k*exp(k*(m * w + o * q + r * s - 1))/(2*pi*(1-exp(2*k))) + G + F - L)^2 /dm

		pdf2_prob_cache[j] = pdf2_prob;
		da[j] = pdf2_prob * (dot_prod + 2 * exp2k / (1 - exp2k)) + pdf2_prob_nk;
	}


	const float difference = prediction - L;
	
	const uint32_t n_total = n_elements / stride * dims;
	const float prediction_sq_plus_epsilon = prediction * prediction + 0.01f;
	float prediction_sq_plus_epsilon_inv = 1.0/prediction_sq_plus_epsilon;
	const float n_total_inv = (1.0/ n_total)*prediction_sq_plus_epsilon_inv;

	
	const float difference2 = (difference * difference) * n_total_inv;



	

	for (int j = 0; j < VMF_COUNT; j++)
	{
		int joffset = j * 5;
		pdf2_prob = pdf2_prob_cache[j];
		kappa = predictions[pred_idx + joffset + 3];
		A = predictions[pred_idx + joffset + 4];

		float ka = difference * loss_scale;
		pdf2_prob *= ka;

		gradients[pred_idx + joffset + 0] = (T)clampGradient(pdf2_prob * kappa * w1* n_total_inv);
		gradients[pred_idx + joffset + 1] = (T)clampGradient(pdf2_prob * kappa * w2* n_total_inv);
		gradients[pred_idx + joffset + 2] = (T)clampGradient(pdf2_prob * kappa * w3* n_total_inv);
		gradients[pred_idx + joffset + 3] = (T)clampGradient(ka * da[j]* n_total_inv );
		gradients[pred_idx + joffset + 4] = (T)clampGradient(pdf2_prob / A * n_total_inv);
# if 0
		gradients[pred_idx + joffset + 0] = (T)(0);
		gradients[pred_idx + joffset + 1] = (T)(0);
		gradients[pred_idx + joffset + 2] = (T)(0);
		gradients[pred_idx + joffset + 3] = (T)(0);
		gradients[pred_idx + joffset + 4] = (T)(0);
#endif
		
		values[pred_idx + joffset + 0] = difference2;
		values[pred_idx + joffset + 1] = 0;
		values[pred_idx + joffset + 2] = 0;
		values[pred_idx + joffset + 3] = 0;
		values[pred_idx + joffset + 4] = 0;



	}

	for (int j = dims; j < stride; j++) {
		values[pred_idx + j] = 0;
		gradients[pred_idx + j] = 0;
	}

}


template <typename T>
class VMFLoss : public Loss<T> {
public:
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

		if (target.m() != 4) {
			throw std::runtime_error(std::string("Target does not have appropriate dimensions ") + std::to_string(target.m()) + "!=" + std::to_string(4));
		}

		linear_kernel(vMF_loss<T>, 0, stream,
			prediction.n(),
			stride,
			dims,
			loss_scale,
			prediction.data(),
			target.data(),
			values.data(),
			gradients.data(),
			data_pdf ? data_pdf->data() : nullptr
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
		GPUMatrix<float>* sao_gradients=nullptr) const override 
	{
		throw std::runtime_error(std::string("NOT SUPPORTED STUFF"));
	}

	void update_hyperparams(const json& params) override { }
};

TCNN_NAMESPACE_END
