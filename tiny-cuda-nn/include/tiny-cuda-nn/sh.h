#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/object.h>

#define M_PI                3.14159265358979323846  // pi
#define M_PI_2              1.57079632679489661923  // pi/2
#define M_PI_4              0.785398163397448309616 // pi/4
#define M_1_PI              0.318309886183790671538 // 1/pi
#define M_2_PI              0.636619772367581343076 // 2/pi
#define M_2PI               6.28318530717958647693  // 2pi

TCNN_NAMESPACE_BEGIN



struct SHTransformData {
	float nx, ny, nz;
	float ix, iy, iz;
	float vx, vy, vz;
	float roughness, pdf;
	float d_a_r_inv_pi, d_a_g_inv_pi, d_a_b_inv_pi;
	float s_a_r, s_a_g, s_a_b;
	float spec_coef;
	float only_cv;
};

__device__ void reflect(float ix, float iy, float iz, float nx, float ny, float nz, float& x, float& y, float& z) {
	float d = ix * nx + iy * ny + iz * nz;

	x = ix - 2 * nx * d;
	y = iy - 2 * ny * d;
	z = iz - 2 * nz * d;
}

template <uint32_t MIN_BAND, uint32_t NUM_BANDS, typename T>
__device__ void compute_sh_templated(float x, float y, float z, const float factors[6], float& r, float& g, float& b, T* __restrict__ data_in) {
	float xy = x * y, xz = x * z, yz = y * z, x2 = x * x, y2 = y * y, z2 = z * z, xyz = xy * z;
	float x4 = x2 * x2, y4 = y2 * y2, z4 = z2 * z2;
	float x6 = x4 * x2, y6 = y4 * y2, z6 = z4 * z2;

	float res_r = 0.0f;
	float res_g = 0.0f;
	float res_b = 0.0f;

	uint32_t offset = 0;
	auto res_compute = [&](float sh_projection, float f) {
		const float c = sh_projection * f;
		res_r += c * float(data_in[offset * 3 + 0]);
		res_g += c * float(data_in[offset * 3 + 1]);
		res_b += c * float(data_in[offset * 3 + 2]);
		offset += 1;
	};

	uint32_t fac_offset = 0;

	if (MIN_BAND == 0) {
		res_compute(0.28209479177387814f, factors[fac_offset]);
	}
	fac_offset += 1;


	if (MIN_BAND <= 1 && MIN_BAND+NUM_BANDS > 1) {
		res_compute(-0.48860251190291987f * y, factors[fac_offset]);
		res_compute(0.48860251190291987f * z, factors[fac_offset]);
		res_compute(-0.48860251190291987f * x, factors[fac_offset]);
	}
	fac_offset += 1;

	if (MIN_BAND <= 2 && MIN_BAND + NUM_BANDS > 2) {
	 	res_compute(1.0925484305920792f * xy, factors[fac_offset]);
		res_compute(-1.0925484305920792f * yz, factors[fac_offset]);
		res_compute(0.94617469575755997f * z2 - 0.31539156525251999f, factors[fac_offset]);
		res_compute(-1.0925484305920792f * xz, factors[fac_offset]);
		res_compute(0.54627421529603959f * x2 - 0.54627421529603959f * y2, factors[fac_offset]);
	}
	fac_offset += 1;

	if (MIN_BAND <= 3 && MIN_BAND + NUM_BANDS > 3) {
		res_compute(0.59004358992664352f * y * (-3.0f * x2 + y2), factors[fac_offset]);
		res_compute(0.45704579946446572f * y * (1.0f - 5.0f * z2), factors[fac_offset]);
		res_compute(0.3731763325901154f * z * (5.0f * z2 - 3.0f), factors[fac_offset]);
		res_compute(0.45704579946446572f * x * (1.0f - 5.0f * z2), factors[fac_offset]);
		res_compute(1.4453057213202769f * z * (x2 - y2), factors[fac_offset]);
		res_compute(0.59004358992664352f * x * (-x2 + 3.0f * y2), factors[fac_offset]);
	}
	fac_offset += 1;


	if (MIN_BAND <= 4 && MIN_BAND + NUM_BANDS > 4) {
		res_compute(2.5033429417967046f * xy * (x2 - y2), factors[fac_offset]);
		res_compute(1.7701307697799304f * yz * (-3.0f * x2 + y2), factors[fac_offset]);
		res_compute(0.94617469575756008f * xy * (7.0f * z2 - 1.0f), factors[fac_offset]);
		res_compute(0.66904654355728921f * yz * (3.0f - 7.0f * z2), factors[fac_offset]);
		res_compute(-3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f, factors[fac_offset]);
		res_compute(0.66904654355728921f * xz * (3.0f - 7.0f * z2), factors[fac_offset]);
		res_compute(0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f), factors[fac_offset]);
		res_compute(1.7701307697799304f * xz * (-x2 + 3.0f * y2), factors[fac_offset]);
		res_compute(-3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4, factors[fac_offset]);
	}
	fac_offset += 1;

	r = res_r;
	g = res_g;
	b = res_b;
};


__device__ SHTransformData readSHTransformData(const float* __restrict__ data, uint32_t elem_idx) {
	SHTransformData res;
	const uint32_t output_add_size = 19;

	uint32_t of = 0;
	// normal direction
	res.nx = data[elem_idx * output_add_size + of]; of++;
	res.ny = data[elem_idx * output_add_size + of]; of++;
	res.nz = data[elem_idx * output_add_size + of]; of++;

	res.ix = data[elem_idx * output_add_size + of]; of++;
	res.iy = data[elem_idx * output_add_size + of]; of++;
	res.iz = data[elem_idx * output_add_size + of]; of++;

	res.vx = data[elem_idx * output_add_size + of]; of++;
	res.vy = data[elem_idx * output_add_size + of]; of++;
	res.vz = data[elem_idx * output_add_size + of]; of++;
	res.roughness = data[elem_idx * output_add_size + of]; of++;
	res.pdf = data[elem_idx * output_add_size + of]; of++;

	res.d_a_r_inv_pi = data[elem_idx * output_add_size + of]; of++;
	res.d_a_g_inv_pi = data[elem_idx * output_add_size + of]; of++;
	res.d_a_b_inv_pi = data[elem_idx * output_add_size + of]; of++;

	res.s_a_r = data[elem_idx * output_add_size + of]; of++;
	res.s_a_g = data[elem_idx * output_add_size + of]; of++;
	res.s_a_b = data[elem_idx * output_add_size + of]; of++;
	res.spec_coef = data[elem_idx * output_add_size + of]; of++;
	res.only_cv = data[elem_idx * output_add_size + of]; of++;

	return res;
}

//template<uint32_t MIN_BAND, uint32_t NUM_BANDS, typename T, uint32_t NUM_DIFFUSE_BANDS = NUM_BANDS>
template<uint32_t MIN_BAND, uint32_t NUM_BANDS, typename T, uint32_t NUM_DIFFUSE_BANDS = (MIN_BAND + NUM_BANDS <= 3) ? NUM_BANDS : 3 - MIN_BAND>
__device__ void compute_sh_cv_templated(T* __restrict__ data_in, SHTransformData sh, float& res_r, float& res_g, float& res_b, Const4FloatArray user_band_weights) {

	// FIX IT!!!
	const float lambd_consts[6] = { M_PI*user_band_weights.a0, (M_PI * (2.0 / 3.0))*user_band_weights.a1, (M_PI / 4.0)*user_band_weights.a2,  0.0, -M_PI*(1.0/24.0), 0.0f};

	//static const float lambd_consts[6] = { M_PI, M_PI * (2.0 / 3.0), M_PI / 4.0,  1.0, -M_PI*(1.0/24.0), 1.0f};


	const float identity_consts[6] = { 1.0f*user_band_weights.a0, 1.0f*user_band_weights.a1, 1.0f*user_band_weights.a2, 1.0f, 1.0f, 1.0f};

	
	float pdf_inv = 1.0 / sh.pdf;

	float cv_diff_r = 0.0f;
	float cv_diff_g = 0.0f;
	float cv_diff_b = 0.0f;

	// Lambert BRDF
	compute_sh_templated<MIN_BAND, NUM_DIFFUSE_BANDS>(sh.nx, sh.ny, sh.nz, lambd_consts, cv_diff_r, cv_diff_g, cv_diff_b, data_in);

	float cv_dir_r;
	float cv_dir_g;
	float cv_dir_b;

	float cos_dir_n = clamp(sh.nx * sh.ix + sh.ny * sh.iy + sh.nz * sh.iz, 0.0f, 1.0f);
	compute_sh_templated<MIN_BAND, NUM_DIFFUSE_BANDS>(sh.ix, sh.iy, sh.iz, identity_consts, cv_dir_r, cv_dir_g, cv_dir_b, data_in);

	float cv_dir_r_sp = cv_dir_r;
	float cv_dir_g_sp = cv_dir_g;
	float cv_dir_b_sp = cv_dir_b;

	float fa = cos_dir_n * pdf_inv;

	float mc_residual_r = cv_dir_r * fa;
	float mc_residual_g = cv_dir_g * fa;
	float mc_residual_b = cv_dir_b * fa;


#if 0
	float rx; float ry; float rz;
	float m2 = roughness * roughness;
	float sharpness = 2.0 / m2;

	reflect(-vx, -vy, -vz, nx, ny, nz, rx, ry, rz);

	float v_dot_n = nx * vx + ny * vy + nz * vz;
	float r_dot_n = nx * rx + ny * ry + nz * rz;
	float r_dot_i = ix * rx + iy * ry + iz * rz;


	sharpness = sharpness;
	sharpness /= (4.0 * max(v_dot_n, 0.0001f));


	float norma = sharpness / (2 * M_PI * (1 - exp(-2 * sharpness)));



	float k2_inv = 1.0 / (2.0 * sharpness);

	auto att_sh = [k2_inv](int l) {
		return exp(-l * (l + 1) * k2_inv);
	};

	const float specular_consts[3] = { att_sh(0),att_sh(1),att_sh(2) };

	float cv_spec_r;
	float cv_spec_g;
	float cv_spec_b;

	compute_sh_templated<2>(rx, ry, rz, specular_consts, cv_spec_r, cv_spec_g, cv_spec_b, act_shmem + shmem_offset + li_i * elem_offset);




	// Parameters needed for the visibility

	float nDotL = saturate(r_dot_n);
	float nDotV = saturate(v_dot_n);


	float hx = rx + vx;
	float hy = ry + vy;
	float hz = rz + vz;

	float hl = sqrt(hx * hx + hy * hy + hz * hz);

	hx /= hl;
	hy /= hl;
	hz /= hl;


	auto GGX_V1 = [](float m2, float nDotX)
	{
		return 1.0f / (nDotX + sqrt(m2 + (1 - m2) * nDotX * nDotX));
	};



	float vis_term = GGX_V1(m2, nDotL) * GGX_V1(m2, nDotV);
	float r_dot_h = rx * hx + ry * hy + rz * hz;

	// Fresnel evaluated at the center of our warped BRDF lobe
	float powTerm = pow((1.0f - saturate(r_dot_h)), 5);

	float fresnel_r = s_a_r + (1.0f - s_a_r) * powTerm;
	float fresnel_g = s_a_g + (1.0f - s_a_g) * powTerm;
	float fresnel_b = s_a_b + (1.0f - s_a_b) * powTerm;

	float cos_term = nDotL;


	float spec_fac = vis_term * cos_term * spec_coef;


	cv_spec_r *= spec_fac * fresnel_r;
	cv_spec_g *= spec_fac * fresnel_g;
	cv_spec_b *= spec_fac * fresnel_b;

	float vmf_pdf = norma * exp(sharpness * (r_dot_i - 1.0));

	float mc_res_spec_r = cv_dir_r_sp * fresnel_r * spec_fac * vmf_pdf * pdf_inv;
	float mc_res_spec_g = cv_dir_g_sp * fresnel_g * spec_fac * vmf_pdf * pdf_inv;
	float mc_res_spec_b = cv_dir_b_sp * fresnel_b * spec_fac * vmf_pdf * pdf_inv;
#endif

	
	if (sh.only_cv == 0.0f) {
		res_r = (cv_diff_r - mc_residual_r) * sh.d_a_r_inv_pi;
		res_g = (cv_diff_g - mc_residual_g) * sh.d_a_g_inv_pi;
		res_b = (cv_diff_b - mc_residual_b) * sh.d_a_b_inv_pi;
	}
	else {
		res_r = cv_diff_r * sh.d_a_r_inv_pi;
		res_g = cv_diff_g * sh.d_a_g_inv_pi;
		res_b = cv_diff_b * sh.d_a_b_inv_pi;
	}
}


template <uint32_t MIN_BAND, uint32_t NUM_BANDS, typename T>
__global__ void compute_sh_cv_store(const uint32_t num_elements, const uint32_t stride, T* __restrict__ data_in, OutputShadingStructure outShadeData)
{

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;


	SHTransformData sh = readSHTransformData(outShadeData.output_trasnform_data, i);
	float res_r, res_g, res_b;
	compute_sh_cv_templated<MIN_BAND, NUM_BANDS>(data_in+i*stride, sh, res_r, res_g, res_b, outShadeData.bands_shade_weights);

	outShadeData.final_output[3 * i] = res_r;
	outShadeData.final_output[3 * i+1] = res_g;
	outShadeData.final_output[3 * i+2] = res_b;
}

TCNN_NAMESPACE_END
