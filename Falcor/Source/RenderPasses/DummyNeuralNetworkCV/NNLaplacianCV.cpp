#include <stdafx.h>
#include <cmath>
#include "NNLaplacianCV.h"
#include "tiny-cuda-nn/network_with_input_encoding.h"
#include "tiny-cuda-nn/loss.h"
#include "tiny-cuda-nn/optimizer.h"
#include "tiny-cuda-nn/networks/fully_fused_mlp.h"
#include "tiny-cuda-nn/losses/relative_sh_luminance.h"
#include "tiny-cuda-nn/trainer.h"
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "RenderPasses/Shared/NeuralNetwork/NeuralStructuresCV.slang"

#pragma comment(lib, "cuda")
#pragma comment(lib, "cudart")
#pragma comment(lib, "curand")
#pragma comment(lib, "cublas")



Falcor::NNLaplacianPart::NNLaplacianPart(
    cudaStream_t training_stream,
    std::string nn_name, 
    std::string& config_path, 
    float3 bbStart, 
    float3 bbEnd,
    uint lod
    ) 
{
    name = nn_name;

    trainingStream = training_stream;

    updateParams(config_path);

    float3 offset = (bbEnd - bbStart) * 0.15f;
    bbEnd += offset;
    bbStart -= offset;
    mLOD = lod;
    if (lod == 1) {
        batchSize = 2048;
    }
    CUDA_CHECK_THROW(cudaMalloc(&bBStart, 3 * sizeof(float)));
    CUDA_CHECK_THROW(cudaMemcpy(bBStart, &bbStart.x, 3 * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK_THROW(cudaMalloc(&bBEnd, 3 * sizeof(float)));
    CUDA_CHECK_THROW(cudaMemcpy(bBEnd, &bbEnd.x, 3 * sizeof(float), cudaMemcpyHostToDevice));
}

void Falcor::NNLaplacianPart::updateParams(std::string& config_path)
{
    std::ifstream networkConfigStream;
    std::string networkConfigFullPath;
    if (findFileInShaderDirectories(config_path, networkConfigFullPath)) {
        networkConfigStream.open(networkConfigFullPath);
    }
    else {
        throw std::runtime_error("Config for " + name + " not found");
    }

    nlohmann::json config;
    networkConfigStream >> config;
    networkConfigStream.close();
    
    outputDims = config["dims"]["output"];
    inputDims = config["dims"]["input"];
    outputTransformDims = config["dims"]["output_transform"];
    outputAfterTransformDims = config["dims"]["output_after_transform"];


    nlohmann::json loss_opts = config.value("loss", nlohmann::json::object());
    nlohmann::json optimizer_opts = config.value("optimizer", nlohmann::json::object());
    nlohmann::json encoding_opts = config.value("encoding", nlohmann::json::object());
    nlohmann::json network_opts = config.value("network", nlohmann::json::object());

    loss.reset(tcnn::create_loss<precision_t>(loss_opts));
    optimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_opts));
    network = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(
        config["dims"]["input"],
        config["dims"]["output"], 
        encoding_opts, 
        network_opts
    );
    
    trainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(network, optimizer, loss);
    epochs = 0;
}

/*
__global__ void select_samples(
    uint32_t n_elements, 
    float* __restrict__ select,
    float* __restrict__ mapping,
    float* __restrict__ result_mapping)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    uint32_t selected_idx = (int)(select[i] * n_elements);

    result_mapping[i] = mapping[selected_idx];
}
*/

float Falcor::NNLaplacianPart::fit(
    tcnn::GPUMatrix<float>& X, 
    tcnn::GPUMatrix<float>& y,
    uint32_t nElements,
    int* mapping,
    tcnn::GPUMatrix<float>* lum,
    bool bGradientDiffusion,
    float* laplace_csrValA,
    int* laplace_csrRowPtrA,
    int* laplace_csrColIndA,
    uint32_t* numNonZeroElements ,
    cudaEvent_t* eventLaplaceFinishMatrixBuild,
    cudaEvent_t* eventLaplaceReceivedNNZE,
    uint32_t denoising_max_iter,
    float diffusion_gradients_blending
    )
{
    if (nElements == 0) {
        return 0;
    }

    float gen_loss_value = 0;
    //batchSize = 200000;
    //batchSize = 10000000;

    uint32_t neededBatchSize = batchSize;
    neededBatchSize = nElements;

    uint32_t numEpochs = (nElements + batchSize - 1) / batchSize;
    if (numEpochs != 4) {
        neededBatchSize = nElements/ 3;
        if (neededBatchSize < 128) {
            neededBatchSize = 128;
        }
        numEpochs = (nElements + neededBatchSize - 1) / neededBatchSize;
    }


    //nElements = (nElements + 127) / 128 * 128;

    //uint32_t offset = 0;
    for (uint32_t i = 0; i < 1; ++i)
    {
        //TODO: test with and without
		//tcnn::generate_random_uniform<float>(*trainingStream.get(), rng, batchSize, batch_selection->data());

        //tcnn::linear_kernel(select_samples, 0, *trainingStream.get(), batchSize, batch_selection->data(), mapping, training_mapping->data());
        float loss_value;

        epochs += 1;
        //uint32_t trainingElements = (i != numEpochs - 1) ? neededBatchSize : nElements % neededBatchSize;
        trainer->training_step(trainingStream, X, y, nElements, mapping, bBStart, bBEnd, &loss_value , lum, bTrain, bGradientDiffusion, laplace_csrValA, laplace_csrRowPtrA, laplace_csrColIndA, numNonZeroElements, eventLaplaceFinishMatrixBuild, eventLaplaceReceivedNNZE, denoising_max_iter, diffusion_gradients_blending);
        gen_loss_value += loss_value;
        //offset += trainingElements;
        /*if (loss_value == std::numeric_limits<float>::quiet_NaN()) {
            std::cout << "NNNAANAN";
        }*/
    }
    lossValue = gen_loss_value / numEpochs;
    numTrainingElements = nElements;
    std::cout << epochs << " " << lossValue << std::endl;
    return lossValue;
}



void Falcor::NNLaplacianPart::predict(
    tcnn::GPUMatrix<float>& X, 
    tcnn::GPUMatrix<float>& y, 
    uint32_t nElements,
    int* mapping,
    tcnn::GPUMatrix<float>& AddData
    )
{
    if (!bPredict)
        return;

    network->inference(trainingStream, X, y, nElements, mapping, bBStart, bBEnd, &AddData);
}

bool Falcor::NNLaplacianPart::renderUI(Gui::Widgets& widget)
{
    if (auto nnGroup = widget.group(name, true))
    {
        nnGroup.text("Loss: " + std::to_string(lossValue));
        nnGroup.text("# Training Elements: " + std::to_string(numTrainingElements));
        nnGroup.text("Step " + std::to_string(epochs));
        nnGroup.checkbox("Predict", bPredict);
        nnGroup.checkbox("Train", bTrain);

        if (auto shGroup = widget.group("SphericalHarmonics", true))
        {
            tcnn::FullyFusedMLP<precision_t, 64>* ff_mlp_network = dynamic_cast<tcnn::FullyFusedMLP<precision_t, 64>*>(network->getNetwork());
            tcnn::RelativeSHLuminanceLoss<precision_t>* sh_loss = dynamic_cast<tcnn::RelativeSHLuminanceLoss<precision_t>*>(loss.get());
            for (uint32_t i = 0; i < 3; i++) {
                float grad_weight = sh_loss->get_band_weight(i);

                std::string grad_str = "Band" + std::to_string(i) + " grad weight";
                if (shGroup.var(grad_str.c_str(), grad_weight, 0.0f, 1.0f, 0.01f)) {
                    sh_loss->update_band_weight(i, grad_weight);
                }

                float shade_weight = ff_mlp_network->getSHBandWeight(i);

                std::string shade_str = "Band" + std::to_string(i) + " shade weight";
                if (shGroup.var(shade_str.c_str(), shade_weight, 0.0f, 1.0f, 0.01f)) {
                    ff_mlp_network->setSHBandWeight(i, shade_weight);
                }
            }
        }
    }

    return false;
}



//uint32_t getPartCount(uint32_t lod)
//{
//    return (uint32_t)std::pow(2, lod * 3);
//}


void getPartDivision(uint32_t lod, uint32_t partNum, const Falcor::AABB bBox, Falcor::float3& bbStart, Falcor::float3& bbEnd, uint32_t scale)
{
    if (lod == 0)
    {
        bbStart = bBox.minPoint;
        bbEnd = bBox.maxPoint;
        return;
    }
    
    Falcor::float3 step = (bBox.maxPoint - bBox.minPoint)/float(scale);
    uint32_t x_index = partNum % scale;
    uint32_t y_index = (partNum / scale) % scale;
    uint32_t z_index = (partNum / scale) / scale;
    Falcor::float3 offset(step.x * x_index, step.y * y_index, step.z * z_index);
    bbStart = bBox.minPoint + offset;
    bbEnd = bBox.minPoint + offset + step;
}

#define ALL_SEQ 0

#if ALL_SEQ
#define RUN_CUDA_SYNC() cudaDeviceSynchronize();
#else
#define RUN_CUDA_SYNC()
#endif

Falcor::NNLaplacianCV::NNLaplacianCV(
    std::string name, 
    std::vector<std::string> config_paths,
    const std::string& pyramid_config_path,
    uint32_t LOD,
    uint32_t level_size,
    const AABB bBox,
    float* denoising_level
    ) 
{
    m_pyramid_config_path = pyramid_config_path;
    CUDA_CHECK_THROW(cudaEventCreate(&fitEventFinishSorting));
    CUDA_CHECK_THROW(cudaEventCreate(&fitEventFinishCounting));

    CUDA_CHECK_THROW(cudaEventCreate(&eventLaplaceFinishSorting));
    CUDA_CHECK_THROW(cudaEventCreate(&eventLaplaceFinishMatrixBuild));
    CUDA_CHECK_THROW(cudaEventCreate(&eventLaplaceReceivedNNZE));


    bPredictionLOD0 = true;
    bTraining = true;
    bTrainingLOD1 = false;
    bPredictingLOD1 = false;

    m_name = name;
    m_LOD = LOD;
    m_level_size = level_size;
    m_temp_storage_bytes = 0;

    m_config_paths = config_paths;
    m_bBox = bBox;
    m_denoising_level = denoising_level;

    create();
}


void Falcor::NNLaplacianCV::updateParams(const std::string& config_path)
{
    std::ifstream networkConfigStream;
    std::string networkConfigFullPath;
    if (findFileInShaderDirectories(config_path, networkConfigFullPath)) {
        networkConfigStream.open(networkConfigFullPath);
    }
    else {
        throw std::runtime_error("Config for " + config_path + " not found");
    }

    nlohmann::json config;
    networkConfigStream >> config;
    networkConfigStream.close();

    m_LOD = config["numLODs"] + 1;
    lodSizes.push_back(1);
    for (uint32_t i = 0; i < m_LOD - 1; i++) {
        lodSizes.push_back(config["lodSize"][i]);
    }

    bGradientDenoising = false;
    if (config["gradient_denoising"]["enable"]) {
        auto data = config["gradient_denoising"];
        bGradientDenoising = true;
        gradientHashGridRes = data["hash_grid_res"];
        gradientHashGridSize = data["hash_grid_size"];

        *m_denoising_level = data.value("level", 10.0f);
        m_denoising_max_iter = data.value("max_iter", 1000);

        auto weighting_data = data["weighting"];

        m_diffusion_weights.distance = weighting_data.value("distance", true);
        if (m_diffusion_weights.distance) {
            m_diffusion_weights.dist_sd = weighting_data.value("distance_sd", 0.1);
        }

        m_diffusion_weights.normal = weighting_data.value("normal", true);
        if (m_diffusion_weights.normal) {
            m_diffusion_weights.normal_power = weighting_data.value("normal_power", 1.0);
            m_diffusion_weights.normal_min = weighting_data.value("normal_min", 0.1);
        }

        m_diffusion_weights.luminance = weighting_data.value("luminance", true);
        if (m_diffusion_weights.luminance) {
            m_diffusion_weights.luminance_sd = weighting_data.value("luminance_sd", 1.0);
            m_diffusion_weights.luminance_min = weighting_data.value("luminance_min", 0.1);
        }

        m_diffusion_gradients_blending = data.value("gradients_blending", 0.05);



    }

    if (!config.value("auto_start", true)) {
        bTraining = false;
        bPredictionLOD0 = false;
    }

    else {
        *m_denoising_level = 10.0f;
        gradientHashGridRes = 256;
        gradientHashGridSize = 1024;
        m_denoising_max_iter = 1000;
    }
}

void Falcor::NNLaplacianCV::create()
{
    lodSizes.clear();
    updateParams(m_pyramid_config_path);
#if !ALL_SEQ
    CUDA_CHECK_THROW(cudaStreamCreate(&m_cuda_sort_stream));
    CUDA_CHECK_THROW(cudaStreamCreate(&m_laplace_stream));
    //CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&0, cudaStreamNonBlocking));
#else
    CUDA_CHECK_THROW(cudaStreamCreate(&m_cuda_sort_stream));
    m_laplace_stream = m_cuda_sort_stream;
    //CUDA_CHECK_THROW(cudaStreamCreate(&m_laplace_stream));
#endif

    //bTraining = false;
    minTrainingElements = 0;
    lumCache = tcnn::GPUMemory<float>(1920 * 1080);
    m_lods_networkds_ids.resize(3);
    uint32_t offset = 0;
    for (uint32_t lod = 0; lod < m_LOD; lod++)
    {
        uint32_t partCount = lodSizes[lod]* lodSizes[lod]* lodSizes[lod];
        
        for (uint32_t i = 0; i < partCount; i++, offset++)
        {
            float3 bbStart;
            float3 bbEnd;
            getPartDivision(lod, i, m_bBox, bbStart, bbEnd, lodSizes[lod]);
            cudaStream_t stream;
#if !ALL_SEQ
            CUDA_CHECK_THROW(cudaStreamCreate(&stream));
#else
            //CUDA_CHECK_THROW(cudaStreamCreate(&stream));
            stream = m_cuda_sort_stream;
#endif

            std::shared_ptr<NNLaplacianPart> crumb(new NNLaplacianPart(
                stream,
                "NNCrumb number " + std::to_string(offset),
                m_config_paths[lod], 
                bbStart, 
                bbEnd, lod));
            
            if (m_input_dims == 0 && m_output_dims == 0)
                crumb->getDims(m_input_dims, m_output_dims, m_out_act_add_data_dims, m_output_after_transform_dims);
            else
            {
                uint32_t input_dims, output_dims, ka, ka2;
                crumb->getDims(input_dims, output_dims, ka, ka2);
                if(m_input_dims != input_dims || m_output_dims != output_dims)
                    throw std::runtime_error("Unequal dims for NNlaplacianParts");
            }
            uint32_t id = m_networks.size();
            m_networks.push_back(crumb);


            m_lods_networkds_ids[lod].push_back(id);
        }
    }

    modelsEventsPredict.resize(m_networks.size());
    modelsEventsFit.resize(m_networks.size());
    for (uint32_t i = 0; i < modelsEventsPredict.size(); i++) {
        CUDA_CHECK_THROW(cudaEventCreate(&modelsEventsPredict[i]));
        CUDA_CHECK_THROW(cudaEventCreate(&modelsEventsFit[i]));
    }

    CUDA_CHECK_THROW(cudaEventCreate(&initFinalBufferEvent));
    CUDA_CHECK_THROW(cudaEventCreate(&eventFinishSorting));
    CUDA_CHECK_THROW(cudaEventCreate(&eventFinishCounting));
}

__global__ void count_idNN(uint32_t num, uint32_t* __restrict__ keys, int* __restrict__ count)
{
    __shared__ uint32_t smem[256];


    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;

    if (threadIdx.x < 256) {
        smem[threadIdx.x] = 0;
    }

    __syncthreads();
    atomicAdd(&smem[keys[i]], 1);
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&count[threadIdx.x], smem[threadIdx.x]);
    }
}

__global__ void count_map_laplace(const uint32_t num, const int* __restrict__ keys, const uint32_t gridSize, int* __restrict__ res_count)
{
    __shared__ uint32_t count[4096];

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    


    if (threadIdx.x < gridSize) {
        for(uint32_t j=0; j < 4; j++)
            count[threadIdx.x*4+j] = 0;
    }

    __syncthreads();
    if (i < num) {
        atomicAdd(&count[keys[i]], 1);
    }
     __syncthreads();

    if (threadIdx.x < gridSize) {
        for (uint32_t j = 0; j < 4; j++)
            atomicAdd(&res_count[threadIdx.x*4+j], count[threadIdx.x * 4 + j]);
    }
}


__global__ void set_zero(uint32_t num, float* __restrict__ data)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;

    data[i] = 0.0f;
}



__device__ int fast_hash(const int3 pos_grid, const int hash_grid_size) {
    
    uint32_t primes[7] = { 25165843, 19349663, 83492791, 25165843, 6291469, 12582917, 3145739 };

    uint32_t result = 0;
    result ^= uint32_t(pos_grid.x) * primes[0];
    result ^= uint32_t(pos_grid.y) * primes[1];
    result ^= uint32_t(pos_grid.z) * primes[2];
    return (result% hash_grid_size);
}

#define TRAVERSE_NEIGHBORS 1
#define DUMMY_IDENTITY 0


__global__ void build_laplace_coo_matrix_finish(const int num_elements, const uint32_t hash_grid_res, const uint32_t hash_grid_size, const float dlr, const int* __restrict__ cells_base_index, const int* __restrict__ cells_sizes, const int* __restrict__ sorted_samples_ids, const float* __restrict__ samples_positions, const int* row_offset, const int* num_elements_per_row, const int* cached_good_cells, float* __restrict__ csrValA, int* __restrict__ csrColIndA)
{
    static const int3 offsets[27] = {
        {0, 0, 0}, {-1, -1, -1}, {-1, -1, 0},{-1, -1, 1}, {-1, 0, -1},{-1, 0, 0}, {-1, 0, 1},{-1, 1, -1},{-1, 1, 0}, {-1, 1, 1},{0, -1, -1}
    , {0, -1, 0},{0, -1, 1}, {0, 0, -1}, {0, 0, 1},{0, 1, -1},{0, 1, 0}, {0, 1, 1},
        {1, -1, -1}, {1, -1, 0},{1, -1, 1}, {1, 0, -1},{1, 0, 0}, {1, 0, 1},{1, 1, -1},{1, 1, 0}, {1, 1, 1}
    };

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    uint32_t base_vert_id = sorted_samples_ids[i]; // the row

    float3 base_vert_pos = make_float3(samples_positions[base_vert_id * 3], samples_positions[base_vert_id * 3 + 1], samples_positions[base_vert_id * 3 + 2]);

    int3 base_cell_grid_pos;
    base_cell_grid_pos.x = floor((hash_grid_res-1) * base_vert_pos.x);
    base_cell_grid_pos.y = floor((hash_grid_res-1) * base_vert_pos.y);
    base_cell_grid_pos.z = floor((hash_grid_res-1) * base_vert_pos.z);


    int num_good_cells = 0;
    uint32_t good_cells_packed = 0;

    // Find not empty neighboring cells
    uint32_t power = 1;
    int32_t total_connected_vertices = 0;

    uint32_t cached_good_cell = cached_good_cells[base_vert_id];

    uint32_t r_offset = row_offset[base_vert_id];
    const uint32_t num_elements_in_row = num_elements_per_row[base_vert_id];
    uint32_t id = 0;
    int c = 0;
    do {
        if (num_elements_in_row - 1 == id)
            break;

        bool bStop = false;
        while (!(cached_good_cell & 1)) {
            c++;
            cached_good_cell = cached_good_cell >> 1;

            if (c >= 1 + 26 * TRAVERSE_NEIGHBORS) {
                bStop = true;
                break;
            }
        }

        if (bStop)
            break;

        int3 cell_grid_pos = base_cell_grid_pos;

        cell_grid_pos.x += offsets[c].x;
        cell_grid_pos.y += offsets[c].y;
        cell_grid_pos.z += offsets[c].z;
        if (cell_grid_pos.x < 0 || cell_grid_pos.x >= hash_grid_res ||
            cell_grid_pos.y < 0 || cell_grid_pos.y >= hash_grid_res ||
            cell_grid_pos.z < 0 || cell_grid_pos.z >= hash_grid_res)
        {
            continue;
        }

        const int cell_id = fast_hash(cell_grid_pos, hash_grid_size);
        const int num_elements = cells_sizes[cell_id];
        int num_real_elements = 0;
        const int cell_offset = cells_base_index[cell_id];

        for (int j = 0; j < num_elements; j++) {
            int vert_id = sorted_samples_ids[cell_offset + j];
            if (vert_id == base_vert_id)
                continue;

            float3 vert_pos = make_float3(samples_positions[vert_id * 3], samples_positions[vert_id * 3 + 1], samples_positions[vert_id * 3 + 2]);

            int3 cell_grid_pos;
            cell_grid_pos.x = floor((hash_grid_res-1) * vert_pos.x);
            cell_grid_pos.y = floor((hash_grid_res-1) * vert_pos.y);
            cell_grid_pos.z = floor((hash_grid_res-1) * vert_pos.z);

            // Because of hash collision some samples from different cells can get into the same cell. This is why we again need to check is it a close neighboring sample or not
            if (abs(cell_grid_pos.x - base_cell_grid_pos.x) > 1 ||
                abs(cell_grid_pos.y - base_cell_grid_pos.y) > 1 ||
                abs(cell_grid_pos.z - base_cell_grid_pos.z) > 1)
            {
                continue;
            }

            csrValA[r_offset + id] = -1.0f * dlr;
            csrColIndA[r_offset + id] = vert_id;
            id += 1;

            if (num_elements_in_row - 1 == id)
                break;
        }
    } while (c < 1 + 26 * TRAVERSE_NEIGHBORS);


    // Add the point itself + indentity matrix
    csrValA[r_offset + id] = id* dlr+1.0f; 
    csrColIndA[r_offset + id] = base_vert_id;
}


__device__ float normal_pdf(float x, float m, float inv_s)
{
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) * inv_s;

    return inv_sqrt_2pi * inv_s * exp(-0.5f * a * a);
}

__device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void build_laplace_coo_matrix_finish_22(const int num_elements, const uint32_t hash_grid_res, const uint32_t hash_grid_size, const float dlr, const int* __restrict__ cells_base_index, const int* __restrict__ cells_sizes, const int* __restrict__ sorted_samples_ids, const float* __restrict__ samples_positions, const float* __restrict__ samples_normals, const int* row_offset, const int* num_elements_per_row, const int* cached_good_cells, float* __restrict__ csrValA, int* __restrict__ csrColIndA, const WeightingData diffusion_weights, const float* radiance)
{
    static const int3 offsets[27] = {
        {0, 0, 0}, {-1, -1, -1}, {-1, -1, 0},{-1, -1, 1}, {-1, 0, -1},{-1, 0, 0}, {-1, 0, 1},{-1, 1, -1},{-1, 1, 0}, {-1, 1, 1},{0, -1, -1}
    , {0, -1, 0},{0, -1, 1}, {0, 0, -1}, {0, 0, 1},{0, 1, -1},{0, 1, 0}, {0, 1, 1},
        {1, -1, -1}, {1, -1, 0},{1, -1, 1}, {1, 0, -1},{1, 0, 0}, {1, 0, 1},{1, 1, -1},{1, 1, 0}, {1, 1, 1}
    };

    uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= num_elements) return;

    uint32_t base_vert_id = sorted_samples_ids[threadid]; // the row

    float3 base_vert_pos = make_float3(samples_positions[base_vert_id * 3], samples_positions[base_vert_id * 3 + 1], samples_positions[base_vert_id * 3 + 2]);
    float3 base_vert_normal = make_float3(samples_normals[base_vert_id * 3], samples_normals[base_vert_id * 3 + 1], samples_normals[base_vert_id * 3 + 2]);
    const float base_vert_lum = radiance[base_vert_id * 3] * 0.33 + radiance[base_vert_id * 3 + 1] * 0.33 + radiance[base_vert_id * 3 + 2] * 0.33;


    int3 base_cell_grid_pos;
    base_cell_grid_pos.x = floor((hash_grid_res - 1) * base_vert_pos.x);
    base_cell_grid_pos.y = floor((hash_grid_res - 1) * base_vert_pos.y);
    base_cell_grid_pos.z = floor((hash_grid_res - 1) * base_vert_pos.z);


    int num_good_cells = 0;
    uint32_t good_cells_packed = 0;

    // Find not empty neighboring cells
    uint32_t power = 1;
    int32_t total_connected_vertices = 0;

    int m = 0;
    int32_t visitedCells[27];
    uint32_t r_offset = row_offset[base_vert_id];
    const uint32_t num_elements_in_row = num_elements_per_row[base_vert_id];
    uint32_t id = 0;

    float total_accum = 0.0f;
    for (uint32_t i = 0; i < 1 + 26 * TRAVERSE_NEIGHBORS; i++) {
        if (num_elements_in_row - 1 == id)
            break;

        int3 cell_grid_pos = base_cell_grid_pos;

        cell_grid_pos.x += offsets[i].x;
        cell_grid_pos.y += offsets[i].y;
        cell_grid_pos.z += offsets[i].z;
        if (cell_grid_pos.x < 0 || cell_grid_pos.x >= hash_grid_res ||
            cell_grid_pos.y < 0 || cell_grid_pos.y >= hash_grid_res ||
            cell_grid_pos.z < 0 || cell_grid_pos.z >= hash_grid_res)
        {
        }
        else {
            const int cell_id = fast_hash(cell_grid_pos, hash_grid_size);
            bool isAlreadyVisited = false;
            for (uint32_t c = 0; c < m; c++) {
                if (visitedCells[c] == cell_id) {
                    isAlreadyVisited = true;
                    break;
                }
            }
            if (!isAlreadyVisited) {
                visitedCells[m] = cell_id;
                m++;

                const int num_elements = cells_sizes[cell_id];
                const int cell_offset = cells_base_index[cell_id];

                for (int j = 0; j < num_elements; j++) {
                    uint32_t vert_id = sorted_samples_ids[cell_offset + j];
                    if (vert_id == base_vert_id)
                        continue;
                    


                    float3 vert_pos = make_float3(samples_positions[vert_id * 3], samples_positions[vert_id * 3 + 1], samples_positions[vert_id * 3 + 2]);

                    int3 cell_grid_pos;
                    cell_grid_pos.x = floor((hash_grid_res - 1) * vert_pos.x);
                    cell_grid_pos.y = floor((hash_grid_res - 1) * vert_pos.y);
                    cell_grid_pos.z = floor((hash_grid_res - 1) * vert_pos.z);



                    // Because of hash collision some samples from different cells can get into the same cell. This is why we again need to check is it a close neighboring sample or not
                    if (abs(cell_grid_pos.x - base_cell_grid_pos.x) > 1 ||
                        abs(cell_grid_pos.y - base_cell_grid_pos.y) > 1 ||
                        abs(cell_grid_pos.z - base_cell_grid_pos.z) > 1)
                    {
                        continue;
                    }

                    float weight = 1.0f;

                    if (diffusion_weights.distance) {
                        const float sd_inv = 1.0f/diffusion_weights.dist_sd;

                        float3 shift;
                        shift.x = vert_pos.x - base_vert_pos.x;
                        shift.x = vert_pos.x - base_vert_pos.x;
                        shift.x = vert_pos.x - base_vert_pos.x;

                        weight *= normal_pdf(vert_pos.x, base_vert_pos.x, sd_inv);
                        weight *= normal_pdf(vert_pos.y, base_vert_pos.y, sd_inv);
                        weight *= normal_pdf(vert_pos.z, base_vert_pos.z, sd_inv);
                    }
                    if (diffusion_weights.normal) {

                        const float3 vert_normal = make_float3(samples_normals[vert_id * 3], samples_normals[vert_id * 3 + 1], samples_normals[vert_id * 3 + 2]);
                        const float kinv = (1.0f - diffusion_weights.normal_min);

                        float normal_d = dot(vert_normal, base_vert_normal);
                        if (normal_d < 0.0) {
                            normal_d = 0;
                        }
                        else {
                            normal_d = pow(normal_d, diffusion_weights.normal_power);
                        }

                        weight *= (diffusion_weights.normal_min + kinv * normal_d);
                    }
                    if (diffusion_weights.luminance) {
                        const float sd_inv = 1.0f / (diffusion_weights.luminance_sd+0.00001);
                        const float kinv = (1.0f - diffusion_weights.luminance_min);

                        float vert_lum = radiance[vert_id * 3] * 0.33 + radiance[vert_id * 3 + 1] * 0.33 + radiance[vert_id * 3 + 2] * 0.33;
                        weight *= normal_pdf(vert_lum, base_vert_lum, sd_inv)*kinv+ diffusion_weights.luminance_min;
                    }

                    total_accum += weight;
                    csrValA[r_offset + id] = -1.0f * dlr*weight;
                    csrColIndA[r_offset + id] = vert_id;
                    id += 1;

                    if (num_elements_in_row - 1 == id)
                        break;
                }
            }
        }
    }

    // Add the point itself + indentity matrix
    csrValA[r_offset + id] = total_accum * dlr + 1.0f;
    csrColIndA[r_offset + id] = base_vert_id;
}

__global__ void build_laplace_coo_matrix_prepare(const int num_elements, const uint32_t hash_grid_res, const uint32_t hash_grid_size, const int* __restrict__ cells_base_index, const int* __restrict__ cells_sizes, const int* __restrict__ sorted_samples_ids, const float* __restrict__ samples_positions, int* num_elements_per_row, int* cached_good_cells)
{
    static const int3 offsets[27] = {
        {0, 0, 0}, {-1, -1, -1}, {-1, -1, 0},{-1, -1, 1}, {-1, 0, -1},{-1, 0, 0}, {-1, 0, 1},{-1, 1, -1},{-1, 1, 0}, {-1, 1, 1},{0, -1, -1}
    , {0, -1, 0},{0, -1, 1}, {0, 0, -1}, {0, 0, 1},{0, 1, -1},{0, 1, 0}, {0, 1, 1},
        {1, -1, -1}, {1, -1, 0},{1, -1, 1}, {1, 0, -1},{1, 0, 0}, {1, 0, 1},{1, 1, -1},{1, 1, 0}, {1, 1, 1}
    };

    uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= num_elements) return;

    uint32_t base_vert_id = sorted_samples_ids[threadid]; // the row

    float3 base_vert_pos = make_float3(samples_positions[base_vert_id * 3], samples_positions[base_vert_id * 3 + 1], samples_positions[base_vert_id * 3 + 2]);



    int3 base_cell_grid_pos;
    base_cell_grid_pos.x = floor((hash_grid_res - 1) * base_vert_pos.x);
    base_cell_grid_pos.y = floor((hash_grid_res - 1) * base_vert_pos.y);
    base_cell_grid_pos.z = floor((hash_grid_res - 1) * base_vert_pos.z);


    int num_good_cells = 0;
    uint32_t good_cells_packed = 0;

    // Find not empty neighboring cells
    uint32_t power = 1;
    int32_t total_connected_vertices = 0;

    int m = 0;
    int32_t visitedCells[27];


    for (uint32_t i = 0; i < 1+26* TRAVERSE_NEIGHBORS; i++) {
        int3 cell_grid_pos = base_cell_grid_pos;
        int num_real_elements = 0;

        cell_grid_pos.x += offsets[i].x;
        cell_grid_pos.y += offsets[i].y;
        cell_grid_pos.z += offsets[i].z;
        if (cell_grid_pos.x < 0 || cell_grid_pos.x >= hash_grid_res ||
            cell_grid_pos.y < 0 || cell_grid_pos.y >= hash_grid_res ||
            cell_grid_pos.z < 0 || cell_grid_pos.z >= hash_grid_res)
        {
        }
        else {
            const int cell_id = fast_hash(cell_grid_pos, hash_grid_size);
            bool isAlreadyVisited = false;
            for (uint32_t c = 0; c < m; c++) {
                if (visitedCells[c] == cell_id) {
                    isAlreadyVisited = true;
                    break;
                }
            }
            if (!isAlreadyVisited) {
                visitedCells[m] = cell_id;
                m++;

                const int num_elements = cells_sizes[cell_id];
                const int cell_offset = cells_base_index[cell_id];

                for (int j = 0; j < num_elements; j++) {
                    uint32_t vert_id = sorted_samples_ids[cell_offset + j];
                    if (vert_id == base_vert_id) {
                        num_real_elements += 1;
                        continue;
                    }


                    float3 vert_pos = make_float3(samples_positions[vert_id * 3], samples_positions[vert_id * 3 + 1], samples_positions[vert_id * 3 + 2]);

                    int3 cell_grid_pos;
                    cell_grid_pos.x = floor((hash_grid_res - 1) * vert_pos.x);
                    cell_grid_pos.y = floor((hash_grid_res - 1) * vert_pos.y);
                    cell_grid_pos.z = floor((hash_grid_res - 1) * vert_pos.z);



                    // Because of hash collision some samples from different cells can get into the same cell. This is why we again need to check is it a close neighboring sample or not
                    if (abs(cell_grid_pos.x - base_cell_grid_pos.x) > 1 ||
                        abs(cell_grid_pos.y - base_cell_grid_pos.y) > 1 ||
                        abs(cell_grid_pos.z - base_cell_grid_pos.z) > 1)
                    {
                        continue;
                    }

                    num_real_elements += 1;
                }
            }
        }


        if(num_real_elements > 0)
            good_cells_packed += power;
        power *= 2; // shift bits left

        total_connected_vertices += num_real_elements;
    }
#if DUMMY_IDENTITY
    total_connected_vertices = 1;
#endif

#if 1
    if (total_connected_vertices == 0) {
        total_connected_vertices = -1;
    }
#endif

    num_elements_per_row[base_vert_id] = total_connected_vertices;
    cached_good_cells[base_vert_id] = good_cells_packed;
}


void Falcor::NNLaplacianCV::build_laplace_operator(
    uint32_t numElements,
    Falcor::Buffer::SharedPtr mapping_key,
    Falcor::Buffer::SharedPtr mapping_value,
    Falcor::Buffer::SharedPtr positions,
    Falcor::Buffer::SharedPtr normals,
    Falcor::Buffer::SharedPtr train_radiance,
    GpuFence::SharedPtr fence,
    uint64_t fenceValue
)
{
    if ((!bTraining && !bTrainingLOD1 && !bTrainingLOD2) || numElements == 0)
        return;

    if (!bGradientDenoising)
        return;

    if (*m_denoising_level == 0.0f)
        return;


    RUN_CUDA_SYNC()
    int* d_keys_in = (int*)mapping_key->getCUDADeviceAddress();
    int* d_values_in = (int*)mapping_value->getCUDADeviceAddress();
    float* d_positions_in = (float*)positions->getCUDADeviceAddress();
    float* d_normals_in = (float*)normals->getCUDADeviceAddress();
    float* d_radiance_in = (float*)train_radiance->getCUDADeviceAddress();
    int num_items = m_level_size;
    bool did_allocate = false;
    RUN_CUDA_SYNC()
    if (m_laplace_sort_temp_storage_size == 0) {
        int MAX_VERTICES = m_level_size / 32*4;
        did_allocate = true;
        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_keys_out, m_level_size * sizeof(int)));
        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_values_out, m_level_size * sizeof(int)));
        CUDA_CHECK_THROW(cub::DeviceRadixSort::SortPairs(m_laplace_sort_temp_storage, m_laplace_sort_temp_storage_size, d_keys_in, m_laplace_keys_out, d_values_in, m_laplace_values_out, MAX_VERTICES, 0, sizeof(uint32_t) * (8), m_laplace_stream));

        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_sort_temp_storage, m_laplace_sort_temp_storage_size));
        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_mapper, (gradientHashGridSize+1) * sizeof(int)));
        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_counter, gradientHashGridSize * sizeof(int)));

        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_elements_per_row, MAX_VERTICES * sizeof(int)));
        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_row_offset, (MAX_VERTICES +1) * sizeof(int)));

        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_cached_cells_mask, gradientHashGridSize * sizeof(uint32_t)));
        RUN_CUDA_SYNC()

        CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(m_laplace_prefixsum_temp_storage, m_laplace_prefixsum_temp_storage_size, m_laplace_elements_per_row, m_laplace_row_offset, MAX_VERTICES, m_laplace_stream));
        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_prefixsum_temp_storage, m_laplace_prefixsum_temp_storage_size));
        RUN_CUDA_SYNC()


        CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(m_laplace_prefixsum_temp_storage_2, m_laplace_prefixsum_temp_storage_size_2, m_laplace_counter, m_laplace_mapper + 1, gradientHashGridSize, m_laplace_stream));
        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_prefixsum_temp_storage_2, m_laplace_prefixsum_temp_storage_size_2));


        int MAX_CONNECT_VERTICES = MAX_VERTICES * 10;
        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_csrValA, MAX_CONNECT_VERTICES * sizeof(float)));
        CUDA_CHECK_THROW(cudaMalloc(&m_laplace_csrColIndA, MAX_CONNECT_VERTICES * sizeof(int)));
    }


    num_items = numElements;

    
        CUDA_CHECK_THROW(cudaMemsetAsync(m_laplace_mapper, 0, 4, m_laplace_stream));
        CUDA_CHECK_THROW(cudaMemsetAsync(m_laplace_row_offset, 0, 4, m_laplace_stream));

        
        CUDA_CHECK_THROW(cub::DeviceRadixSort::SortPairs(m_laplace_sort_temp_storage, m_laplace_sort_temp_storage_size, d_keys_in, m_laplace_keys_out, d_values_in, m_laplace_values_out, num_items, 0, sizeof(uint32_t) * (8), m_laplace_stream));
        cudaEventRecord(eventLaplaceFinishSorting, m_laplace_stream);

        CUDA_CHECK_THROW(cudaMemsetAsync(m_laplace_counter, 0, gradientHashGridSize * 4, m_laplace_stream));
        
        count_map_laplace<<<(num_items + 1023) / 1024, 1024, 0, m_laplace_stream >> > (num_items, m_laplace_keys_out, gradientHashGridSize, m_laplace_counter);
        
        CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(m_laplace_prefixsum_temp_storage_2, m_laplace_prefixsum_temp_storage_size_2, m_laplace_counter, m_laplace_mapper + 1, gradientHashGridSize, m_laplace_stream));


        // compute number of connected vertices per vertex and also some masks and other data
        // that can speed up the final stage of building laplace operator
        tcnn::linear_kernel(build_laplace_coo_matrix_prepare, 0, m_laplace_stream, num_items,
            gradientHashGridRes, gradientHashGridSize, m_laplace_mapper, m_laplace_counter,
            m_laplace_values_out, d_positions_in, m_laplace_elements_per_row, m_laplace_cached_cells_mask);

        
        CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(m_laplace_prefixsum_temp_storage, m_laplace_prefixsum_temp_storage_size, m_laplace_elements_per_row, m_laplace_row_offset + 1, num_items, m_laplace_stream));
        
        CUDA_CHECK_THROW(cudaMemcpyAsync(&m_laplace_num_non_zero_elements, m_laplace_row_offset + num_items, 4, cudaMemcpyDeviceToHost, m_laplace_stream));

#if 0
        std::vector<int> elements_per_hash_cell(100000);
        std::vector<int> offsets_hash_cell(100000);
        std::vector<int> sorted_vertices(100000);
        std::vector<int> sorted_vertices_keys(100000);
        std::vector<int> elementes_per_row(100000);
        std::vector<Falcor::float3> pos(100000);

        CUDA_CHECK_THROW(cudaMemcpy(elements_per_hash_cell.data(), m_laplace_counter, 4 * gradientHashGridSize, cudaMemcpyDeviceToHost));
        CUDA_CHECK_THROW(cudaMemcpy(offsets_hash_cell.data(), m_laplace_mapper, 4 * (gradientHashGridSize + 1), cudaMemcpyDeviceToHost));
        CUDA_CHECK_THROW(cudaMemcpy(sorted_vertices.data(), m_laplace_values_out, 4 * num_items, cudaMemcpyDeviceToHost));
        CUDA_CHECK_THROW(cudaMemcpy(sorted_vertices_keys.data(), m_laplace_keys_out, 4 * num_items, cudaMemcpyDeviceToHost));
        CUDA_CHECK_THROW(cudaMemcpy(pos.data(), d_positions_in, 4 * num_items * 3, cudaMemcpyDeviceToHost));
        CUDA_CHECK_THROW(cudaMemcpy(elementes_per_row.data(), m_laplace_elements_per_row, 4 * num_items, cudaMemcpyDeviceToHost));

        RUN_CUDA_SYNC()
#endif

        if(m_diffusion_weights.luminance)
        {
            if (!bInitCudaFenceForTrainingData) {
                cudaExternalSemaphoreHandleDesc d;
                memset(&d, 0, sizeof(cudaExternalSemaphoreHandleDesc));
                d.type = cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeD3D12Fence;
                d.handle.win32.handle = fence->getSharedApiHandle();
                CUDA_CHECK_THROW(cudaImportExternalSemaphore(&cudaFenceForTrainingData, &d));
                bInitCudaFenceForTrainingData = true;
            }

            cudaExternalSemaphoreWaitParams a;
            memset(&a, 0, sizeof(cudaExternalSemaphoreWaitParams));
            a.params.fence.value = fenceValue;

       
           CUDA_CHECK_THROW(cudaWaitExternalSemaphoresAsync(&cudaFenceForTrainingData, &a, 1, m_laplace_stream));
        }

    tcnn::linear_kernel(build_laplace_coo_matrix_finish_22, 0, m_laplace_stream, num_items,
    gradientHashGridRes, gradientHashGridSize, *m_denoising_level, m_laplace_mapper,
    m_laplace_counter, m_laplace_values_out, d_positions_in, d_normals_in, m_laplace_row_offset,
    m_laplace_elements_per_row, m_laplace_cached_cells_mask, m_laplace_csrValA, m_laplace_csrColIndA,
    m_diffusion_weights, d_radiance_in);
        
     
    cudaEventRecord(eventLaplaceFinishMatrixBuild, m_laplace_stream);
    cudaEventRecord(eventLaplaceReceivedNNZE, m_laplace_stream);
#if 0
    std::vector<float> laplace_csrValA(500000);
    std::vector<int> laplace_csrColIndA(500000);
    std::vector<int> laplace_row_offset(500000);

    CUDA_CHECK_THROW(cudaMemcpy(laplace_csrValA.data(), m_laplace_csrValA, (m_laplace_num_non_zero_elements)*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK_THROW(cudaMemcpy(laplace_csrColIndA.data(), m_laplace_csrColIndA, (m_laplace_num_non_zero_elements) * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK_THROW(cudaMemcpy(laplace_row_offset.data(), m_laplace_row_offset, (num_items+1) * 4, cudaMemcpyDeviceToHost));
    RUN_CUDA_SYNC()
#endif
}

void Falcor::NNLaplacianCV::fit_sorting(
    uint32_t numElements,
    Falcor::Buffer::SharedPtr mapping_key,
    Falcor::Buffer::SharedPtr mapping_value
) {
    if ((!bTraining && !bTrainingLOD1 && !bTrainingLOD2) || numElements == 0)
        return;

    RUN_CUDA_SYNC()
    uint32_t* d_keys_in = (uint32_t*)mapping_key->getCUDADeviceAddress();
    int* d_values_in = (int*)mapping_value->getCUDADeviceAddress();

    int num_items = m_level_size * m_LOD;

    RUN_CUDA_SYNC()
    if (!fitWaitingEvents.empty()) {
        // Delete all events
        //CUDA_CHECK_THROW(cudaEventDestroy(fitEventFinishSorting));
        
    }
    RUN_CUDA_SYNC()
        if (m_temp_storage_bytes == 0) {
            if (m_LOD > 1) {
                CUDA_CHECK_THROW(cudaMalloc(&m_d_values_out, m_level_size * m_LOD * sizeof(int)));
                CUDA_CHECK_THROW(cudaMalloc(&m_d_keys_out, m_level_size * m_LOD * sizeof(uint32_t)));
            }
            else {
                m_d_keys_out = d_keys_in;
                m_d_values_out = d_values_in;
            }   

            CUDA_CHECK_THROW(cudaMalloc(&m_d_values_out_predict, m_level_size * (m_LOD - 1) * sizeof(uint32_t)));
            CUDA_CHECK_THROW(cudaMalloc(&m_d_keys_out_predict, m_level_size * (m_LOD - 1) * sizeof(uint32_t)));

            CUDA_CHECK_THROW(cub::DeviceRadixSort::SortPairs(m_d_temp_storage, m_temp_storage_bytes, d_keys_in, m_d_keys_out, d_values_in, m_d_values_out, num_items, 0, sizeof(uint8_t) * (8), m_cuda_sort_stream));
            //m_temp_storage_bytes = m_level_size * m_LOD * sizeof(int) + 128;

            CUDA_CHECK_THROW(cudaMalloc(&m_d_temp_storage, m_temp_storage_bytes));
            CUDA_CHECK_THROW(cudaMalloc(&m_counts_device, (m_networks.size() + 1) * sizeof(int)));

            m_counts.resize(m_networks.size() + 1);
        }

    num_items = numElements * m_LOD;

    RUN_CUDA_SYNC()
    if (m_LOD > 1) {
        // Sorting async
        CUDA_CHECK_THROW(cub::DeviceRadixSort::SortPairs(m_d_temp_storage, m_temp_storage_bytes, d_keys_in, m_d_keys_out, d_values_in, m_d_values_out, num_items, 0, sizeof(uint8_t) * (8), m_cuda_sort_stream));
    }
    cudaEventRecord(fitEventFinishSorting, m_cuda_sort_stream);
    RUN_CUDA_SYNC()

}

void Falcor::NNLaplacianCV::fit(
    uint32_t numElements,
    Falcor::Buffer::SharedPtr X,
    Falcor::Buffer::SharedPtr y,
    Falcor::Buffer::SharedPtr mapping_key,
    Falcor::Buffer::SharedPtr mapping_value,
    GpuFence::SharedPtr fence,
    uint64_t fenceValue
    )
{
    if ((!bTraining && !bTrainingLOD1 && !bTrainingLOD2) || numElements == 0)
        return;
    RUN_CUDA_SYNC()
    // Waiting shaders execution. They fill all structure buffers for our networks
    {
        if (!bInitCudaFenceForTrainingData) {
            cudaExternalSemaphoreHandleDesc d;
            memset(&d, 0, sizeof(cudaExternalSemaphoreHandleDesc));
            d.type = cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeD3D12Fence;
            d.handle.win32.handle = fence->getSharedApiHandle();
            CUDA_CHECK_THROW(cudaImportExternalSemaphore(&cudaFenceForTrainingData, &d));
            bInitCudaFenceForTrainingData = true;
        }
        
        cudaExternalSemaphoreWaitParams a;
        memset(&a, 0, sizeof(cudaExternalSemaphoreWaitParams));
        a.params.fence.value = fenceValue;

        for (uint32_t i = 0; i < m_networks.size(); i++) {
            CUDA_CHECK_THROW(cudaWaitExternalSemaphoresAsync(&cudaFenceForTrainingData, &a, 1, m_networks[i]->getCudaStream()));
        }

        //CUDA_CHECK_THROW(cudaWaitExternalSemaphoresAsync(&cudaFenceForTrainingData, &a, 1, m_cuda_sort_stream));
//        CUDA_CHECK_THROW(cudaWaitExternalSemaphoresAsync(&cudaFenceForTrainingData, &a, 1, 0));
    }
RUN_CUDA_SYNC()

    int num_items = m_level_size * m_LOD;

    

    uint32_t buffer_len = X->getElementCount();

    tcnn::GPUMemory<float> tcnn_X((float*)X->getCUDADeviceAddress(), numElements * m_input_dims);
    tcnn_X.setShared(true);

    tcnn::GPUMemory<float> tcnn_y((float*)y->getCUDADeviceAddress(), numElements * sizeof(Falcor::YData) / sizeof(float));
    tcnn_y.setShared(true);

    

    tcnn::GPUMatrix<float> matrix_X(tcnn_X.data(), m_input_dims, numElements);
    tcnn::GPUMatrix<float> matrix_y(tcnn_y.data(), sizeof(Falcor::YData)/sizeof(float), numElements);
    tcnn::GPUMatrix<float> matrix_Lum(lumCache.data(), 1, numElements);

RUN_CUDA_SYNC()

    fitWaitingEvents.reserve(modelsEventsFit.size());
    if (!fitWaitingEvents.empty()) {
        // Delete all events
        /*for (uint32_t j = 0; j < fitWaitingEvents.size(); j++) {
            CUDA_CHECK_THROW(cudaEventDestroy(fitWaitingEvents[j]));
        }*/
        fitWaitingEvents.clear();
    }
RUN_CUDA_SYNC()
    num_items = numElements*m_LOD;
    
    uint32_t* d_keys_in = (uint32_t*)mapping_key->getCUDADeviceAddress();

    

RUN_CUDA_SYNC()
    if (m_LOD > 1) {
        // Counting async
        CUDA_CHECK_THROW(cudaMemsetAsync(m_counts_device, 0, (m_networks.size() + 1) * sizeof(int), 0));
        tcnn::linear_kernel(count_idNN, 0, 0, num_items, d_keys_in, m_counts_device);
        //CUDA_CHECK_THROW(cudaEventRecord(fitEventFinishCounting, 0));


        RUN_CUDA_SYNC()
            // Sync copy count results from a device to the host
        //CUDA_CHECK_THROW(cudaStreamWaitEvent(0, fitEventFinishCounting));
        CUDA_CHECK_THROW(cudaMemcpy(m_counts.data(), m_counts_device, (m_networks.size() + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    }
    else {
        if(m_counts.empty())
            m_counts.resize(m_networks.size() + 1);

        m_counts[0] = numElements;
    }
RUN_CUDA_SYNC()
    uint32_t offset = 0;

   // CUDA_CHECK_THROW(cudaMemsetAsync(matrix_Lum.data(), 0.0f, matrix_Lum.n_elements() * sizeof(float), m_networks[0]->getCudaStream()));
    for (uint32_t lod = 0; lod < 1+bTrainingLOD1+bTrainingLOD2; lod++)
    {
        uint32_t numModelsInLOD = m_lods_networkds_ids[lod].size();
        uint32_t waitNum = fitWaitingEvents.size();
        for (uint32_t i = 0; i < numModelsInLOD; i++) {
            uint32_t modelID = m_lods_networkds_ids[lod][i];
            uint32_t mappings_count = m_counts[modelID];
            if (mappings_count == 0)
                continue;
            uint32_t trainElements = mappings_count;
            if (!(modelID == 0 && !bTraining) && (trainElements >= minTrainingElements)) {
                RUN_CUDA_SYNC()
                // Waiting the finish of sorting
                CUDA_CHECK_THROW(cudaStreamWaitEvent(m_networks[modelID]->getCudaStream(), fitEventFinishSorting));

                const uint32_t numFences = (bSeqTraining) ? fitWaitingEvents.size() : waitNum;
                for (uint32_t j = 0; j < numFences; j++) {
                    // Waiting the finish of executing models from last LODs
                    CUDA_CHECK_THROW(cudaStreamWaitEvent(m_networks[modelID]->getCudaStream(), fitWaitingEvents[j]));   
                }
                m_networks[modelID]->fit(matrix_X, matrix_y, trainElements, (m_LOD > 1) ? m_d_values_out + offset : nullptr , &matrix_Lum, bGradientDenoising && *m_denoising_level != 0.0f, m_laplace_csrValA, m_laplace_row_offset, m_laplace_csrColIndA, &m_laplace_num_non_zero_elements, &eventLaplaceReceivedNNZE, &eventLaplaceFinishMatrixBuild, m_denoising_max_iter);
                RUN_CUDA_SYNC()
                // Dispatch the event of finish of executiong
                CUDA_CHECK_THROW(cudaEventRecord(modelsEventsFit[modelID], m_networks[modelID]->getCudaStream()));
                fitWaitingEvents.push_back(modelsEventsFit[modelID]);
                //break;
            }
            offset += mappings_count;
        }
        //break;
    }


RUN_CUDA_SYNC()
//CUDA_CHECK_THROW(cudaEventDestroy(fitEventFinishCounting));


#if 0
    for (uint32_t j = 0; j < m_networks.size(); j++) {
        for (uint32_t i = 0; i < fitWaitingEvents.size(); i++)
        {
            CUDA_CHECK_THROW(cudaStreamWaitEvent(m_networks[j]->getCudaStream(), fitWaitingEvents[i]));
        }
    }
#endif


    if (m_LOD <= 1) {
        m_training_epoch = m_networks[0]->epochs;
        m_training_error = m_networks[0]->lossValue;
        
    }
}



bool Falcor::NNLaplacianCV::predict(
    uint32_t numElements,
    Falcor::Buffer::SharedPtr X,
    Falcor::Buffer::SharedPtr y,
    Falcor::Buffer::SharedPtr mapping_key,
    Falcor::Buffer::SharedPtr mapping_value,
    GpuFence::SharedPtr fence,
    uint64_t fenceValue,
    GpuFence::SharedPtr outfence,
    Falcor::Buffer::SharedPtr act_add_data
    )
{
    
    if (!(bPredictionLOD0 || bPredictingLOD1 || bPredictingLOD2)) {
        // Sync with Rendering backend. 
        if (!bInitCudaFenceOut) {
            cudaExternalSemaphoreHandleDesc d;
            memset(&d, 0, sizeof(cudaExternalSemaphoreHandleDesc));
            d.type = cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeD3D12Fence;
            d.handle.win32.handle = outfence->getSharedApiHandle();
            CUDA_CHECK_THROW(cudaImportExternalSemaphore(&cudaFenceOut, &d));
            bInitCudaFenceOut = true;
        }

        cudaExternalSemaphoreSignalParams a2;
        memset(&a2, 0, sizeof(cudaExternalSemaphoreSignalParams));
        a2.params.fence.value = outfence->beforeExternalSignal();
        CUDA_CHECK_THROW(cudaSignalExternalSemaphoresAsync_v2(&cudaFenceOut, &a2, 1, m_networks[0]->getCudaStream()));
        /*for (uint32_t i = 0; i < m_networks.size(); i++) {
            CUDA_CHECK_THROW(cudaSignalExternalSemaphoresAsync_v2(&cudaFenceOut, &a2, 1, m_networks[i]->getCudaStream()));

        }*/
        //CUDA_CHECK_THROW(cudaSignalExternalSemaphoresAsync_v2(&cudaFenceOut, &a2, 1, 0));


        return false;
    }
        //return;

    

    cudaExternalSemaphoreWaitParams a;
    memset(&a, 0, sizeof(cudaExternalSemaphoreWaitParams));
    a.params.fence.value = fenceValue;
    // Waiting shaders execution. They fill all structure buffers for our networks
    {
        if (!bInitCudaFenceForPredictionData) {
            cudaExternalSemaphoreHandleDesc d;
            memset(&d, 0, sizeof(cudaExternalSemaphoreHandleDesc));
            d.type = cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeD3D12Fence;
            d.handle.win32.handle = fence->getSharedApiHandle();
            CUDA_CHECK_THROW(cudaImportExternalSemaphore(&cudaFenceForPredictionData, &d));
            bInitCudaFenceForPredictionData = true;
        }

        
        for (uint32_t i = 1; i < 1 + bPredictingLOD1 * m_lods_networkds_ids[1].size()+bPredictingLOD2*m_lods_networkds_ids[2].size(); i++) {
            CUDA_CHECK_THROW(cudaWaitExternalSemaphoresAsync(&cudaFenceForPredictionData, &a, 1, m_networks[i]->getCudaStream()));
        }
        if(bPredictingLOD1 || bPredictingLOD2)
            CUDA_CHECK_THROW(cudaWaitExternalSemaphoresAsync(&cudaFenceForPredictionData, &a, 1, m_cuda_sort_stream));
    }


    uint32_t buffer_len = X->getElementCount();

    tcnn::GPUMemory<float> tcnn_X((float*)X->getCUDADeviceAddress(), numElements * m_input_dims);
    tcnn_X.setShared(true);

    tcnn::GPUMemory<float> tcnn_y((float*)y->getCUDADeviceAddress(), numElements * m_output_dims);
    tcnn_y.setShared(true);

    
    tcnn::GPUMemory<float> tcnn_add_data((float*)act_add_data->getCUDADeviceAddress(), numElements * sizeof(Falcor::OutputTransformData)/sizeof(float));
    tcnn_add_data.setShared(true);


    tcnn::GPUMatrix<float> matrix_X(tcnn_X.data(), m_input_dims, numElements);
    tcnn::GPUMatrix<float> matrix_y(tcnn_y.data(), m_output_dims, numElements);
    tcnn::GPUMatrix<float> matrix_OutActAddData(tcnn_add_data.data(), sizeof(Falcor::OutputTransformData) / sizeof(float), numElements);


    CUDA_CHECK_THROW(cudaMemsetAsync(tcnn_y.data(), 0, numElements * m_output_after_transform_dims * sizeof(float), m_networks[0]->getCudaStream()));
    CUDA_CHECK_THROW(cudaWaitExternalSemaphoresAsync(&cudaFenceForPredictionData, &a, 1, m_networks[0]->getCudaStream()));
    CUDA_CHECK_THROW(cudaEventRecord(initFinalBufferEvent, m_networks[0]->getCudaStream()));
    

    
    std::vector<cudaEvent_t> waitingEvents;
    waitingEvents.reserve(modelsEventsPredict.size());

    if (bPredictionLOD0) {
        m_networks[0]->predict(matrix_X, matrix_y, numElements, nullptr, matrix_OutActAddData);
        CUDA_CHECK_THROW(cudaEventRecord(modelsEventsPredict[0], m_networks[0]->getCudaStream()));
        waitingEvents.push_back(modelsEventsPredict[0]);
    }



    if (bPredictingLOD1 || bPredictingLOD2) {
        uint32_t* d_keys_in = (uint32_t*)mapping_key->getCUDADeviceAddress();
        int* d_values_in = (int*)mapping_value->getCUDADeviceAddress();

        int num_items = m_level_size * m_LOD;
        if (m_temp_storage_bytes == 0) {
            CUDA_CHECK_THROW(cudaMalloc(&m_d_values_out, m_level_size * m_LOD * sizeof(int)));
            CUDA_CHECK_THROW(cudaMalloc(&m_d_keys_out, m_level_size * m_LOD * sizeof(uint32_t)));

            CUDA_CHECK_THROW(cudaMalloc(&m_d_values_out_predict, m_level_size * (m_LOD - 1) * sizeof(int)));
            CUDA_CHECK_THROW(cudaMalloc(&m_d_keys_out_predict, m_level_size * (m_LOD - 1) * sizeof(uint32_t)));

            CUDA_CHECK_THROW(cub::DeviceRadixSort::SortPairs(m_d_temp_storage, m_temp_storage_bytes, d_keys_in, m_d_keys_out_predict, d_values_in, m_d_values_out_predict, num_items, 0, sizeof(uint8_t) * (8), m_cuda_sort_stream));
            //m_temp_storage_bytes = m_level_size * m_LOD * sizeof(int) + 128;

            CUDA_CHECK_THROW(cudaMalloc(&m_d_temp_storage, m_temp_storage_bytes));
            CUDA_CHECK_THROW(cudaMalloc(&m_counts_device, (m_networks.size() + 1) * sizeof(int)));

            m_counts.resize(m_networks.size() + 1);
        }
        num_items = numElements * (m_LOD - 1);


        // Sorting async
        CUDA_CHECK_THROW(cub::DeviceRadixSort::SortPairs(m_d_temp_storage, m_temp_storage_bytes, d_keys_in, m_d_keys_out_predict, d_values_in, m_d_values_out_predict, num_items, 0, sizeof(uint8_t) * (8), m_cuda_sort_stream));
        cudaEventRecord(eventFinishSorting, m_cuda_sort_stream);

        // Counting async
        CUDA_CHECK_THROW(cudaMemsetAsync(m_counts_device, 0, (m_networks.size() + 1) * sizeof(int), 0));
        tcnn::linear_kernel(count_idNN, 0, 0, num_items, d_keys_in, m_counts_device);
        //cudaEventRecord(eventFinishCounting, 0);

        // Sync copy count results from a device to the host
        //CUDA_CHECK_THROW(cudaStreamWaitEvent(0, eventFinishCounting));
        CUDA_CHECK_THROW(cudaMemcpy(m_counts.data(), m_counts_device, (m_networks.size() + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        

        //tcnn_y.memset(0);

        //tcnn::linear_kernel(set_zero, 0, m_networks[0]->getCudaStream(), numElements*m_output_dims, tcnn_y.data());
        //matrix_y.initialize_constant(0.0f);

        if (m_counts[0] == 0) {
            uint32_t offset = 0;

            //uint32_t numNetworks = m_networks.size();
            uint32_t numNetworks = 1 + bPredictingLOD1 * m_lods_networkds_ids[1].size()+ bPredictingLOD2 * m_lods_networkds_ids[2].size();
            for (uint32_t nn_indx = 1; nn_indx < numNetworks; nn_indx++)
            {
                uint32_t mappings_count = m_counts[nn_indx];
                if (mappings_count == 0)
                    continue;


                waitingEvents.push_back(modelsEventsPredict[nn_indx]);

                CUDA_CHECK_THROW(cudaStreamWaitEvent(m_networks[nn_indx]->getCudaStream(), initFinalBufferEvent));
                CUDA_CHECK_THROW(cudaStreamWaitEvent(m_networks[nn_indx]->getCudaStream(), eventFinishSorting));


                if (bSeqPredicting) {
                    for (uint32_t i = 0; i < waitingEvents.size(); i++) {
                        CUDA_CHECK_THROW(cudaStreamWaitEvent(m_networks[nn_indx]->getCudaStream(), waitingEvents[i]));
                    }
                }

                m_networks[nn_indx]->predict(matrix_X, matrix_y, mappings_count, m_d_values_out_predict + offset, matrix_OutActAddData);
                CUDA_CHECK_THROW(cudaEventRecord(modelsEventsPredict[nn_indx], m_networks[nn_indx]->getCudaStream()));
                offset += mappings_count;
                //break;
            }
        }
 /*       CUDA_CHECK_THROW(cudaEventDestroy(eventFinishSorting));
        CUDA_CHECK_THROW(cudaEventDestroy(eventFinishCounting));*/
    }
    


    // Wait all executing for future syncronization with render backend
    if (bPredictingLOD1 || bPredictingLOD2) {
        for (uint32_t i = 0; i < waitingEvents.size(); i++) {
            CUDA_CHECK_THROW(cudaStreamWaitEvent(m_networks[0]->getCudaStream(), waitingEvents[i]));
        }
    }


    // Delete all events
    //CUDA_CHECK_THROW(cudaEventDestroy(initFinalBufferEvent));
    //for (uint32_t j = 0; j < waitingEvents.size(); j++) {
    //    CUDA_CHECK_THROW(cudaEventDestroy(waitingEvents[j]));
    //}

    // Sync with Rendering backend. 
    if (!bInitCudaFenceOut) {
        cudaExternalSemaphoreHandleDesc d;
        memset(&d, 0, sizeof(cudaExternalSemaphoreHandleDesc));
        d.type = cudaExternalSemaphoreHandleType::cudaExternalSemaphoreHandleTypeD3D12Fence;
        d.handle.win32.handle = outfence->getSharedApiHandle();
        CUDA_CHECK_THROW(cudaImportExternalSemaphore(&cudaFenceOut, &d));
        bInitCudaFenceOut = true;
    }

    cudaExternalSemaphoreSignalParams a2;
    memset(&a2, 0, sizeof(cudaExternalSemaphoreSignalParams));
    a2.params.fence.value = outfence->beforeExternalSignal();
    CUDA_CHECK_THROW(cudaSignalExternalSemaphoresAsync_v2(&cudaFenceOut, &a2, 1, m_networks[0]->getCudaStream()));

    return true;
}

int32_t Falcor::NNLaplacianCV::getTrainingEpoch() const {
    return m_training_epoch;
}

float Falcor::NNLaplacianCV::getTrainingError() const {
    return m_training_error;
}



void Falcor::NNLaplacianCV::reloadModel() {
    m_input_dims = 0;
    m_output_dims = 0;
    m_networks.clear();
    m_lods_networkds_ids.clear();
    create();
}

bool Falcor::NNLaplacianCV::renderUI(Gui::Widgets& widget)
{

    if (auto nnGroup = widget.group(m_name, true))
    {
        widget.checkbox("Training", bTraining);
        widget.checkbox("Predicting", bPredictionLOD0);

        widget.checkbox("TrainingLOD1", bTrainingLOD1);
        widget.checkbox("PredictingLOD1", bPredictingLOD1);

        widget.checkbox("TrainingLOD2", bTrainingLOD2);
        widget.checkbox("PredictingLOD2", bPredictingLOD2);

        widget.checkbox("Sequantial Training", bSeqTraining);
        widget.checkbox("Sequantial Predicting", bSeqPredicting);
        if (nnGroup.button("Reload model", false))
        {
            reloadModel();
        }

        widget.slider("Min Training Elements", minTrainingElements, (uint32_t)0, (uint32_t)1024);
        
        if (auto nnGroup = widget.group("Crumbs info", false))
        {
            for (uint32_t nn_indx = 0; nn_indx < m_networks.size(); nn_indx++)
            { 
                m_networks[nn_indx]->renderUI(widget);
            }
        }

        if (auto t = widget.group("Gradient Denoising", true)) {
            widget.checkbox("Enable", bGradientDenoising);
            widget.var("Denoising Level", *m_denoising_level, 0.0f, 10.0f, 0.1f);
        }
    }
    return false;
}


