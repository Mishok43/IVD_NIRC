#pragma once

#include "FalcorCUDA.h"
#include "Falcor.h"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/object.h>

#include "json/json.hpp"

#include <memory>
#include "NNLaplacianDataTypes.h"


namespace tcnn {
    template<typename, typename, typename>
    class Trainer;

    template<typename>
    class Loss;

    template<typename>
    class Optimizer;

    template<typename>
    class NetworkWithInputEncoding;

    template <typename T>
    class Encoding;
};

using precision_t = tcnn::network_precision_t;

struct WeightingData {
    bool distance;
    float dist_sd;
    bool normal;
    float normal_power;
    float normal_min;
    bool luminance;
    float luminance_sd;
    float luminance_min;
};

namespace Falcor
{
    class NNLaplacianPart
    {
    public:
        NNLaplacianPart(
            cudaStream_t training_stream,
            std::string nn_name, 
            std::string& config_path, 
            float3 bbStart, 
            float3 bbEnd,
            uint lod,
            bool bSeparateInput
        );

        void updateBuffers(uint32_t nElems, uint32_t epoch = 1);
        void updateParams(std::string& config_path);
        float fit(
            tcnn::NeuralInputData<float>& X,
            tcnn::GPUMatrix<float>& y,
            uint32_t nElements,
            int* mapping,
            tcnn::GPUMatrix<float>* lum,
            bool bGradientDiffusion = false,
            float* laplace_csrValA = nullptr,
            int* laplace_csrRowPtrA = nullptr,
            int* laplace_csrColIndA = nullptr,
            uint32_t* numNonZeroElements = nullptr,
            cudaEvent_t* eventLaplaceFinishMatrixBuild = nullptr,
            cudaEvent_t* eventLaplaceReceivedNNZE = nullptr,
            uint32_t denoising_max_iter = 1000,
            float diffusion_gradients_blending = 0.0f,
            bool bGraph = false
        );
        void predict(
            tcnn::NeuralInputData<float>& X,
            tcnn::GPUMatrix<float>& y, 
            uint32_t nElements,
            int* mapping,
            char* thp = nullptr,
            char* pixel_m = nullptr
        );
        void getDims(uint32_t& input_dims, uint32_t& output_dims, uint32_t& outputAfterTransformDims_) { input_dims = inputDims; output_dims = outputDims; outputAfterTransformDims_ = outputAfterTransformDims; };

        bool renderUI(Gui::Widgets& widget);

        cudaStream_t getCudaStream() {
            return trainingStream;
        }

        float lossValue{ 0 };
        uint32_t epochs;

        std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network;
        uint32_t numStepsPerFrame = 2;
    private:
        std::string name;
        float* bBStart;
        float* bBEnd;
        
        bool m_separate_input;
        uint32_t numTrainingElements=0;
        uint32_t numPredictionElements= 0;
        uint32_t inputDims;
        uint32_t outputDims;
        uint32_t outputAfterTransformDims;

        
        uint32_t mLOD;
        uint32_t batchSize = 16384;
        bool bPredict=true;
        bool bTrain= true;
        bool bLearningRateDecay;
        float initialLR;
        
        cudaStream_t trainingStream;
        tcnn::default_rng_t rng{1337};

        std::shared_ptr<tcnn::Loss<precision_t>> loss;
        std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer;
        std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> trainer;
        tcnn::GPUMemory<int> indices;
    };

    struct NeuralInputDataFalcor {
    public:
        tcnn::NeuralInputData<float> generateInputData() {
            tcnn::NeuralInputData<float> res;

            for (uint32_t i = 0; i < data.size(); i++) {
                uint32_t dims = data[i]->getElementSize() / sizeof(float);

                tcnn::GPUMemory<float> tcnn_X((float*)data[i]->getCUDADeviceAddress(), numElementsPerChannel[i] * dims);
                tcnn_X.setShared(true);

                tcnn::GPUMatrix<float> matrix_X(tcnn_X.data(), dims, numElementsPerChannel[i]);
                int* map = (mappings[i] == nullptr) ? nullptr : (int*)mappings[i]->getCUDADeviceAddress();
                res.addInput(matrix_X, map);
            }
           
            return res;
        }

        void addInput(Buffer::SharedPtr elements, Buffer::SharedPtr mapping = nullptr, uint32_t numElements = 0) {
            data.push_back(elements);
            mappings.push_back(mapping);
            
            if (numElements != 0)
                numElementsPerChannel.push_back(numElements);
            else
                numElementsPerChannel.push_back(elements->getElementCount());
        }

        uint32_t num_channels() const {
            return data.size();
        }

        std::vector<Buffer::SharedPtr> data;
        std::vector<uint32_t> numElementsPerChannel;
        std::vector<Buffer::SharedPtr> mappings;
    };

    class NNLaplacian
    {
    public:

        NNLaplacian(
            std::string nn_name, 
            std::vector<std::string> config_paths,
            const std::string& pyramid_config_path,
            uint32_t level_size,
            uint32_t LOD,
            const AABB bBox,
            float* denoising_level,
            bool bSeparateInput = true
        );

        void create();
        void updateParams(const std::string& config_path);
        void fit(
            uint32_t numElements,
            NeuralInputDataFalcor& X,
            Buffer::SharedPtr y,
            GpuFence::SharedPtr fence,
            uint64_t fenceValue,
            bool bGraph=false
        );

        void accumulatedRadianceForEstimatingResidualError(
            uint32_t numTrainingElements,
            Falcor::Buffer::SharedPtr predColorIn,
            Falcor::Buffer::SharedPtr predColorOut,
            Falcor::Buffer::SharedPtr samplesMapping,
            Falcor::Buffer::SharedPtr samplesThroughput,
            float outScale = 1.0f
        );

        void accumulatedRadianceOnlyPart(
            uint32_t numPredictedElements,
            uint32_t numTrainingElements,
            Falcor::Buffer::SharedPtr predColorIn,
            Falcor::Buffer::SharedPtr predColorOut,
            Falcor::Buffer::SharedPtr samplesMapping,
            Falcor::Buffer::SharedPtr samplesThroughput,
            float outScale = 1.0f,
            bool bResetBuffer = true
        );

        void transformNetworkOutput(
            uint32_t numPredictedElements,
            uint32_t numTrainingElements,
            Falcor::Buffer::SharedPtr predColorIn,
            Falcor::Buffer::SharedPtr predColorOut,
            Falcor::Buffer::SharedPtr samplesThroughput,
            cudaEvent_t eventSync,
            float outScale = 1.0f,
            bool bResetBuffer = true,
            bool bClamp = true
        );

        void accumulatedRadianceForSelfLearning(
            uint32_t numPredictedElements,
            uint32_t numTrainingElements,
            Falcor::Buffer::SharedPtr predColorIn,
            Falcor::Buffer::SharedPtr predColorOut,
            Falcor::Buffer::SharedPtr samplesMapping,
            Falcor::Buffer::SharedPtr samplesThroughput,
            float outScale = 1.0f,
            bool bResetBuffer = true
        );

        void syncEvent(cudaEvent_t event) {
            CUDA_CHECK_THROW(cudaStreamWaitEvent(m_networks[0]->getCudaStream(), event));
        }

        void pushEvent(cudaEvent_t event) {
            if (event)
                CUDA_CHECK_THROW(cudaEventRecord(event, m_networks[0]->getCudaStream()));
        }

        void assemblyTrainingData(
            uint32_t numElements,
            Falcor::Buffer::SharedPtr predColorIn,
            Falcor::Buffer::SharedPtr trainSamples,
            Falcor::Buffer::SharedPtr lightPaths,
            Falcor::Buffer::SharedPtr output,
            Falcor::Buffer::SharedPtr residualError = nullptr,
            Falcor::Buffer::SharedPtr varianceLossSum = nullptr,
            Falcor::Buffer::SharedPtr varianceLossMapping = nullptr,
            Falcor::Buffer::SharedPtr estimations = nullptr,
            Falcor::Buffer::SharedPtr onlyPart = nullptr,
                
            float outScale = 1.0f,
            GpuFence::SharedPtr fence = nullptr,
            uint64_t fenceValue = 0
        );

        
        void build_laplace_operator(
            uint32_t numElements,
            Buffer::SharedPtr mapping_key,
            Buffer::SharedPtr mapping_value,
            Buffer::SharedPtr positions,
            Buffer::SharedPtr normals,
            Buffer::SharedPtr train_radiance,
            GpuFence::SharedPtr fence,
            uint64_t fenceValue

        );

        bool predict(
            uint32_t numElements,
            NeuralInputDataFalcor& X,
            Buffer::SharedPtr y,
            GpuFence::SharedPtr fence,
            uint64_t fenceValue,
            GpuFence::SharedPtr outfence,
            Falcor::Buffer::SharedPtr thp = nullptr,
            Falcor::Buffer::SharedPtr pixel_mapping = nullptr,
            bool bForcePredict=false
        );
        bool renderUI(Gui::Widgets& widget);
        bool getTrainFlag() { return m_train_flag && (bTraining); };

        void reloadModel();

        float getTrainingError() const;
        int32_t getTrainingEpoch() const;
        

        float getOutputScale() const {
            return mOutputScale;
        }

        void enableTraining(bool bEnable);
        void enablePrediction(bool bEnable);

        float* m_denoising_level;;
        bool bGradientDenoising;

        float m_loss = 0;
        float m_loss_ema = 0;
        uint32_t mTrainingStep = 0;
        bool m_separate_input = false;
    private:
  

        std::string m_name;
        bool m_train_flag {true};

        uint32_t m_input_dims = 0;
        uint32_t m_output_dims = 0;
        uint32_t m_output_after_transform_dims = 0;
        std::vector<uint32_t> lodSizes;
        uint32_t m_LOD;
        uint32_t m_level_size;
        size_t m_temp_storage_bytes = 0;
        
        std::vector<std::string> m_config_paths;
        std::string m_pyramid_config_path;
        AABB m_bBox;

        float* m_X;
        float* m_y;
        float* m_mapping;

        float mOutputScale = 1.0f;
        uint32_t*m_d_keys_out{nullptr};
        int  *m_d_values_out{nullptr};
        void *m_d_temp_storage{nullptr};

        uint32_t* m_d_keys_out_predict{ nullptr };
        int* m_d_values_out_predict{ nullptr };

        int  *m_counts_device{nullptr};
        int* m_counts_device_predict{ nullptr };
        int* m_counts_device_pred{ nullptr };

        std::vector<int> m_counts;
        bool bInitCudaFenceForAssembly = false;
        bool bInitCudaFenceForPredictionData = false;
        bool bInitCudaFenceForTrainingData = false;
        
        bool bInitCudaFenceOut = false;

        std::vector<cudaEvent_t> fitWaitingEvents;


        cudaStream_t m_laplace_stream;
        cudaEvent_t eventLaplaceFinishSorting;
        cudaEvent_t eventLaplaceFinishMatrixBuild;
        cudaEvent_t eventLaplaceReceivedNNZE;
        tcnn::CudaGraph m_diffuse_graph;
        int* m_laplace_keys_out{ nullptr };
        int* m_laplace_values_out{ nullptr };
        void* m_laplace_sort_temp_storage{ nullptr };
        void* m_laplace_prefixsum_temp_storage{ nullptr };
        void* m_laplace_prefixsum_temp_storage_2{ nullptr };
        int* m_laplace_mapper{ nullptr };
        int* m_laplace_counter{ nullptr };
        int* m_laplace_elements_per_row{ nullptr };
        int* m_laplace_row_offset{ nullptr };
        int* m_laplace_cached_cells_mask{ nullptr };
        float* m_laplace_csrValA{ nullptr };
        int* m_laplace_csrColIndA{ nullptr };
        uint32_t m_laplace_num_non_zero_elements;

        size_t m_laplace_sort_temp_storage_size = 0;
        size_t m_laplace_prefixsum_temp_storage_size = 0;
        size_t m_laplace_prefixsum_temp_storage_size_2 = 0;


        cudaEvent_t fitEventFinishSorting;
        cudaEvent_t fitEventFinishCounting;
        cudaEvent_t initFinalBufferEvent;
        cudaEvent_t eventFinishSorting;
        cudaEvent_t eventFinishCounting;

        uint32_t gradientHashGridRes;
        uint32_t gradientHashGridSize;
        WeightingData m_diffusion_weights;
        
        float m_diffusion_gradients_blending;
        uint32_t m_denoising_max_iter;

        cudaExternalSemaphore_t cudaFenceForAssembly;
        cudaExternalSemaphore_t cudaFenceForPredictionData;
        cudaExternalSemaphore_t cudaFenceForTrainingData;
        
        cudaExternalSemaphore_t cudaFenceOut;
        cudaStream_t m_cuda_sort_stream;
        cudaStream_t m_cuda_count_stream;


        std::vector<std::shared_ptr<NNLaplacianPart>> m_networks;
        std::vector<std::vector<uint32_t>> m_lods_networkds_ids;
        bool bTraining = true;
        bool bPredictionLOD0 = true;
        
        bool bSeqTraining = false;
        bool bSeqPredicting = false;
        bool bFusedOutTransform = true;
        uint32_t minTrainingElements = 128;
        std::vector<cudaEvent_t> modelsEventsFit;
        std::vector<cudaEvent_t> modelsEventsPredict;

        tcnn::GPUMemory<float> lumCache;

        float m_training_error;
        int32_t m_training_epoch;
        int m_num_grad_steps = 1;
    };

}
