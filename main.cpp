#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <thread>
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "buffers.h"
#include "TRTLogger.hpp"
#include <cuda_runtime_api.h>

// TODO: yolo v3

// from samples common.h
inline void enableDLA2(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig *config, int useDLACore, bool allowGPUFallback = true) {
    if (useDLACore >= 0) {
        if (builder->getNbDLACores() == 0) {
            std::cerr << "Error: trying to use DLA core " << useDLACore << " on a platform with no DLA core\n";
            assert(false);
        }
    }
    if (allowGPUFallback) {config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);}
    if (!config->getFlag(nvinfer1::BuilderFlag::kINT8)) {config->setFlag(nvinfer1::BuilderFlag::kFP16);}
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setDLACore(useDLACore);
    config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
}

struct trtDeleter {
    template <typename T>
    void operator()(T* object) const {
        if (object) {object->destroy();}
    }
};

class myMNISTSample {
    tensorRTLogger m_trtLogger = tensorRTLogger();
    std::shared_ptr<nvinfer1::ICudaEngine> m_Engine = nullptr;
    int batchSize = 1; // num of inputs in a batch
    int dlaCore = -1;
    bool int8 = false;
    bool fp16 = false;
    nvinfer1::Dims m_InputDims;
    
    bool processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const;
    bool verifyOutput(const samplesCommon::BufferManager &buffers, const std::string &outputTensorName, int groundTruthDigit) const;
    
public:
    std::string dataDir; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string onnxFilePath;
    
    // create the network, builder, network engine
    bool build() {
        m_trtLogger.setVerboseLevel(3);
        auto builder = std::unique_ptr<nvinfer1::IBuilder, trtDeleter>(nvinfer1::createInferBuilder(m_trtLogger));
        if (!builder) return false;
        const auto explicitBatch = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition, trtDeleter>(builder->createNetworkV2(explicitBatch));
        if (!network) return false;
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig, trtDeleter>(builder->createBuilderConfig());
        if (!config) return false;
        auto parser = std::unique_ptr<nvonnxparser::IParser, trtDeleter>(nvonnxparser::createParser(*network, m_trtLogger));
        if (!parser) return false;
        // construct Network
        auto parsed = parser->parseFromFile(onnxFilePath.c_str(), 4);
        if (!parsed) return false;
        builder->setMaxBatchSize(batchSize);
        config->setMaxWorkspaceSize(1 << 28); // 256 MB
        // config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        // config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
        if (int8) {config->setFlag(nvinfer1::BuilderFlag::kINT8);}
        if (fp16) {config->setFlag(nvinfer1::BuilderFlag::kFP16);}
        enableDLA2(builder.get(), config.get(), dlaCore);
        m_Engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), trtDeleter());
        if (!m_Engine) {return false;}
        assert(network->getNbInputs() == 1);
        m_InputDims = network->getInput(0)->getDimensions();
        assert(m_InputDims.nbDims == 4);
        return true;
    }
    
    bool infer() {
        samplesCommon::BufferManager buffers(m_Engine, batchSize);
        
        auto context = std::unique_ptr<nvinfer1::IExecutionContext, trtDeleter>(m_Engine->createExecutionContext());
        if (!context) {return false;}
        int digit = 3;
        assert(inputTensorNames.size() == 1);
        if (!processInput(buffers, inputTensorNames[0], digit)) {return false;}
        buffers.copyInputToDevice();
        bool status = context->executeV2(buffers.getDeviceBindings().data());
        if (!status) return false;
        buffers.copyOutputToHost();
        bool outputCorrect = verifyOutput(buffers, outputTensorNames[0], digit);
        
        return outputCorrect;
    }
    
};

bool myMNISTSample::processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const {

    return true;
}

bool myMNISTSample::verifyOutput(const samplesCommon::BufferManager &buffers, const std::string &outputTensorName, int groundTruthDigit) const {
    return false;
}

int main(int argc, const char * argv[]) {
    myMNISTSample myMNISTSample;
    myMNISTSample.dataDir = "../data/yolo/";
    myMNISTSample.onnxFilePath = "yolov3.onnx";
    myMNISTSample.inputTensorNames.push_back("000_net");
    myMNISTSample.outputTensorNames.push_back("082_convolutional");
    myMNISTSample.outputTensorNames.push_back("094_convolutional");
    myMNISTSample.outputTensorNames.push_back("106_convolutional");
    
    if (!myMNISTSample.build()) {std::cout << "sample build failed.\n";}
    if (!myMNISTSample.infer()) {std::cout << "sample infer failed.\n";}
    return 0;
}
