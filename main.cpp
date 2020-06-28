#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <experimental/filesystem>
#include <thread>
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "buffers.h"
#include "TRTLogger.hpp"
#include <cuda_runtime_api.h>

// TODO: yolo v3

struct trtDeleter {
    template <typename T>
    void operator()(T* object) const {if (object) {object->destroy();}}
};

class myMNISTSample {
    int batchSize = 1; // num of inputs in a batch
    int dlaCore = -1;
    bool int8 = false;
    bool fp16 = false;
    nvinfer1::Dims m_InputDims;
    
    bool processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const;
    bool verifyOutput(const samplesCommon::BufferManager &buffers, const std::string &outputTensorName, int groundTruthDigit) const;
    
public:
    tensorRTLogger m_trtLogger = tensorRTLogger();
    std::shared_ptr<nvinfer1::ICudaEngine> m_Engine = nullptr;
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
//        enableDLA2(builder.get(), config.get(), dlaCore);
        m_trtLogger.printDims(network->getInput(0)->getDimensions());
        network->getInput(0)->setDimensions(nvinfer1::Dims{4,1,3,608,608});
        m_trtLogger.printDims(network->getInput(0)->getDimensions());

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
        assert(inputTensorNames.size() == 1);
        // TODO: process input and put into buffer
        buffers.copyInputToDevice();
        bool status = context->executeV2(buffers.getDeviceBindings().data());
        if (!status) return false;
        buffers.copyOutputToHost();
//        bool outputCorrect = verifyOutput(buffers, outputTensorNames[0], digit);
        
        return true;
    }
    
    bool serializeEngine(nvinfer1::ICudaEngine *engine, std::string filePath) {
        auto serializedEngine = std::unique_ptr<IHostMemory, trtDeleter>(engine->serialize());
        ofstream engineFile(filePath);
        assert(engineFile.is_open());
        engineFile.write((const char*) serializedEngine->data(), serializedEngine->size());
        return !engineFile.fail();
    }
    
    bool loadEngine(std::string filePath) {
        auto runtime = std::unique_ptr<IRuntime, trtDeleter>(createInferRuntime(m_trtLogger));
        ifstream engineFile(filePath);
        assert(engineFile.is_open());
        size_t fileSize = std::experimental::filesystem::file_size(filePath);
        std::unique_ptr<char []> fileBuffer(new char[fileSize]);
        engineFile.read(fileBuffer.get(), fileSize);
        m_Engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(fileBuffer.get(), fileSize, nullptr), trtDeleter());
        return m_Engine != nullptr;
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
//    if (!myMNISTSample.serializeEngine(myMNISTSample.m_Engine.get(), "yolov3Engine.trt")) {std::cout << "serialize engine failed.\n";};
//    if (!myMNISTSample.loadEngine("yolov3Engine.trt")) {std::cout << "load engine failed.\n";}
    // preprocess image
     if (!myMNISTSample.infer()) {std::cout << "sample infer failed.\n";}
    // postprocess image
    return 0;
}
