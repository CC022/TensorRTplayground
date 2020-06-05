#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include "NvInferRuntimeCommon.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"

#include "buffers.h"
#include <cuda_runtime_api.h>

// from samples common.h
inline void setAllTensorScales2(nvinfer1::INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f) {
    for (int i=0; i < network->getNbLayers(); i++) {
        auto layer = network->getLayer(i);
        for (int j=0; j < layer->getNbInputs(); j++) { // ensure all inputs have a scale
            nvinfer1::ITensor* input{layer->getInput(j)};
            if (input != nullptr && !input->dynamicRangeIsSet()) { // optional inputs are from RNN
                input->setDynamicRange(-inScales, inScales);
            }
        }
    }
    for (int i=0; i < network->getNbInputs(); i++) {
        auto layer = network->getLayer(i);
        for (int j=0; j < layer->getNbOutputs(); j++) {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            if (output != nullptr && !output->dynamicRangeIsSet()) {
                if (layer->getType() == nvinfer1::LayerType::kPOOLING) {
                    output->setDynamicRange(-inScales, inScales); // pooling layer must have the same in/out scales
                } else {
                    output->setDynamicRange(-outScales, outScales);
                }
            }
        }
    }
}
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
inline void readPGMFile2(const std::string &fileName, uint8_t *buffer, int inH, int inW) {
    std::ifstream inFile(fileName, std::ifstream::binary);
    assert(inFile.is_open() && "inFile open failed");
    std::string magic, h, w, max;
    inFile >> magic >> h >> w >> max;
    inFile.seekg(1, inFile.cur);
    inFile.read(reinterpret_cast<char*>(buffer), inH * inW);
}


struct trtDeleter {
    template <typename T>
    void operator()(T* object) const {
        if (object) {object->destroy();}
    }
};

class tensorRTLogger: public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) override {
        std::cout << "TRT: " << msg << std::endl;
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
    std::unique_ptr<nvcaffeparser1::IBinaryProtoBlob, trtDeleter> m_MeanBlob; // the mean blob to keep until build is done
    
    void constructNetwork(std::unique_ptr<nvcaffeparser1::ICaffeParser, trtDeleter>& parser, std::unique_ptr<nvinfer1::INetworkDefinition, trtDeleter>& network);
    bool processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const;
    bool verifyOutput(samplesCommon::BufferManager &buffers, const std::string &outputTensorName, int groundTruthDigit) const;
    
public:
    std::string dataDir; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string caffePrototxtFileName;
    std::string caffeWeightsFileName;
    std::string caffeMeanFileName;
    
    // create the network, builder, network engine
    bool build() {
        auto builder = std::unique_ptr<nvinfer1::IBuilder, trtDeleter>(nvinfer1::createInferBuilder(m_trtLogger));
        if (!builder) return false;
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition, trtDeleter>(builder->createNetworkV2(0U));
        if (!network) return false;
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig, trtDeleter>(builder->createBuilderConfig());
        if (!config) return false;
        auto parser = std::unique_ptr<nvcaffeparser1::ICaffeParser, trtDeleter>(nvcaffeparser1::createCaffeParser());
        if (!parser) return false;
        
        constructNetwork(parser, network);
        builder->setMaxBatchSize(batchSize);
        config->setMaxWorkspaceSize(16 * (1 << 20)); // 16 MB
        config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
        if (int8) {config->setFlag(nvinfer1::BuilderFlag::kINT8);}
        if (fp16) {config->setFlag(nvinfer1::BuilderFlag::kFP16);}
        enableDLA2(builder.get(), config.get(), dlaCore);
        m_Engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), trtDeleter());
        if (!m_Engine) {return false;}
        assert(network->getNbInputs() == 1);
        m_InputDims = network->getInput(0)->getDimensions();
        assert(m_InputDims.nbDims == 3);
        return true;
    }
    
    bool infer() {
        samplesCommon::BufferManager buffers(m_Engine, batchSize);
        
        auto context = std::unique_ptr<nvinfer1::IExecutionContext, trtDeleter>(m_Engine->createExecutionContext());
        if (!context) {return false;}
        int digit = 3;
        assert(inputTensorNames.size() == 1);
        if (!processInput(buffers, inputTensorNames[0], digit)) {return false;}
        cudaStream_t stream;
        auto status = cudaStreamCreate(&stream);
        if (status != 0) {std::cerr << "cudaStreamCreate failed\n"; return false;}
        buffers.copyInputToDeviceAsync(stream);
        if (!context->enqueue(batchSize, buffers.getDeviceBindings().data(), stream, nullptr)) {
            return false;
        }
        buffers.copyOutputToHostAsync(stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        assert(outputTensorNames.size() == 1);
        bool outputCorrect = verifyOutput(buffers, outputTensorNames[0], digit);
        
        return outputCorrect;
    }
    
    bool teardown() {
        nvcaffeparser1::shutdownProtobufLibrary();
        return true;
    }
    
};

void myMNISTSample::constructNetwork(std::unique_ptr<nvcaffeparser1::ICaffeParser, trtDeleter> &parser, std::unique_ptr<nvinfer1::INetworkDefinition, trtDeleter> &network) {
    const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor = parser->parse(caffePrototxtFileName.c_str(), caffeWeightsFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);
    
    for (auto & outputTensor: outputTensorNames) {
        network->markOutput(*blobNameToTensor->find(outputTensor.c_str()));
    }
    
    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    m_MeanBlob = std::unique_ptr<nvcaffeparser1::IBinaryProtoBlob, trtDeleter>(parser->parseBinaryProto(caffeMeanFileName.c_str()));
    nvinfer1::Weights meanWeights{nvinfer1::DataType::kFLOAT, m_MeanBlob->getData(), inputDims.d[1] * inputDims.d[2]};
    float maxMean = *std::max_element(static_cast<const float*>(meanWeights.values), static_cast<const float*>(meanWeights.values) + std::accumulate(inputDims.d, inputDims.d + inputDims.nbDims, 1, std::multiplies<int64_t>()));
    auto mean = network->addConstant(nvinfer1::Dims3(1, inputDims.d[1], inputDims.d[2]), meanWeights);
    mean->getOutput(0)->setDynamicRange(-maxMean, maxMean);
    network->getInput(0)->setDynamicRange(-maxMean, maxMean);
    auto meanSub = network->addElementWise(*network->getInput(0), *mean->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
    meanSub->getOutput(0)->setDynamicRange(-maxMean, maxMean);
    network->getLayer(0)->setInput(0, *meanSub->getOutput(0));
    setAllTensorScales2(network.get(), 127.0f, 127.0f);
}

bool myMNISTSample::processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const {
    const int inputH = m_InputDims.d[1];
    const int inputW = m_InputDims.d[2];
    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(dataDir + std::to_string(inputFileIdx) + ".pgm", fileData.data(), inputH, inputW);

    std::cout << "Input image\n";
    for (int i=0; i<inputH*inputW; i++) {
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i+1) % inputW) ? "" : "\n");
    }
    std::cout << std::endl;
    
    float *hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));
    for (int i=0; i < inputH * inputW; i++) {
        hostInputBuffer[i] = float(fileData[i]);
    }
    
    return true;
}

bool myMNISTSample::verifyOutput(samplesCommon::BufferManager &buffers, const std::string &outputTensorName, int groundTruthDigit) const {
    const float* prob = static_cast<const float*>(buffers.getHostBuffer(outputTensorName));
    
    std::cout << "Output:\n";
    float val = 0;
    int idx = 0;
    const int DIGITS = 10;
    for (int i=0; i<DIGITS; i++) {
        if (val < prob[i]) {
            val = prob[i];
            idx = i;
        }
        std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    return (idx == groundTruthDigit);
}

int main(int argc, const char * argv[]) {
    myMNISTSample myMNISTSample;
    myMNISTSample.dataDir = "../data/mnist/";
    myMNISTSample.caffePrototxtFileName = "../data/mnist/mnist.prototxt";
    myMNISTSample.caffeWeightsFileName = "../data/mnist/mnist.caffemodel";
    myMNISTSample.caffeMeanFileName = "../data/mnist/mnist_mean.binaryproto";
    myMNISTSample.inputTensorNames.push_back("data");
    myMNISTSample.outputTensorNames.push_back("prob");
    
    if (!myMNISTSample.build()) {std::cout << "sample build failed.\n";}
    if (!myMNISTSample.infer()) {std::cout << "sample infer failed.\n";}
    if (!myMNISTSample.teardown()) {std::cout << "sample teardown failed.\n";}
    
    return 0;
}