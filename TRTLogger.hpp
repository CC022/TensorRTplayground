// Simple TensorRT logger class

#ifndef TRTLogger_hpp
#define TRTLogger_hpp

#include "NvInfer.h"
#include <iostream>

class tensorRTLogger: public nvinfer1::ILogger {
    int verboseLevel = 3;
    
public:
    void log(Severity severity, const char* msg) override {
        if (int(severity) <= verboseLevel) {
            switch (severity) {
                case Severity::kINTERNAL_ERROR:
                    std::cout << "\e[38;5;9mTRTInternalError: ";
                    break;
                case Severity::kERROR:
                    std::cout << "\e[38;5;9mTRTError: ";
                    break;
                case Severity::kWARNING:
                    std::cout << "\e[38;5;11mTRTWarning: ";
                    break;
                default:
                    std::cout << "\e[38;5;156mTRT: ";
                    break;
            }
            std::cout << msg << "\e[0m\n";
        }
    }
    
    void setVerboseLevel(int level) {
        if (level >= 0 && level <= 4) {
            verboseLevel = level;
        } else {
            std::cout << "setVerboseLevel: value out of range\n";
        }
    }
    
    void printDims(nvinfer1::Dims Dims) {
        std::cout << "Dims: [";
        for (int i=0; i<Dims.nbDims; i++) {
            std::cout << Dims.d[i] << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    void printTensorName(std::vector<std::string> names) {
        for (std::string &name : names) {
            std::cout << "Tensor name: " << name << std::endl;
        }
    }
    
    void printEngine(const ICudaEngine *engine) {
        std::cout << "Number of binding indices " << engine->getNbBindings() << std::endl;
        std::cout << "Max Batch Size " << engine->getMaxBatchSize() << std::endl;
        std::cout << "Number of layers in the network " << engine->getNbLayers() << std::endl;
        std::cout << "Device memory required by an execution context " << engine->getDeviceMemorySize() << std::endl;
        std::cout << "Network name " << engine->getName() << std::endl;
    }
    
};

#endif /* TRTLogger_hpp */
