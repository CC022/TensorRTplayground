// Simple TensorRT logger class

#ifndef TRTLogger_hpp
#define TRTLogger_hpp

#include "NvInfer.h"
#include <iostream>

class tensorRTLogger: public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) override {
        std::cout << "TRT: " << msg << std::endl;
    }
    // NeoLoggerTest
};

#endif /* TRTLogger_hpp */
