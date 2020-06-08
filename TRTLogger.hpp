// Simple TensorRT logger class

#ifndef TRTLogger_hpp
#define TRTLogger_hpp

#include "NvInfer.h"
#include <iostream>

class tensorRTLogger: public nvinfer1::ILogger {
    bool verboseLevel = 3;
    
public:
    void log(Severity severity, const char* msg) override {
        if (severity < verboseLevel) {
            switch (severity) {
                case kINTERNAL_ERROR:
                    std::cerr << "\u001b[38;5;9TRTInternalError: ";
                    break;
                case kERROR:
                    std::cerr << "\u001b[38;5;9TRTError: ";
                    break;
                case kWARNING:
                    std::cerr << "\u001b[38;5;11TRTWarning: ";
                    break;
                default:
                    std::cerr << "\u001b[38;5;10TRT: ";
                    break;
            }
            std::cerr << msg << "\u001b[0m\n";
        }
    }
};

#endif /* TRTLogger_hpp */
