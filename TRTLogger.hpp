// Simple TensorRT logger class

#ifndef TRTLogger_hpp
#define TRTLogger_hpp

#include "NvInfer.h"
#include <iostream>

class tensorRTLogger: public nvinfer1::ILogger {
    int verboseLevel = 3;
    
public:
    void log(Severity severity, const char* msg) override {
        if (int(severity) < verboseLevel) {
            switch (severity) {
                case Severity::kINTERNAL_ERROR:
                    std::cout << "\u001b[38;5;9TRTInternalError: ";
                    break;
                case Severity::kERROR:
                    std::cout << "\u001b[38;5;9TRTError: ";
                    break;
                case Severity::kWARNING:
                    std::cout << "\u001b[38;5;11TRTWarning: ";
                    break;
                default:
                    std::cout << "\u001b[38;5;10TRT: ";
                    break;
            }
            std::cout << msg << "\u001b[0m\n";
        }
    }
};

#endif /* TRTLogger_hpp */
