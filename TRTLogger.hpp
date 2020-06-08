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
                    std::cout << "\e[38;5;10mTRT: ";
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
};

#endif /* TRTLogger_hpp */
