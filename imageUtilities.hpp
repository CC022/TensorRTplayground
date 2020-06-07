// Image utilities

#ifndef imageUtilities_hpp
#define imageUtilities_hpp

#include <iostream>
#include <string>
#include <fstream>

/** Read a RGB PPM file into a buffer using CHW memory layout.
The function will allocate the memory space for buffer, and the
user needs to free the buffer.
*/
bool readPPMToBuffer(const std::string &imagePath, char* &buffer) {
    std::ifstream imageFile(imagePath);
    if (!imageFile.is_open()) {
        std::cerr << "failed to open image file: " << imagePath << "\n";
        return false;
    }
    std::string magic;
    int H, W, maxValue, R, G, B, pixelCount = 0;
    imageFile >> magic >> H >> W >> maxValue;
    size_t bufferSize = H * W * 3;
    buffer = (char*)malloc(bufferSize);
    if (buffer == nullptr) {
        std::cerr << "malloc failed\n";
        return false;
    }
    while (imageFile >> R >> G >> B) {
        buffer[pixelCount] = R;
        buffer[pixelCount + H * W] = G;
        buffer[pixelCount + H * W * 2] = B;
        pixelCount++;
    }
    return true;
}

/** Write the content of a RGB planer uint8 image buffer to a PPM file.
*/
bool writeBufferToPPM(const std::string &filePath, const char* buffer, const int width, const int height) {
    std::ofstream imageFile(filePath);
    if (!imageFile.is_open()) {
        std::cerr << "Creating file " << filePath << " failed\n";
        return false;
    }
    imageFile << "P3\n" << width << " " << height << "\n255\n";
    for (int y=0; y < height; y++) {
        for (int x=0; x < width; x++) {
            imageFile << std::to_string((uint8_t)buffer[y*width + x])<< " " << std::to_string((uint8_t)buffer[y*width + x + width * height]) << " " << std::to_string((uint8_t)buffer[y*width + x + width * height * 2]) << "\n";
        }
    }
    return true;
}

#endif /* imageUtilities_hpp */
