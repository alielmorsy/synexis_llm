#pragma once
#include <string>

struct SynexisArguments {
    std::string modelPath;
    std::string modelProjectorPath;

    int numberOfGpuLayers = 999;
    int numberOfThreads = 4;
    bool use_mmap = true;

    int n_ctx = 4096;
    int n_batch = 2048;
    int n_keep = 512;
    int n_discard = 0;

    int n_slots = 8;

    explicit SynexisArguments(std::string modelPath): modelPath(std::move(modelPath)) {
    }
};
