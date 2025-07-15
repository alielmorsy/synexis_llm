#pragma once
#include "sampler/StructParams.h"
#include <functional>

struct TaskParams {
    std::string prompt;
    SamplingParams samplerParams;
    bool stream = false;
    std::function<void(const std::string&)> on_token = nullptr;
    std::function<void(const std::string&)> on_done = nullptr;
};
