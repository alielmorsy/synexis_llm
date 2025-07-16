#pragma once
#include "sampler/StructParams.h"
#include <functional>

struct TaskParams {
    std::string prompt;
    SamplingParams samplerParams = SamplingParams(); // default value
    bool stream = false;

    int maximumTokens = -1;

    TaskParams() = default;

    TaskParams(std::string prompt, SamplingParams samplerParams = SamplingParams()): prompt(std::move(prompt)),
        samplerParams(std::move(samplerParams)) {
    }

    std::function<void(const std::string &)> on_token = nullptr;
    std::function<void(const std::string &)> on_done = nullptr;
};
