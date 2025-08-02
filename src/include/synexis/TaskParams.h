#pragma once
#include "sampler/StructParams.h"
#include <functional>

struct MediaDataView {
    const uint8_t *data = nullptr;
    size_t size = 0;
};

struct TaskParams {
    std::string prompt;
    SamplingParams samplerParams = SamplingParams(); // default value
    bool stream = false;

    int maximumTokens = -1;

    std::vector<MediaDataView> media;

    std::vector<std::string> stopTokens;

    TaskParams() = default;

    TaskParams(std::string prompt, SamplingParams samplerParams = SamplingParams(), int maximumTokens = -1,
               std::vector<std::string> stopTokens={}): prompt(std::move(prompt)),
                                                     samplerParams(std::move(samplerParams)),
                                                     maximumTokens(maximumTokens),
                                                     stopTokens(std::move(stopTokens)) {
    }

    void addMedia(const std::string_view &viewData) {
        MediaDataView mediaData = {reinterpret_cast<const uint8_t *>(viewData.data()), viewData.size()};
        media.emplace_back(mediaData);
    }

    std::function<void(const std::string &)> on_token = nullptr;
    std::function<void(const std::string &)> on_done = nullptr;
    std::function<void(const std::string &)> on_error = nullptr;
};
