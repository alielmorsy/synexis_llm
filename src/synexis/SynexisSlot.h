#pragma once

#include <iostream>
#include <unordered_map>
#include <memory>

#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampler/Sampler.h"
#include "TaskTokens.h"
#include "Request.h"

enum SlotState {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED,
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct SynexisSlot {
    int id;
    std::unique_ptr<Request> request;

    llama_context *ctx = nullptr;

    SlotState state = SLOT_STATE_IDLE;
    size_t index = 0;
    int n_past;
    std::string generatedText;
    SynexisSampler *sampler;
    TaskTokens tokens, cacheTokens;
    bool truncated = false;
    int32_t i_batch;

    llama_token sampled;
    int32_t n_prompt_tokens_processed;
    int n_decoded;

    bool reuse = false;

    SynexisSlot() = default;

    SynexisSlot(SynexisSlot &&) = default;

    SynexisSlot &operator=(SynexisSlot &&) = default;

    void reset(bool error = true) {
        std::cout << "I had to reset" << std::endl;
        if (error && request && request->params.on_error) {
            request->params.on_error("Force reset from the model");
            try {
                throw std::runtime_error("Failed to generate from model");
            } catch (...) {
                request->promise.set_exception(std::current_exception());
            }
        }
        n_past = 0;
        n_prompt_tokens_processed = 0;
        n_decoded = 0;
        state = SLOT_STATE_IDLE;
        request.reset();

        generatedText.clear();
        sampler->reset();
        reuse = true;
    }

    bool idle() const {
        return state == SLOT_STATE_IDLE;
    }

    bool canBeBatchedWith(SynexisSlot *other) const {
        return state == other->state;
    }

    void release() {
        if (request) {
            reuse = true;
            request->promise.set_value(generatedText);
            if (request->params.on_done) {
                request->params.on_done(generatedText);
            }
        }
        reset(false);
    }


    bool processToken(const llama_vocab *vocab, int32_t id, std::string &token_str);

    size_t promptSize() {
        return tokens.size();
    }
};
