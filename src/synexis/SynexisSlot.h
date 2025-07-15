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

    std::string result;

    SynexisSlot() = default;

    SynexisSlot(SynexisSlot &&) = default;

    SynexisSlot &operator=(SynexisSlot &&) = default;

    void reset() {
        n_past = 0;
        state = SLOT_STATE_IDLE;
        request.reset();
        generatedText.clear();
        result.clear();
    }

    bool idle() const {
        return state == SLOT_STATE_IDLE;
    }

    bool canBeBatchedWith(SynexisSlot *other) const {
        return state == other->state;
    }

    void release() {
        if (request) {
            request->promise.set_value(generatedText);
            if (request->params.on_done) {
                request->params.on_done(generatedText);
            }
        }
        reset();
    }

    std::string generate_now(std::string prompt);;

    bool processToken(SynexisSlot *slot, const llama_vocab *vocab, int32_t id);;

    size_t promptSize() {
        return tokens.size();
    }
};
