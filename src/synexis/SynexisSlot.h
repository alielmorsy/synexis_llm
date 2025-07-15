#pragma once

#include <iostream>
#include <unordered_map>
#include <synexis/sampler/Sampler.h>

#include "mtmd-helper.h"
#include "mtmd.h"


enum SlotState {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED,
    // TODO: this state is only used for setting up the initial prompt processing; maybe merge it with launch_slot_with_task in the future
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};


class TaskTokens {
public:
    TaskTokens() = default;

    TaskTokens(const TaskTokens &) = delete;

    TaskTokens &operator=(const TaskTokens &) = delete;

    void clear() {
        tokens.clear();
    }

    void add(llama_token int32);

    void TaskTokens::insert(std::vector<llama_token> &tokens);

    void shiftTokens(int n_keep, int n_discard);

    TaskTokens(TaskTokens &&) = default;

    TaskTokens &operator=(TaskTokens&&) = default;


    explicit TaskTokens(mtmd::input_chunks &mtmd_chunks) {
        hasMtmd = true;
        for (size_t i = 0; i < mtmd_chunks.size(); ++i) {
            parseMtmdChunk(mtmd_chunks[i]);
        }
    }

    explicit TaskTokens(std::vector<llama_token> tokens) : tokens(std::move(tokens)) {
    }

    void parseMtmdChunk(const mtmd_input_chunk *chunk) noexcept;

    const std::vector<llama_token> &getTokens() const {
        return tokens;
    }

    size_t size() {
        return tokens.size();
    }

    void keepFirst(size_t n);

    const mtmd::input_chunk_ptr &TaskTokens::find_chunk(llama_pos pos) const;

    int32_t process_chunk(
        llama_context *ctx,
        mtmd_context *mctx,
        llama_pos n_past,
        int32_t seq_id,
        llama_pos &n_pos_out) {
        auto &chunk = find_chunk(n_past);
        const char *name = mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_IMAGE
                               ? "image"
                               : "audio";
        std::cout << "Processing Image" << std::endl;
        int32_t n_batch = llama_n_batch(ctx);
        int64_t t0 = ggml_time_ms();
        llama_pos new_n_past = n_past;
        int32_t result = mtmd_helper_eval_chunk_single(mctx, ctx,
                                                       chunk.get(),
                                                       n_past,
                                                       seq_id,
                                                       n_batch,
                                                       true, // logits last
                                                       &new_n_past);
        std::printf("%s processed in %" PRId64 " ms\n", name, ggml_time_ms() - t0);
        if (result != 0) {
            std::printf("mtmd_helper_eval failed with status %d\n", result);
            n_pos_out = n_past;
            return result;
        }
        n_pos_out = new_n_past;
        return 0;
    }

private:
    bool hasMtmd = false;
    std::vector<llama_token> tokens;
    std::unordered_map<size_t, mtmd::input_chunk_ptr> mediaPosition;
};

struct SynexisSlot {
    int id;

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

    void reset() {
        n_past = 0;
        state = SLOT_STATE_IDLE;
    }

    bool idle() const {
        return state == SLOT_STATE_IDLE;
    }

    bool canBeBatchedWith(SynexisSlot *other) const {
        return state == other->state;
    }

    void release() {
        state = SLOT_STATE_IDLE;
        std::cout << "Generated Text: " << generatedText << std::endl;
    }

    std::string generate_now(std::string prompt);;

    bool processToken(SynexisSlot *slot, const llama_vocab *vocab, int32_t id);;

    size_t promptSize() {
        return tokens.size();
    }
};
