#pragma once

#include <unordered_map>
#include <vector>
#include "llama.h"
#include "mtmd.h"


class TaskTokens {
public:
    TaskTokens() = default;

    TaskTokens(const TaskTokens &) = delete;

    TaskTokens &operator=(const TaskTokens &) = delete;

    explicit TaskTokens(mtmd::input_chunks &mtmd_chunks) {
        hasMtmd = true;
        for (size_t i = 0; i < mtmd_chunks.size(); ++i) {
            parseMtmdChunk(mtmd_chunks[i]);
        }
    }

    explicit TaskTokens(std::vector<llama_token> tokens) : tokens(std::move(tokens)) {
    }


    void add(llama_token int32);

    void insert(std::vector<llama_token> &tokens);

    void shiftTokens(int n_keep, int n_discard);

    TaskTokens(TaskTokens &&) = default;

    TaskTokens &operator=(TaskTokens &&) = default;


    void parseMtmdChunk(const mtmd_input_chunk *chunk) noexcept;

    const std::vector<llama_token> &getTokens() const {
        return tokens;
    };

    size_t size() {
        return tokens.size();
    };

    void keepFirst(size_t n);

    const mtmd::input_chunk_ptr &find_chunk(llama_pos pos) const;

    int32_t process_chunk(
        llama_context *ctx,
        mtmd_context *mctx,
        llama_pos n_past,
        int32_t seq_id,
        llama_pos &n_pos_out);

private:
    bool hasMtmd = false;
    std::vector<llama_token> tokens;
    std::unordered_map<size_t, mtmd::input_chunk_ptr> mediaPosition;
};
