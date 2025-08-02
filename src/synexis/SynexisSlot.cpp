#include "SynexisSlot.h"

#include <stdexcept>


void TaskTokens::add(llama_token token) {
    tokens.push_back(token);
}

void TaskTokens::insert(std::vector<llama_token> &tokens) {
    this->tokens.insert(this->tokens.end(), tokens.begin(), tokens.end());
}

#include <iostream>  // for std::cout
#include <cstring>   // for std::memmove

void TaskTokens::shiftTokens(int n_keep, int n_discard) {
    if (n_discard <= 0 || n_keep < 0) {
        return;
    }

    if (tokens.size() <= n_keep + n_discard) {
        return;
    }

    const size_t src_start = n_keep + n_discard;
    const size_t dest_start = n_keep;
    const size_t copy_count = tokens.size() - src_start;

    std::memmove(&tokens[dest_start], &tokens[src_start], copy_count * sizeof(llama_token));

    tokens.resize(tokens.size() - n_discard);
}


void TaskTokens::parseMtmdChunk(const mtmd_input_chunk *chunk) noexcept {
    auto type = mtmd_input_chunk_get_type(chunk);
    if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        const int n_pos = mtmd_input_chunk_get_n_pos(chunk);
        const size_t start_pos = tokens.size();
        for (int i = 0; i < n_pos; ++i) {
            tokens.emplace_back(LLAMA_TOKEN_NULL);
        }
        mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
        mediaPosition[start_pos] = std::move(new_chunk);
    } else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const llama_token *text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens.emplace_back(text_tokens[i]);
        }
    } else {
        GGML_ABORT("Invalid chunk type");
    }
}

const mtmd::input_chunk_ptr &TaskTokens::find_chunk(llama_pos pos) const {
    auto it = mediaPosition.find(pos);
    if (it != mediaPosition.end()) {
        return it->second;
    }
    throw std::runtime_error("Chunk not found");
}

int32_t TaskTokens::process_chunk(llama_context *ctx, mtmd_context *mctx, llama_pos n_past, int32_t seq_id,
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


void TaskTokens::keepFirst(size_t n) {
    GGML_ASSERT(n <= tokens.size());
    if (hasMtmd) {
        if (n == tokens.size()) {
            return; // nothing to do
        }
        // we throw an error if we try to remove a token in the middle of an image
        // for ex. with input of 5 text tokens and 2 images:
        //    [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
        // n  1   2   3   4   5   6      7      8      9      10
        // allowed to resize      ^                    ^
        // disallowed to resize          ^      ^             ^
        if (n > 0) {
            llama_token last_token = tokens[n - 1];
            // make sure we never remove tokens in the middle of an image
            if (last_token == LLAMA_TOKEN_NULL) {
#ifndef NDEBUG
                find_chunk(n - 1); // will throw an error if the token is not begin-of-chunk
#endif
            }
        }
        // remove all image chunks that are not used anymore
        for (auto it = mediaPosition.begin(); it != mediaPosition.end();) {
            llama_pos pos = it->first;
            if (pos >= (llama_pos) n) {
                it = mediaPosition.erase(it);
            } else {
                ++it;
            }
        }
    }
    tokens.resize(n);
}


bool SynexisSlot::processToken(const llama_vocab *vocab, int32_t id,std::string &token_str) {
    sampled = id;
    if (llama_vocab_is_eog(vocab, id)) {
        return false;
    }

    if (n_decoded >= request->params.maximumTokens) {
        return false;
    }

    for (const auto &stop_word: request->params.stopTokens) {
        if (token_str.find(stop_word) != std::string::npos) {
            return false;
        }
    }

    return true;
}
