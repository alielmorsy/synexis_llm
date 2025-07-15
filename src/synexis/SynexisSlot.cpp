#include "SynexisSlot.h"

#include <stdexcept>


void TaskTokens::add(llama_token token) {
    tokens.push_back(token);
}

void TaskTokens::insert(std::vector<llama_token> &tokens) {
    this->tokens.insert(this->tokens.end(), tokens.begin(), tokens.end());
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
    } else {
        throw std::runtime_error("Chunk not found");
    }
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
                find_chunk(n - 1); // will throw an error if the token is not begin-of-chunk
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


bool SynexisSlot::processToken(SynexisSlot *slot, const llama_vocab *vocab, int32_t id) {
    slot->sampled = id;
    return !llama_vocab_is_eog(vocab, id);
}
