#include "synexis/Synexis.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>

#include "batch_helper.h"
#include "mtmd-helper.h"
#include "SynexisSlot.h"


Synexis::Synexis(const std::string &model_path, const llama_context_params &params, int n_slots) {
    ggml_backend_load_all();
    auto modelParams = llama_model_default_params();
    modelParams.use_mmap = true;
    model = llama_model_load_from_file(model_path.c_str(), modelParams);
    if (model == nullptr) {
        throw std::runtime_error("Failed to load model");
    }
    ctx = llama_init_from_model(model, params);

    if (ctx == nullptr) {
        throw std::runtime_error("Failed to create context");
    }

    slots.reserve(n_slots);
    for (int i = 0; i < n_slots; ++i) {
        auto slot = std::make_unique<SynexisSlot>();
        slot->id = i;
        slots.push_back(std::move(slot));
    }
    // {
    mtmd_context_params mparams = mtmd_context_params_default();
    //     mparams.use_gpu = true;
    //     mparams.print_timings = true;
    //     mparams.n_threads = 12;
    //     mparams.verbosity = GGML_LOG_LEVEL_ERROR;
    //     mtmd_context = (mtmd_init_from_file(R"(D:\models\qwen\mmproj-Qwen2.5-Omni-7B-Q8_0.gguf)", model, mparams));
    // }
    batch = llama_batch_init(params.n_batch, 0, 1);
}

void Synexis::stop() {
    running = false;
}


//TODO: better support for files :)
int Synexis::addTask(const std::string &prompt, const SamplingParams &sampling_params) {
    const bool hasMtmd = mtmd_context != nullptr;
    auto slot = findEmptySlot();
    if (!slot) {
    }
    if (hasMtmd) {
        mtmd::bitmaps bitmaps;
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        mtmd_input_text inp_txt = {
            prompt.c_str(),
            /* add_special */ false,
            /* parse_special */ true,
        };
        auto bitmaps_c_ptr = bitmaps.c_ptr();
        int32_t tokenized = mtmd_tokenize(mtmd_context,
                                          chunks.ptr.get(),
                                          &inp_txt,
                                          bitmaps_c_ptr.data(),
                                          bitmaps_c_ptr.size());

        if (tokenized != 0) {
            throw std::runtime_error("Failed to tokenize prompt");
        }
        //TODO add bitmaps

        TaskTokens tokens(chunks);
        slot->tokens = std::move(tokens);
    } else {
        int n_tokens = prompt.length();
        auto vocab = llama_model_get_vocab(model);
        std::vector<llama_token> tokenized(n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.data(), prompt.length(), tokenized.data(), tokenized.size(), false,
                                  true);

        tokenized.resize(n_tokens);

        TaskTokens tokens(std::move(tokenized));
        slot->tokens = std::move(tokens);
        if (slot->sampler) {
            delete slot->sampler;
        }
    }
    slot->sampler = new SynexisSampler(model, sampling_params);
    slot->state = SLOT_STATE_STARTED;
    return slot->id;
}

std::string Synexis::get_result(int task_id) {
    SynexisSlot *foundSLot = nullptr;
    for (auto &slot: slots) {
        if (slot->id == task_id) {
            foundSLot = slot.get();
        }
    }
    while (!foundSLot->idle()) {
        return foundSLot->generatedText;
    }
    return ""; // Task not found
}

std::string Synexis::tokenToPiece(llama_token token, bool special) {
    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::string piece;
    piece.resize(piece.capacity()); // using string internal cache, 15 bytes + '\n'

    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }

    return piece;
}

void Synexis::run() {
    auto runner = [this]() {
        running = true;
        while (running) {
            updateLoop();
        }
    };
    workerThread = std::thread(runner);
}

void Synexis::updateLoop() {
    bool all_idle = true;
    for (auto &slot: slots) {
        if (!slot->idle()) {
            all_idle = false;
            break;
        }
    }
    if (all_idle) {
        return;
    }

    for (auto &slot: slots) {
        if (!slot->idle() && slot->n_past + 1 >= 8096) {
            if (mtmd_context) {
                continue;
            }

            auto vocab = llama_model_get_vocab(model);
            bool add_bos_token = llama_vocab_get_add_bos(vocab);
            const int n_keep = 128 + add_bos_token;
            const int n_left = slot->n_past - n_keep;
            const int n_discard = n_left / 2;

            llama_memory_seq_rm(llama_get_memory(ctx), slot->id, n_keep, n_keep + n_discard);
            llama_memory_seq_add(llama_get_memory(ctx), slot->id, n_keep + n_discard, slot->n_past, -n_discard);

            slot->cacheTokens.shiftTokens(n_keep, n_discard);
            slot->n_past -= n_discard;
            slot->truncated = true;
        }
    }

    clear_batch(batch);

    std::vector<SynexisSlot *> compatible_slots;
    compatible_slots.reserve(slots.size());

    SynexisSlot *slot_batched = nullptr;
    int32_t n_batch = llama_n_batch(ctx);
    int32_t n_ubatch = llama_n_ubatch(ctx);

    for (auto &slot: slots) {
        if (slot->idle()) continue;

        if (!slot_batched) {
            slot_batched = slot.get();
            compatible_slots.push_back(slot.get());
        } else if (slot_batched->canBeBatchedWith(slot.get())) {
            compatible_slots.push_back(slot.get());
        }
    }

    for (auto slot: compatible_slots) {
        if (slot->state == SLOT_STATE_GENERATING) {
            slot->i_batch = batch.n_tokens;
            batch_add(batch, slot->sampled, slot->n_past++, {slot->id}, true);
            slot->cacheTokens.add(slot->sampled);
        }
    }

    for (auto slot: compatible_slots) {
        if (slot->state == SLOT_STATE_PROCESSING_PROMPT || slot->state == SLOT_STATE_STARTED) {
            if (slot->state == SLOT_STATE_STARTED) {
                slot->n_past = 0;
                slot->state = SLOT_STATE_PROCESSING_PROMPT;

                if (slot->promptSize() == 0) {
                    slot->reset();
                    continue;
                }

                if (slot->promptSize() > n_ubatch || slot->promptSize() > 8096) {
                    slot->reset();
                    continue;
                }

                slot->n_prompt_tokens_processed = 0;
            }

            if (batch.n_tokens + (slot->promptSize() - slot->n_past) > n_batch) {
                continue;
            }

            if (!llama_memory_seq_rm(llama_get_memory(ctx), slot->id, slot->n_past, -1)) {
                llama_memory_seq_rm(llama_get_memory(ctx), slot->id, -1, -1);
                slot->n_past = 0;
            }

            slot->cacheTokens.keepFirst(slot->n_past);

            if (slot->n_past < slot->promptSize() && slot->tokens.getTokens()[slot->n_past] == LLAMA_TOKEN_NULL) {
                int32_t new_n_past;
                int32_t res = slot->tokens.process_chunk(ctx, mtmd_context, slot->n_past, slot->id, new_n_past);
                int32_t n_pos = new_n_past - slot->n_past;

                if (res != 0) {
                    slot->reset();
                    continue;
                }

                const auto &chunk = slot->tokens.find_chunk(slot->n_past);
                slot->cacheTokens.parseMtmdChunk(chunk.get());

                slot->n_past += n_pos;
                slot->n_prompt_tokens_processed += n_pos;
            }

            while (slot->n_past < slot->promptSize() && batch.n_tokens < n_batch) {
                llama_token cur_tok = slot->tokens.getTokens()[slot->n_past];
                if (cur_tok == LLAMA_TOKEN_NULL) {
                    break;
                }

                batch_add(batch, cur_tok, slot->n_past, {slot->id}, false);
                slot->cacheTokens.add(cur_tok);

                slot->n_prompt_tokens_processed++;
                slot->n_past++;
            }

            if (slot->n_past == slot->promptSize()) {
                slot->state = SLOT_STATE_DONE_PROMPT;
                slot->sampler->reset();

                const auto &tokens = slot->tokens.getTokens();
                for (int i = 0; i < slot->promptSize(); ++i) {
                    llama_token id = tokens[i];
                    if (id != LLAMA_TOKEN_NULL) {
                        slot->sampler->accept(id, false);
                    }
                }

                batch.logits[batch.n_tokens - 1] = true;
                slot->n_decoded = 0;
                slot->i_batch = batch.n_tokens - 1;
            }
        }

        if (batch.n_tokens >= n_batch) {
            break;
        }
    }

    if (batch.n_tokens == 0) {
        return;
    }

    int32_t i_next = 0;
    for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
        const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

        llama_batch batch_view = {
            n_tokens,
            batch.token + i,
            nullptr,
            batch.pos + i,
            batch.n_seq_id + i,
            batch.seq_id + i,
            batch.logits + i,
        };

        const int ret = llama_decode(ctx, batch_view);

        if (ret != 0) {
            if (n_batch == 1 && ret == 1) {
                for (auto &slot: slots) {
                    slot->reset();
                }
                break;
            }
            if (ret == -1 || ret < -1) {
                for (auto &slot: slots) {
                    slot->reset();
                }
                break;
            }

            n_batch /= 2;
            if (n_batch == 0) {
                break;
            }
            continue;
        }

        i_next = i + n_tokens;
        n_batch = llama_n_batch(ctx);

        for (auto &slot: slots) {
            if (slot->i_batch < (int) i || slot->i_batch >= (int) (i + n_tokens)) {
                continue;
            }

            if (slot->state == SLOT_STATE_DONE_PROMPT) {
                slot->state = SLOT_STATE_GENERATING;
            } else if (slot->state != SLOT_STATE_GENERATING) {
                continue;
            }

            const int tok_idx = slot->i_batch - i;
            llama_token id = slot->sampler->sample(ctx, tok_idx);

            slot->i_batch = -1;
            slot->sampler->accept(id, true);
            slot->n_decoded += 1;

            auto vocab = llama_model_get_vocab(model);
            slot->result += tokenToPiece(id, false);
            std::cout << "Generated for slot " << slot->id << ": " << slot->result << std::endl;
            if (!slot->processToken(slot.get(), vocab, id)) {
                slot->release();
                continue;
            }

            slot->sampled = id;
        }
    }
}



SynexisSlot *Synexis::findEmptySlot() const {
    for (auto &slot: slots) {
        if (slot->state == SLOT_STATE_IDLE) {
            return slot.get();
        }
    }
    return nullptr;
}


Synexis::~Synexis() {
    if (running) {
        stop();
    }
    if (workerThread.joinable()) {
        workerThread.join();
    }
    llama_free(ctx);
    llama_model_free(model);
    mtmd_free(mtmd_context);
    llama_backend_free();
}
