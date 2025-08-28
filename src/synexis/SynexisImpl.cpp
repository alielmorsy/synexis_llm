#include "SynexisImpl.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "batch_helper.h"
#include "SynexisSlot.h"
#include "synexis/TaskParams.h"
#include "TaskTokens.h"
#include "utils.h"
#include "../vendor/llama.cpp/ggml/src/ggml-impl.h"
#include "../vendor/llama.cpp/src/llama-model.h"
#include "synexis/SynexisArguments.h"

#define TOKEN_PIECE_MAX_SIZE 64
#define CHATML_TEMPLATE_SRC \
"{%- for message in messages -%}\n" \
"  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
"{%- endfor -%}\n" \
"{%- if add_generation_prompt -%}\n" \
"  {{- '<|im_start|>assistant\n' -}}\n" \
"{%- endif -%}"

SynexisImpl::SynexisImpl(const SynexisArguments &args): params(args) {
    ggml_backend_load_all();
    llama_log_set([](ggml_log_level level, const char *text, void * /*user_data*/) {
        if (level != GGML_LOG_LEVEL_DEBUG) {
            std::cerr << text;
        }
    }, nullptr);

    auto modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = args.numberOfGpuLayers;
    modelParams.use_mmap = args.use_mmap;
    model = llama_model_load_from_file(params.modelPath.c_str(), modelParams);
    if (model == nullptr) {
        throw std::runtime_error("Failed to load model");
    }

    auto contextParams = llama_context_default_params();
    contextParams.n_ctx = params.n_ctx;
    contextParams.n_batch = params.n_batch;
    contextParams.n_ubatch = 512;
    contextParams.n_threads_batch = params.numberOfThreads;
    contextParams.embeddings = args.embedding;
    ctx = llama_init_from_model(model, contextParams);

    if (ctx == nullptr) {
        throw std::runtime_error("Failed to create context");
    }

    slots.reserve(args.n_slots);
    for (int i = 0; i < args.n_slots; ++i) {
        auto slot = std::make_unique<SynexisSlot>();
        slot->id = i;
        slots.push_back(std::move(slot));
    }
    if (!args.modelProjectorPath.empty()) {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = args.numberOfGpuLayers > 0;
        mparams.n_threads = params.numberOfThreads;
        mparams.verbosity = GGML_LOG_LEVEL_ERROR;
        mtmd_context = mtmd_init_from_file(args.modelProjectorPath.c_str(), model, mparams);
    }

    batch = llama_batch_init(params.n_batch, 0, 1);
}

void common_embd_normalize(const float *inp, float *out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

std::vector<std::vector<float> > SynexisImpl::getEmbedding(const std::string &prompt) {
    clear_batch(batch);
    llama_memory_clear(llama_get_memory(ctx), true);
    const int n_embd = llama_model_n_embd(model);

    std::vector<std::vector<float> > embeddings_res;
    std::vector embeddings(n_embd, 0.0f);

    size_t n_tokens = prompt.size();
    auto vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokenized(n_tokens);
    n_tokens = llama_tokenize(vocab, prompt.data(), n_tokens, tokenized.data(),
                              tokenized.size(), true,
                              true);
    tokenized.resize(n_tokens);
    int n_past = 0;
    for (int i = 0; i < n_tokens; ++i) {
        batch_add(batch, tokenized[i], n_past++, {0}, true);
    }
    int res = llama_decode(ctx, batch);
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!batch.logits[i]) {
            continue;
        }

        const float *embd = nullptr;
        if (llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_NONE) {
            embd = llama_get_embeddings_ith(ctx, i);
        } else {
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        }
        if (llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE) {
            //embd_normalize (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)
            common_embd_normalize(embd, embeddings.data(), n_embd, 2);
            embeddings_res.push_back(embeddings);
            break;
        } else
            embeddings_res.emplace_back(embd, embd + n_embd);
    }
    return embeddings_res;
}

std::future<std::string> SynexisImpl::addTask(const std::string &prompt, const TaskParams &params) {
    auto request = std::make_unique<Request>();
    request->prompt = prompt;
    request->params = params;

    // Create a promise/future pair
    std::future<std::string> future = request->promise.get_future();

    // Find a free slot
    SynexisSlot *slot = nullptr;
    //Forcing one slot to be selected at atime
    std::unique_lock lock(slotLock);
    GGML_LOG_INFO("Waiting for a free slot\n");
    while (slot == nullptr) {
        slot = findEmptySlot();
        if (slot == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    GGML_LOG_INFO("Found a free slot\n");
    // Tokenization
    if (mtmd_context != nullptr) {
        mtmd::bitmaps bitmaps;
        for (auto &[data, size]: request->params.media) {
            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(mtmd_context, data, size));
            std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
            bmp.set_id(hash.c_str());
            bitmaps.entries.push_back(std::move(bmp));
        }

        mtmd_input_text inp_txt = {
            request->prompt.c_str(),
            /* add_special */ true,
            /* parse_special */ true,
        };

        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        auto bitmaps_c_ptr = bitmaps.c_ptr();
        int32_t tokenized = mtmd_tokenize(
            mtmd_context,
            chunks.ptr.get(),
            &inp_txt,
            bitmaps_c_ptr.data(),
            bitmaps_c_ptr.size()
        );

        if (tokenized != 0) {
            throw std::runtime_error("Failed to tokenize prompt");
        }

        slot->tokens = TaskTokens(chunks);
    } else {
        size_t n_tokens = request->prompt.length();
        auto vocab = llama_model_get_vocab(model);
        std::vector<llama_token> tokenized(n_tokens);
        n_tokens = llama_tokenize(
            vocab,
            request->prompt.data(),
            request->prompt.length(),
            tokenized.data(),
            tokenized.size(),
            false,
            true
        );
        tokenized.resize(n_tokens);
        slot->tokens = TaskTokens(std::move(tokenized));
    }

    // Setup the sampler and slot
    delete slot->sampler;
    slot->sampler = new SynexisSampler(model, request->params.samplerParams);
    slot->request = std::move(request);
    slot->state = SLOT_STATE_STARTED;

    return future;
}

std::string SynexisImpl::tokenToPiece(llama_token token, bool special) const {
    const llama_vocab *vocab = llama_model_get_vocab(model);

    // Try with a reasonably small buffer on the stack first
    char buf[TOKEN_PIECE_MAX_SIZE]; // fast path for most tokens
    const int32_t n_chars = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, special);

    if (n_chars >= 0) {
        return std::string(buf, n_chars);
    }

    const size_t required = -n_chars;
    std::string piece(required, '\0');
    const int check = llama_token_to_piece(vocab, token, piece.data(), required, 0, special);
    GGML_ASSERT(check ==required);
    return piece;
}


void SynexisImpl::run() {
    running = true;
    //tokenization_thread = std::thread(&SynexisImpl::tokenizationLoop, this);
    workerThread = std::thread(&SynexisImpl::updateLoop, this);
    auto c = workerThread.get_id();
    std::cout << "C++ thread ID: " << c << std::endl;
}

void SynexisImpl::stop() {
    running = false;
    tokenization_queue_cv.notify_all();
}

void SynexisImpl::tokenizationLoop() {
    mtmd_default_marker();
    while (running) {
        std::unique_ptr<Request> request; {
            std::unique_lock lock(tokenization_queue_mutex);
            tokenization_queue_cv.wait(lock, [this] { return !tokenization_queue.empty() || !running; });
            if (!running) break;
            request = std::move(tokenization_queue.front());
            tokenization_queue.pop_front();
        }
        SynexisSlot *slot = nullptr;
        GGML_LOG_INFO("Waiting for a free slot\n");
        while (running && slot == nullptr) {
            slot = findEmptySlot();
            if (slot == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        GGML_LOG_INFO("Found a free slot\n");
        if (!running) break;

        //In case we have mtmd context we would have to parse media files
        if (mtmd_context != nullptr) {
            mtmd::bitmaps bitmaps;
            for (auto &[data, size]: request->params.media) {
                mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(mtmd_context, data, size));
                // calculate bitmap hash (for KV caching)
                std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
                bmp.set_id(hash.c_str());
                bitmaps.entries.push_back(std::move(bmp));
            }
            mtmd_input_text inp_txt = {
                request->prompt.c_str(),
                /* add_special */ true,
                /* parse_special */ true,
            };
            mtmd::input_chunks chunks(mtmd_input_chunks_init());
            auto bitmaps_c_ptr = bitmaps.c_ptr();
            int32_t tokenized = mtmd_tokenize(mtmd_context,
                                              chunks.ptr.get(),
                                              &inp_txt,
                                              bitmaps_c_ptr.data(),
                                              bitmaps_c_ptr.size());
            if (tokenized != 0) {
                throw std::runtime_error("Failed to tokenize prompt");
            }
            slot->tokens = TaskTokens(chunks);
        } else {
            size_t n_tokens = request->prompt.length();
            auto vocab = llama_model_get_vocab(model);
            std::vector<llama_token> tokenized(n_tokens);
            n_tokens = llama_tokenize(vocab, request->prompt.data(), request->prompt.length(), tokenized.data(),
                                      tokenized.size(), false,
                                      true);
            tokenized.resize(n_tokens);

            slot->tokens = TaskTokens(std::move(tokenized));
        }


        delete slot->sampler;
        slot->sampler = new SynexisSampler(model, request->params.samplerParams);
        slot->request = std::move(request);
        slot->state = SLOT_STATE_STARTED;
    }
}

void SynexisImpl::updateLoop() {
    while (running) {
        bool all_idle = true;
        for (auto &slot: slots) {
            if (!slot->idle()) {
                all_idle = false;
                break;
            }
        }
        if (all_idle) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }


        std::vector<SynexisSlot *> compatible_slots;
        SynexisSlot *slot_batched = nullptr;
        for (auto &slot: slots) {
            if (slot->idle()) continue;

            if (!slot_batched) {
                slot_batched = slot.get();
                compatible_slots.push_back(slot.get());
            } else if (slot_batched->canBeBatchedWith(slot.get())) {
                compatible_slots.push_back(slot.get());
            }
        }
        for (const auto &slot: compatible_slots) {
            if (slot->n_past + 1 >= params.n_ctx) {
                if (mtmd_context) {
                    continue;
                }
                auto vocab = llama_model_get_vocab(model);
                bool add_bos_token = llama_vocab_get_add_bos(vocab);
                const int n_keep = params.n_batch + add_bos_token;
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


        compatible_slots.reserve(slots.size());


        int32_t n_batch = llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);


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
                    // if (!(llama_get_memory(ctx) && llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_LAST)) {
                    //     if (slot->promptSize() > n_ubatch || slot->promptSize() > params.n_ctx) {
                    //         slot->reset();
                    //         continue;
                    //     }
                    // }


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
            continue;
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
                std::cerr << "Retrying Batch" << std::endl;
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
                if (slot->i_batch < i || slot->i_batch >= i + n_tokens) {
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
                std::string token_str = tokenToPiece(id, false);

                if (slot->request->params.stream) {
                    if (slot->request->params.on_token) {
                        slot->request->params.on_token(token_str);
                    }
                } else {
                    slot->generatedText += token_str;
                }

                if (!slot->processToken(vocab, id, token_str)) {
                    slot->release();
                    continue;
                }

                slot->sampled = id;
            }
        }
    }
}


SynexisSlot *SynexisImpl::findEmptySlot() {
    for (auto &slot: slots) {
        if (slot->state == SLOT_STATE_IDLE) {
            return slot.get();
        }
    }
    return nullptr;
}


std::string SynexisImpl::getTemplate() {
    const auto *modelTemplate = llama_model_chat_template(model, nullptr);
    std::string templateSource;
    if (modelTemplate) {
        templateSource = modelTemplate;
    } else {
        templateSource = CHATML_TEMPLATE_SRC;
    }
    return templateSource;
}


SynexisImpl::~SynexisImpl() {
    if (running) {
        stop();
    }
    // if (tokenization_thread.joinable()) {
    //     tokenization_thread.join();
    // }
    if (workerThread.joinable()) {
        workerThread.join();
    }
    llama_free(ctx);
    llama_model_free(model);
    mtmd_free(mtmd_context);
    llama_backend_free();
}

std::string SynexisImpl::getToken(std::string &str) {
    auto vocab = llama_model_get_vocab(model);
    llama_token token;
    if (str == "BOS") {
        token = llama_vocab_bos(vocab);
    } else if (str == "EOS") {
        token = llama_vocab_eos(vocab);
    } else {
        GGML_ABORT("Unknown requested token");
    }
    return tokenToPiece(token, true);
}
