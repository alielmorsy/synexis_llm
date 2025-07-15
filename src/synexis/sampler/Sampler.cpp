#include <cassert>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <Sampler.h>

#include <llama-cpp.h>



SynexisSampler::SynexisSampler(llama_model *model, const SamplingParams &params)
    : params_(params)
      , model_(model)
      , vocab_(llama_model_get_vocab(model))
      , token_history_(std::max(32, params.n_prev)) {
    if (!initialize_grammar_sampler() || !initialize_chain_sampler()) {
        throw std::runtime_error("Failed to initialize samplers");
    }

    initialized_ = true;
}

SynexisSampler::SynexisSampler(SynexisSampler &&other) noexcept
    : params_(std::move(other.params_))
      , model_(other.model_)
      , vocab_(other.vocab_)
      , grammar_sampler_(std::move(other.grammar_sampler_))
      , chain_sampler_(std::move(other.chain_sampler_))
      , token_history_(std::move(other.token_history_))
      , current_candidates_(std::move(other.current_candidates_))
      , current_candidates_array_(std::move(other.current_candidates_array_))
      , initialized_(other.initialized_) {
    other.grammar_sampler_ = nullptr;
    other.chain_sampler_ = nullptr;
    other.initialized_ = false;
}

SynexisSampler &SynexisSampler::operator=(SynexisSampler &&other) noexcept {
    if (this != &other) {
        reset();

        params_ = std::move(other.params_);
        model_ = other.model_;
        vocab_ = other.vocab_;
        grammar_sampler_ = other.grammar_sampler_;
        chain_sampler_ = other.chain_sampler_;
        token_history_ = std::move(other.token_history_);
        current_candidates_ = std::move(other.current_candidates_);
        current_candidates_array_ = std::move(other.current_candidates_array_);
        initialized_ = other.initialized_;

        other.grammar_sampler_ = nullptr;
        other.chain_sampler_ = nullptr;
        other.initialized_ = false;
    }
    return *this;
}

llama_token SynexisSampler::sample(llama_context *ctx, int idx, bool grammar_first) {
    if (!initialized_ || !ctx) {
        throw std::runtime_error("Sampler not initialized or context is null");
    }

    setLogits(ctx, idx);

    if (grammar_first) {
        llama_sampler_apply(grammar_sampler_, current_candidates_array_.get());
    }

    llama_sampler_apply(chain_sampler_, current_candidates_array_.get());

    assert(current_candidates_array_->selected != -1 && "no selected token during sampling");

    const llama_token id = current_candidates_array_->data[current_candidates_array_->selected].id;

    if (grammar_first) {
        return id;
    }

    // Check if the sampled token fits the grammar
    llama_token_data single_token_data = {id, 1.0f, 0.0f};
    llama_token_data_array single_token_data_array = {&single_token_data, 1, -1, false};

    llama_sampler_apply(grammar_sampler_, &single_token_data_array);

    const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
    if (is_valid) {
        return id;
    }

    // Resampling: apply grammar first, then sampling chain
    setLogits(ctx, idx);

    llama_sampler_apply(grammar_sampler_, current_candidates_array_.get());
    llama_sampler_apply(chain_sampler_, current_candidates_array_.get());

    assert(current_candidates_array_->selected != -1 && "no selected token during re-sampling");

    return current_candidates_array_->data[current_candidates_array_->selected].id;
}

void SynexisSampler::accept(llama_token token, bool accept_grammar) {
    if (!initialized_) {
        throw std::runtime_error("Sampler not initialized");
    }

    if (accept_grammar) {
        llama_sampler_accept(grammar_sampler_, token);
    }

    llama_sampler_accept(chain_sampler_, token);
    token_history_.push_back(token);
}

void SynexisSampler::set_grammar(const std::string &grammar_str, bool lazy) {
    params_.grammar = grammar_str;
    params_.grammar_lazy = lazy;

    if (initialized_) {
        // Reinitialize grammar sampler
        if (grammar_sampler_) {
            llama_sampler_free(grammar_sampler_);
        }
        initialize_grammar_sampler();
    }
}



bool SynexisSampler::initialize_grammar_sampler() {
    std::vector<std::string> trigger_patterns;
    std::vector<std::string> patterns_anywhere;
    std::vector<llama_token> trigger_tokens;

    for (const auto &trigger: params_.grammar_triggers) {
        switch (trigger.type) {
            case GRAMMAR_TRIGGER_TYPE_WORD: {
                const auto &word = trigger.value;
                patterns_anywhere.push_back(escapeRegex(word));
                break;
            }
            case GRAMMAR_TRIGGER_TYPE_PATTERN: {
                patterns_anywhere.push_back(trigger.value);
                break;
            }
            case GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                trigger_patterns.push_back(trigger.value);
                break;
            }
            case GRAMMAR_TRIGGER_TYPE_TOKEN: {
                trigger_tokens.push_back(trigger.token);
                break;
            }
            default:
                throw std::runtime_error("Unknown grammar trigger type");
        }
    }

    if (!patterns_anywhere.empty()) {
        trigger_patterns.push_back("^[\\s\\S]*?(" + joinStrings(patterns_anywhere, "|") + ")[\\s\\S]*");
    }

    std::vector<const char *> trigger_patterns_c;
    trigger_patterns_c.reserve(trigger_patterns.size());
    for (const auto &regex: trigger_patterns) {
        trigger_patterns_c.push_back(regex.c_str());
    }

    auto ptr = params_.grammar_lazy
                   ? llama_sampler_init_grammar_lazy_patterns(vocab_, params_.grammar.c_str(), "root",
                                                              trigger_patterns_c.data(), trigger_patterns_c.size(),
                                                              trigger_tokens.data(), trigger_tokens.size())
                   : llama_sampler_init_grammar(vocab_, params_.grammar.c_str(), "root");

    grammar_sampler_ = ptr;

    return true;
}

bool SynexisSampler::initialize_chain_sampler() {
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    lparams.no_perf = params_.no_perf;

    chain_sampler_ = llama_sampler_chain_init(lparams);
    if (!chain_sampler_) {
        return false;
    }
    //TODO
    // // Add logit bias sampler
    // llama_sampler_chain_add(chain_sampler_,
    //                         llama_sampler_init_logit_bias(
    //                             llama_vocab_n_tokens(vocab_),
    //                             params_.logit_bias.size(),
    //                             params_.logit_bias.data()));

    if (params_.mirostat == 0) {
        // Add samplers in the specified order
        for (const auto &sampler_type: params_.samplers) {
            switch (sampler_type) {
                case SAMPLER_TYPE_DRY: {
                    std::vector<const char *> c_breakers;
                    c_breakers.reserve(params_.dry_sequence_breakers.size());
                    for (const auto &str: params_.dry_sequence_breakers) {
                        c_breakers.push_back(str.c_str());
                    }

                    llama_sampler_chain_add(chain_sampler_,
                                            llama_sampler_init_dry(vocab_, llama_model_n_ctx_train(model_),
                                                                   params_.dry_multiplier, params_.dry_base,
                                                                   params_.dry_allowed_length,
                                                                   params_.dry_penalty_last_n,
                                                                   c_breakers.data(), c_breakers.size()));
                    break;
                }
                case SAMPLER_TYPE_TOP_K:
                    llama_sampler_chain_add(chain_sampler_, llama_sampler_init_top_k(params_.top_k));
                    break;
                case SAMPLER_TYPE_TOP_P:
                    llama_sampler_chain_add(chain_sampler_,
                                            llama_sampler_init_top_p(params_.top_p, params_.min_keep));
                    break;
                case SAMPLER_TYPE_TOP_N_SIGMA:
                    llama_sampler_chain_add(chain_sampler_, llama_sampler_init_top_n_sigma(params_.top_n_sigma));
                    break;
                case SAMPLER_TYPE_MIN_P:
                    llama_sampler_chain_add(chain_sampler_,
                                            llama_sampler_init_min_p(params_.min_p, params_.min_keep));
                    break;
                case SAMPLER_TYPE_XTC:
                    llama_sampler_chain_add(chain_sampler_,
                                            llama_sampler_init_xtc(params_.xtc_probability, params_.xtc_threshold,
                                                                   params_.min_keep, params_.seed));
                    break;
                case SAMPLER_TYPE_TYPICAL_P:
                    llama_sampler_chain_add(chain_sampler_,
                                            llama_sampler_init_typical(params_.typ_p, params_.min_keep));
                    break;
                case SAMPLER_TYPE_TEMPERATURE:
                    llama_sampler_chain_add(chain_sampler_,
                                            llama_sampler_init_temp_ext(params_.temp, params_.dynatemp_range,
                                                                        params_.dynatemp_exponent));
                    break;
                case SAMPLER_TYPE_INFILL:
                    llama_sampler_chain_add(chain_sampler_, llama_sampler_init_infill(vocab_));
                    break;
                case SAMPLER_TYPE_PENALTIES:
                    llama_sampler_chain_add(chain_sampler_,
                                            llama_sampler_init_penalties(
                                                params_.penalty_last_n, params_.penalty_repeat, params_.penalty_freq,
                                                params_.penalty_present));
                    break;
                default:
                    throw std::runtime_error("Unknown sampler type");
            }
        }
        llama_sampler_chain_add(chain_sampler_, llama_sampler_init_dist(params_.seed));
    } else if (params_.mirostat == 1) {
        llama_sampler_chain_add(chain_sampler_, llama_sampler_init_temp(params_.temp));
        llama_sampler_chain_add(chain_sampler_,
                                llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab_), params_.seed,
                                                            params_.mirostat_tau, params_.mirostat_eta, 100));
    } else if (params_.mirostat == 2) {
        llama_sampler_chain_add(chain_sampler_, llama_sampler_init_temp(params_.temp));
        llama_sampler_chain_add(chain_sampler_,
                                llama_sampler_init_mirostat_v2(params_.seed, params_.mirostat_tau,
                                                               params_.mirostat_eta));
    } else {
        throw std::runtime_error("Unknown mirostat version");
    }

    return true;
}


void SynexisSampler::setLogits(llama_context *ctx, int idx) {
    const float *logits = nullptr;
    if (idx == -1) {
        logits = llama_get_logits(ctx);
    } else {
        logits = llama_get_logits_ith(ctx, idx);
    }

    if (!logits) {
        throw std::runtime_error("Failed to get logits from context");
    }

    // Get vocabulary size
    const int n_vocab = llama_vocab_n_tokens(vocab_);
    if (n_vocab <= 0) {
        throw std::runtime_error("Invalid vocabulary size");
    }


    current_candidates_.resize(n_vocab);

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        current_candidates_[token_id] = {
            token_id,
            logits[token_id],
            0.0f
        };
    }

    current_candidates_array_ = std::make_unique<llama_token_data_array>();
    current_candidates_array_->data = current_candidates_.data();
    current_candidates_array_->size = current_candidates_.size();
    current_candidates_array_->selected = -1;
    current_candidates_array_->sorted = false;
    // Initialize the token data array structure
    // *current_candidates_array_ = {
    //     current_candidates_.data(), // data pointer
    //     current_candidates_.size(), // size
    //     -1, // selected (no selection yet)
    //     false // sorted (not sorted yet)
    // };
}

std::string SynexisSampler::escapeRegex(const std::string &str) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(str, special_chars, "\\$&");
}

std::string SynexisSampler::joinStrings(const std::vector<std::string> &strings, const std::string &delimiter) {
    std::ostringstream result;
    for (size_t i = 0; i < strings.size(); ++i) {
        if (i > 0) {
            result << delimiter;
        }
        result << strings[i];
    }
    return result.str();
}


void SynexisSampler::reset() {

    llama_sampler_reset(chain_sampler_);
    llama_sampler_reset(grammar_sampler_);
}

SynexisSampler::~SynexisSampler() {
    token_history_.clear();
    llama_sampler_free(chain_sampler_);
    llama_sampler_free(grammar_sampler_);
}
