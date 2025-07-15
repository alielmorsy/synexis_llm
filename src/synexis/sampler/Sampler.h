#pragma once

#include "../../include/synexis/sampler/Enums.h"
#include "RingBuffer.h"
#include "../../include/synexis/sampler/StructParams.h"


struct llama_sampler;
struct llama_vocab;
struct llama_context;
struct llama_model;
struct llama_token_data;
struct llama_token_data_array;

class SynexisSampler {
public:
    explicit SynexisSampler(llama_model *model, const SamplingParams &params = SamplingParams{});


    // Disable copy constructor and assignment
    SynexisSampler(const SynexisSampler &) = delete;

    SynexisSampler &operator=(const SynexisSampler &) = delete;

    // Enable move constructor and assignment
    SynexisSampler(SynexisSampler &&other) noexcept;

    SynexisSampler &operator=(SynexisSampler &&other) noexcept;

    // Main sampling interface
    int32_t sample(llama_context *ctx, int idx = -1, bool grammar_first = false);

    void accept(int32_t token, bool accept_grammar = true);

    // Configuration methods
    void set_grammar(const std::string &grammar_str, bool lazy = false);

    void reset();


    ~SynexisSampler();

private:
    // Internal methods
    bool initialize_grammar_sampler();

    bool initialize_chain_sampler();


    void setLogits(llama_context *ctx, int idx);

    std::string escapeRegex(const std::string &str);

    std::string joinStrings(const std::vector<std::string> &strings, const std::string &delimiter);

    // Member variables
    SamplingParams params_;
    const llama_model *model_;
    const llama_vocab *vocab_;

    llama_sampler *grammar_sampler_;
    llama_sampler *chain_sampler_;

    RingBuffer<int32_t> token_history_;
    std::vector<llama_token_data> current_candidates_;
    std::unique_ptr<llama_token_data_array> current_candidates_array_;


    bool initialized_ = false;
};
