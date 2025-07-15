#pragma once
#include <set>
#include <string>
#include <vector>

#include <synexis/sampler/Enums.h>

struct SamplingParams {
    uint32_t seed = 1900; // Default seed

    // Sampling parameters
    int32_t n_prev = 64;
    int32_t n_probs = 0;
    int32_t min_keep = 0;
    int32_t top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float xtc_probability = 0.00f;
    float xtc_threshold = 0.10f;
    float typ_p = 1.00f;
    float temp = 0.80f;
    float dynatemp_range = 0.00f;
    float dynatemp_exponent = 1.00f;

    // Penalty parameters
    int32_t penalty_last_n = 64;
    float penalty_repeat = 1.00f;
    float penalty_freq = 0.00f;
    float penalty_present = 0.00f;

    // DRY parameters
    float dry_multiplier = 0.0f;
    float dry_base = 1.75f;
    int32_t dry_allowed_length = 2;
    int32_t dry_penalty_last_n = -1;
    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};

    // Mirostat parameters
    int32_t mirostat = 0;
    float top_n_sigma = -1.00f;
    float mirostat_tau = 5.00f;
    float mirostat_eta = 0.10f;

    // Flags
    bool ignore_eos = false;
    bool no_perf = false;
    bool timing_per_token = false;

    // Sampler chain configuration
    std::vector<SamplerType> samplers = {
        SAMPLER_TYPE_PENALTIES,
        SAMPLER_TYPE_DRY,
        SAMPLER_TYPE_TOP_N_SIGMA,
        SAMPLER_TYPE_TOP_K,
        SAMPLER_TYPE_TYPICAL_P,
        SAMPLER_TYPE_TOP_P,
        SAMPLER_TYPE_MIN_P,
        SAMPLER_TYPE_XTC,
        SAMPLER_TYPE_TEMPERATURE,
    };

    // Grammar parameters
    std::string grammar;
    bool grammar_lazy = false;
    std::vector<GrammarTrigger> grammar_triggers;
    std::set<int32_t> preserved_tokens;

};
