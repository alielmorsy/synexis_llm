#pragma once

#include "llama.h"

void clear_batch(llama_batch &batch) {
    batch.n_tokens = 0;
}


void batch_add(llama_batch &batch, llama_token tokenID, int32_t nPast,
               const std::vector<llama_seq_id> &seq_ids,
               bool logits) {
    batch.token[batch.n_tokens] = tokenID;
    batch.pos[batch.n_tokens] = nPast;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}
