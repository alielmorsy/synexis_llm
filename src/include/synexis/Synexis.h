#pragma once

#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <memory>
#include "sampler/Sampler.h"
#include "llama.h"
#include "mtmd.h"


struct SynexisSlot;

class Synexis {
public:
    Synexis(const std::string &model_path, const llama_context_params &params, int n_slots = 1);

    void stop();

    std::string generate_now(std::string prompt);

    ~Synexis();

    int add_task(const std::string &prompt, const SamplingParams &sampling_params);

    std::string get_result(int task_id);

    std::string tokenToPiece(llama_token token, bool special);

    void run();

    void updateLoop();

    void batch_add(llama_batch &batch, llama_token tokenID, int32_t nPast, const std::vector<llama_seq_id> &seq_ids,
                   bool logits);

private:
    [[nodiscard]] SynexisSlot *findEmptySlot() const;

    llama_model *model;
    llama_context *ctx;
    mtmd_context *mtmd_context = nullptr;
    std::vector<std::unique_ptr<SynexisSlot> > slots;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread workerThread;
    std::atomic<bool> running{false};
    llama_batch batch;
};
