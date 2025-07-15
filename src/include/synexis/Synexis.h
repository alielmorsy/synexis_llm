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
    Synexis(const std::string &model_path, const llama_context_params &params, int n_slots);


    ~Synexis();

    int addTask(const std::string &prompt, const SamplingParams &sampling_params);

    std::string get_result(int task_id);

    std::string tokenToPiece(llama_token token, bool special);

    void run();

    void stop();

private:
    void updateLoop();

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
