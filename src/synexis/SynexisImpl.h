//
// Created by Ali Elmorsy on 7/15/2025.
//

#ifndef SYNEXISIMPL_H
#define SYNEXISIMPL_H
#include <mtmd.h>
#include <llama-cpp.h>
#include <mutex>

#include "SynexisSlot.h"

struct SamplingParams;

class SynexisImpl {
public:
    ~SynexisImpl();

    int addTask(const std::string &prompt, const SamplingParams &sampling_params);

    std::string get_result(int task_id);


    void run();

    SynexisImpl(const std::string &model_path, const llama_context_params &params, int n_slots);

    void stop();

private:
    void updateLoop();

    [[nodiscard]] SynexisSlot *findEmptySlot() const;

    std::string tokenToPiece(int32_t token, bool special);

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


#endif //SYNEXISIMPL_H
