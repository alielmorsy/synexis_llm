//
// Created by Ali Elmorsy on 7/15/2025.
//

#ifndef SYNEXISIMPL_H
#define SYNEXISIMPL_H
#include <mtmd.h>
#include <llama-cpp.h>
#include <mutex>

#include "SynexisSlot.h"
#include "synexis/TaskParams.h"
#include "Request.h"
#include <future>

class SynexisImpl {
public:
    ~SynexisImpl();

    std::future<std::string> addTask(const std::string &prompt, const TaskParams &params);

    void run();

    std::string getTemplate();

    SynexisImpl(const std::string &model_path, const llama_context_params &params, int n_slots);

    void stop();

private:
    void updateLoop();
    void tokenizationLoop();

    SynexisSlot *findEmptySlot();


    std::string tokenToPiece(int32_t token, bool special) const;

    llama_model *model;
    llama_context *ctx;
    mtmd_context *mtmd_context = nullptr;
    std::vector<std::unique_ptr<SynexisSlot> > slots;
    std::mutex slotLock;
    std::condition_variable cv;
    std::thread workerThread;
    std::atomic<bool> running{false};
    llama_batch batch;
    
    std::deque<std::unique_ptr<Request>> tokenization_queue;
    std::mutex tokenization_queue_mutex;
    std::condition_variable tokenization_queue_cv;
    std::thread tokenization_thread;
};


#endif //SYNEXISIMPL_H
