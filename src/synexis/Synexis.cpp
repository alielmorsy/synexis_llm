#include <synexis/Synexis.h>

#include "SynexisImpl.h"
#include <llama.h>

Synexis::Synexis(const std::string &model_path, int n_slots) {
    llama_context_params lparams = llama_context_default_params();
    lparams.n_ctx = 2048;
    lparams.n_threads = 12;
    impl = new SynexisImpl(model_path, lparams, n_slots);
}

Synexis::~Synexis() {
}

int Synexis::addTask(const std::string &prompt, const SamplingParams &sampling_params) {
    return impl->addTask(prompt, sampling_params);
}

std::string Synexis::get_result(int task_id) {
    return impl->get_result(task_id);
}

void Synexis::run() const {
    impl->run();
}

void Synexis::stop() const {
    impl->stop();
}
