#include <synexis/Synexis.h>

#include "SynexisImpl.h"
#include <llama.h>

Synexis::Synexis(SynexisArguments args) {
    impl = new SynexisImpl(args);
}


std::future<std::string> Synexis::addTask(const std::string &prompt, const TaskParams &params) {
    return impl->addTask(prompt, params);
}

std::string Synexis::get_result(int task_id) {
    return "";
}

void Synexis::run() const {
    impl->run();
}

void Synexis::stop() const {
    impl->stop();
}

std::string Synexis::getTemplate() const {
    return impl->getTemplate();
}


Synexis::~Synexis() {
    delete impl;
}
