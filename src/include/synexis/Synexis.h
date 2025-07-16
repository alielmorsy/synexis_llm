#pragma once
#include <future>
#include <string>
#include "sampler/StructParams.h"
#include "TaskParams.h"
class SynexisImpl;


class Synexis {
public:
    Synexis(const std::string &model_path, int n_slots);


    ~Synexis();

    std::future<std::string> addTask(const std::string &prompt, const TaskParams &sampling_params);

    std::string get_result(int task_id);


    void run() const;

    void stop() const;

    std::string getTemplate() const;

private:
    SynexisImpl *impl;
};
