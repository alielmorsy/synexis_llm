#pragma once
#include <string>
#include "sampler/StructParams.h"

class SynexisImpl;


class Synexis {
public:
    Synexis(const std::string &model_path, int n_slots);


    ~Synexis();

    int addTask(const std::string &prompt, const SamplingParams &sampling_params);

    std::string get_result(int task_id);


    void run() const;

    void stop() const;

private:
    SynexisImpl *impl;
};
