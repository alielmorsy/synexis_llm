#pragma once
#include <future>
#include <string>

#include "SynexisArguments.h"
#include "TaskParams.h"
class SynexisImpl;


class Synexis {
public:
    Synexis(SynexisArguments args);


    ~Synexis();

    std::future<std::string> addTask(const std::string &prompt, const TaskParams &sampling_params);

    std::string get_result(int task_id);


    void run() const;

    void stop() const;

    [[nodiscard]] std::string getTemplate() const;

    std::string getToken(std::string str) const;

private:
    SynexisImpl *impl;
};
