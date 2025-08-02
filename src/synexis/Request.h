#pragma once

#include <string>
#include <future>
#include "synexis/TaskParams.h"
#include "synexis/sampler/StructParams.h"

struct Request {
    int id;
    std::string prompt;
    TaskParams params;
    std::promise<std::string> promise;
};
