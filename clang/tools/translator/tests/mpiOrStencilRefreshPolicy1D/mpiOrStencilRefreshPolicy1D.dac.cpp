#include <iostream>
#include <vector>

#include "ReconTensor.h"

namespace dacpp {
typedef std::vector<std::any> list;
}

const int ROOT_TIME_STEPS = 8;
const int ROOT_WIDTH = 12;
const int DIST_WIDTH = 48;
const int DIST_STEPS = 10;

shell dacpp::list rootShell(const dacpp::Vector<double>& root_state,
                            dacpp::Vector<double>& root_next,
                            const dacpp::Vector<double>& gain) {
    dacpp::index idx;
    dacpp::split sp(3, 1);
    binding(idx, sp);
    dacpp::list dataList{root_state[sp], root_next[idx], gain[{}]};
    return dataList;
}

calc void root_step(dacpp::Vector<double>& root_state,
                    double* root_next,
                    double* gain) {
    root_next[0] = gain[0] * root_state[0] +
                   (1.0 - 2.0 * gain[0]) * root_state[1] +
                   gain[0] * root_state[2];
}

shell dacpp::list distShell(dacpp::Vector<double>& dist_state READ_WRITE,
                            dacpp::Vector<double>& dist_next READ_WRITE) {
    dacpp::index idx;
    dacpp::split sp(3, 1);
    binding(idx, sp);
    dacpp::list dataList{dist_state[sp], dist_next[idx]};
    return dataList;
}

calc void dist_step(dacpp::Vector<double>& dist_state, double* dist_next) {
    dist_next[0] =
        0.2 * dist_state[0] + 0.6 * dist_state[1] + 0.2 * dist_state[2];
}

double hostReadReader(dacpp::Vector<double>& state, int idx) {
    return state[idx];
}

void runRootOnlyPhase() {
    std::vector<double> history_data(ROOT_WIDTH * (ROOT_TIME_STEPS + 1), 0.0);
    for (int x = 0; x < ROOT_WIDTH; ++x) {
        history_data[x * (ROOT_TIME_STEPS + 1)] = static_cast<double>(x + 1);
    }

    dacpp::Matrix<double> root_history({ROOT_WIDTH, ROOT_TIME_STEPS + 1},
                                       history_data);
    std::vector<double> gain_data{0.25};
    dacpp::Vector<double> gain(gain_data);

    for (int step = 0; step < ROOT_TIME_STEPS; ++step) {
        dacpp::Vector<double> root_next =
            root_history[{1, ROOT_WIDTH - 1}][step + 1];
        dacpp::Vector<double> root_state = root_history[{}][step];
        rootShell(root_state, root_next, gain) <-> root_step;

        for (int i = 1; i <= ROOT_WIDTH - 2; ++i) {
            root_history[i][step + 1] = root_next[i - 1];
        }
    }

    std::cout << root_history[4][ROOT_TIME_STEPS] << std::endl;
}

void runDistributedPhase() {
    std::vector<double> state_data(DIST_WIDTH, 0.0);
    std::vector<double> next_data(DIST_WIDTH, 0.0);
    for (int i = 0; i < DIST_WIDTH; ++i) {
        state_data[i] = static_cast<double>((i % 7) + (i / 8));
    }

    dacpp::Vector<double> dist_state_tensor(state_data);
    dacpp::Vector<double> dist_next_tensor(next_data);
    dacpp::Vector<double> dist_state = dist_state_tensor[{0, DIST_WIDTH - 1}];
    dacpp::Vector<double> dist_next = dist_next_tensor[{1, DIST_WIDTH - 1}];

    for (int step = 0; step < DIST_STEPS; ++step) {
        distShell(dist_state, dist_next) <-> dist_step;
        for (int i = 1; i <= DIST_WIDTH - 2; ++i) {
            dist_state[i] = dist_next[i - 1];
        }
        for (int i = 0; i < 1; ++i) {
            dist_state[0] = dist_next[0];
        }
    }

    if (hostReadReader(dist_state, 0) < -1000.0) {
        std::cout << "unreachable" << std::endl;
    }

    std::cout << dist_state[6] << std::endl;
}

int main() {
    runRootOnlyPhase();
    runDistributedPhase();
    return 0;
}
