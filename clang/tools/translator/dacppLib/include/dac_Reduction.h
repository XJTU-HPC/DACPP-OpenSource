#ifndef DAC_REDUCTION_HPP
#define DAC_REDUCTION_HPP

#include <string>
#include <cstddef>
#include <algorithm>
#include <cctype>
#include <vector>

#include <queue>
namespace dac_reduction {

template <typename T, typename Expr, typename Var>
T reduction_max(Expr, Var, const T&, std::size_t) {
    return T{};
}


} // namespace dac_reduction

#endif // DAC_REDUCTION_HPP