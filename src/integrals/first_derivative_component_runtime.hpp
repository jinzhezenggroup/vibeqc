#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace vibeqc::integrals {
using ComponentDerivativeDispatch = int (*)(unsigned, const double*, std::size_t, double*);

// Scheduling only: every component/primitive weight visits an existing generated
// derivative dispatcher. No integral recurrence, density formula or screening.
inline bool component_index(double value, std::size_t limit, std::size_t& out) {
  if (!std::isfinite(value) || value < 0 || value >= static_cast<double>(limit) ||
      std::floor(value) != value)
    return false;
  out = static_cast<std::size_t>(value);
  return true;
}
}  // namespace vibeqc::integrals

extern "C" int vibeqc_component_contract_cpu(
    const double* centers, std::size_t atoms, const double* primitives, std::size_t nprimitive,
    const double* aos, std::size_t nao, const std::int64_t* labels, const std::int64_t* bindings,
    std::size_t binding_rows, std::size_t label_count,
    const vibeqc::integrals::ComponentDerivativeDispatch* dispatch, std::size_t libraries,
    unsigned op, const std::int64_t* indices, double weight, std::int64_t nucleus, double* records,
    std::size_t capacity, double* output, std::uint64_t* work) {
  using vibeqc::integrals::component_index;
  if (!centers || !primitives || !aos || !labels || !bindings || !dispatch || !indices ||
      !records || !output || !work || !atoms || !nprimitive || !nao || !libraries || !capacity ||
      capacity > 4096 || op > 3 || !std::isfinite(weight) || !label_count || label_count > 10)
    return 1;
  const auto pairs = label_count * label_count;
  if (binding_rows != 3 * pairs + pairs * pairs) return 1;
  const unsigned rank = op == 3 ? 4 : 2, owners = rank + (op == 2);
  if ((op == 2 && (nucleus < 0 || static_cast<std::size_t>(nucleus) >= atoms)) ||
      (op != 2 && nucleus != -1))
    return 1;
  const double* rows[4]{};
  std::size_t atom[4]{}, begin[4]{}, count[4]{}, terms[4]{};
  std::size_t component_products = 1, primitive_products = 1;
  for (unsigned i = 0; i < rank; ++i) {
    if (indices[i] < 0 || static_cast<std::size_t>(indices[i]) >= nao) return 1;
    rows[i] = aos + 16 * indices[i];
    if (!component_index(rows[i][0], atoms, atom[i]) ||
        !component_index(rows[i][1], nprimitive, begin[i]) ||
        !component_index(rows[i][2], nprimitive + 1, count[i]) || !count[i] ||
        count[i] > nprimitive - begin[i] || !component_index(rows[i][3], 4, terms[i]) || !terms[i])
      return 1;
    component_products *= terms[i];
    // Match the diagnostic's finite primitive admission ceiling, without overflow.
    if (count[i] > (std::uint64_t{1} << 40) / primitive_products) return 1;
    primitive_products *= count[i];
  }
  if (op == 2) atom[2] = static_cast<std::size_t>(nucleus);
  if (primitive_products > (std::uint64_t{1} << 40) / component_products) return 1;
  double total[12]{};
  std::uint64_t evaluated = 0;
  for (std::size_t component = 0; component < component_products; ++component) {
    std::size_t selected[4]{}, remainder = component;
    for (unsigned i = rank; i-- > 0;) {
      selected[i] = remainder % terms[i];
      remainder /= terms[i];
    }
    std::size_t slot = 0;
    double normalization = 1;
    for (unsigned i = 0; i < rank; ++i) {
      const auto label = labels[3 * indices[i] + selected[i]];
      if (label < 0 || static_cast<std::size_t>(label) >= label_count) return 1;
      slot = label_count * slot + static_cast<std::size_t>(label);
      normalization *= rows[i][7 + 4 * selected[i]];
    }
    normalization *= weight;
    const auto* binding = bindings + 9 * (op * pairs + slot);
    if (binding[0] < 0 || static_cast<std::size_t>(binding[0]) >= libraries || binding[1] < 0 ||
        !dispatch[binding[0]])
      return 1;
    unsigned center_mask = 0, axis_mask = 0;
    for (unsigned i = 0; i < owners; ++i) {
      if (binding[2 + i] < 0 || binding[2 + i] >= owners) return 1;
      center_mask |= 1u << binding[2 + i];
    }
    for (unsigned i = 0; i < 3; ++i) {
      if (binding[6 + i] < 0 || binding[6 + i] >= 3) return 1;
      axis_mask |= 1u << binding[6 + i];
    }
    if (center_mask != (1u << owners) - 1 || axis_mask != 7 || (op == 2 && binding[4] != 2))
      return 1;
    std::fill(records, records + 17 * capacity, 0.0);
    for (std::size_t p = 0; p < capacity; ++p)
      for (unsigned i = 0; i < owners; ++i)
        for (unsigned axis = 0; axis < 3; ++axis)
          records[17 * p + 4 + 3 * i + axis] =
              centers[3 * atom[binding[2 + i]] + binding[6 + axis]];
    std::size_t used = 0;
    for (std::size_t primitive = 0; primitive < primitive_products; ++primitive) {
      std::size_t ids[4]{}, remainder_primitive = primitive;
      for (unsigned i = rank; i-- > 0;) {
        ids[i] = begin[i] + remainder_primitive % count[i];
        remainder_primitive /= count[i];
      }
      auto* record = records + 17 * used;
      double radial = 1;
      for (unsigned i = 0; i < rank; ++i) {
        record[i] = primitives[2 * ids[binding[2 + i]]];
        radial *= primitives[2 * ids[i] + 1];
      }
      record[16] = normalization * radial;
      if (++used == capacity || primitive + 1 == primitive_products) {
        double result[12]{};
        if (dispatch[binding[0]](static_cast<unsigned>(binding[1]), records, used, result))
          return 1;
        for (unsigned i = 0; i < owners; ++i)
          for (unsigned axis = 0; axis < 3; ++axis)
            total[3 * binding[2 + i] + binding[6 + axis]] += result[3 * i + axis];
        evaluated += used;
        used = 0;
      }
    }
  }
  for (double value : total)
    if (!std::isfinite(value)) return 1;
  // Transactional output and work publication, including late dispatch failures.
  std::copy(total, total + 12, output);
  *work = evaluated;
  return 0;
}
