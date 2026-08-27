#include "api/error.hpp"

#include "methods/method.hpp"

#include <exception>
#include <new>
#include <stdexcept>

namespace vibeqc::api {

vibeqc_status map_exception(std::string* detail) noexcept {
  try {
    throw;
  } catch (const methods::MethodError& error) {
    if (detail != nullptr) *detail = error.what();
    return error.status();
  } catch (const std::bad_alloc& error) {
    if (detail != nullptr) *detail = error.what();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument& error) {
    if (detail != nullptr) *detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    if (detail != nullptr) *detail = error.what();
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    if (detail != nullptr) *detail = "unknown internal exception";
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

}  // namespace vibeqc::api
