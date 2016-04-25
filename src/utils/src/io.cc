#pragma "once"

#include <cassert>
#include "io.h"

namespace utils {
namespace io {

    namespace detail {

        line_iter::line_iter(LineRange *parent) : parent_(parent) {
            if (parent_) {
                parent_->advance();
            }
        }

        std::string line_iter::operator*() {
            assert(parent_);
            return parent_->value_;
        }

        line_iter line_iter::operator++() {
            assert(parent_);
            bool has_next = parent_->advance();
            if (!has_next) {
                parent_ = nullptr;
            }
            return *this;
        }

        bool line_iter::operator==(const line_iter &other) {
            return this->parent_ == other.parent_;
        }
    };

    LineRange::LineRange(std::istream &in) : in_(in) { }

    bool LineRange::advance() {
        if (in_.eof()) {
            return false;
        }
        std::getline(in_, value_);
        return true;
    }

    detail::line_iter LineRange::begin() {
        return utils::io::detail::line_iter{this};
    }

    detail::line_iter LineRange::end() {
        return utils::io::detail::line_iter{nullptr};
    }
}
};



