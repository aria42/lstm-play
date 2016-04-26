#pragma once

#include <cassert>

namespace utils {
namespace range {

    template <typename Range, typename OutputIterator>
    void copy(Range&& src, OutputIterator out) {
        std::copy(src.begin(), src.end(), out);
    };

    template <typename Range>
    std::vector<typename Range::value_type> to_vec(Range&& src) {
        using T = typename Range::value_type;
        std::vector<T> out;
        copy(src, std::back_inserter(out));
        return out;
    };

    /// A function that writes sequence of values to T reference. returns
    /// true when there are more elements and false otherwise
    template <typename T>
    using GeneratingFn = std::function<bool(T&)>;

    /// Single pass range
    template <typename OutT>
    class GeneratingRange {

        public:
        struct iterator;

        iterator begin() {
            return iterator{this};
        }

        iterator end() {
            return iterator{nullptr};
        }

        GeneratingRange(GeneratingFn<OutT>&& gen_fn) : gen_fn_(gen_fn) {}

        struct iterator : std::iterator<std::forward_iterator_tag, OutT> {
            GeneratingRange* parent_;

            OutT operator*() {
                return parent_->value_;
            }

            iterator operator++() {
                assert(parent_);
                this->advance();
                return *this;
            }

            void advance() {
                bool has_next = parent_->gen_fn_(parent_->value_);
                if (!has_next) {
                    parent_ = nullptr;
                }
            }

            bool operator==(const iterator& out) {
                return parent_ == out.parent_;
            }

            bool operator!=(const iterator& out) {
                return !this->operator==(out);
            }

            iterator(GeneratingRange* parent) : parent_(parent)  {
                if (parent_) {
                    this->advance();
                }
            }
        };

        using iter_type = iterator;
        using value_type = typename iterator::value_type;


    private:
        std::function<bool(OutT&)> gen_fn_;
        OutT value_;
    };

    template <typename Range>
    using ElemT = typename Range::value_type;

    template <typename T, typename Range>
    GeneratingRange<T> transform(Range&& range, std::function<T(ElemT<Range>)>&& fn) {
        using InT = ElemT<Range>;
        auto it = range.begin();
        auto gen_fn = [range, fn, it](T& out_value) mutable {
            if (it == range.end()) {
                return false;
            }
            InT v = (*it);
            out_value = fn(v);
            ++it;
            return true;
        };
        return GeneratingRange<T>(gen_fn);
    };
}
}


