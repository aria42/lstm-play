#pragma once

#include <cassert>

namespace utils {
namespace range {

    template <typename Range, typename OutputIterator>
    void copy(Range& src, OutputIterator out) {
        std::copy(src.begin(), src.end(), out);
    };

    template <typename Range>
    std::vector<typename Range::value_type> to_vec(Range& src) {
        using T = typename Range::value_type;
        std::vector<T> out;
        copy(src, std::back_inserter(out));
        return out;
    };

    /// A function that writes sequence of values to T reference. The lifetime of the
    /// reference is valid until the IteratorFn is called again. Returns true
    /// if a valid reference was written and false when we're out of elements.
    /// Nothing is written to T when the function returns false.
    template <typename T>
    using IteratorFn = std::function<bool(T&)>;

    /// A function that yields a new T when called
    template <typename T>
    using GeneratorFn = std::function<T()>;

    /// The iterators here are `fat`, they contain
    /// a copy of a generator function and a pointer to the value
    template <typename OutT>
    class GeneratingRange {

        public:
        struct iterator;

        iterator begin() {
            return iterator{std::move(gen_fn_())};
        }

        iterator end() {
            return iterator{nullptr};
        }

        GeneratingRange(GeneratorFn<IteratorFn<OutT>>&& gen_fn)  {
            gen_fn_ = std::move(gen_fn);
        }

        struct iterator : std::iterator<std::forward_iterator_tag, OutT> {
            IteratorFn<OutT> iter_fn_;
            OutT* value_;

            const OutT& operator*() {
                assert(value_);
                return *value_;
            }

            iterator operator++() {
                this->advance();
                return *this;
            }

            void advance() {
                bool has_next = iter_fn_(*value_);
                if (!has_next) {
                    iter_fn_ = nullptr;
                }
            }

            bool operator==(const iterator& out) {
                if (!iter_fn_ || !out.iter_fn_) {
                    return !iter_fn_ && !out.iter_fn_;
                }
                return &iter_fn_ == &out.iter_fn_;
            }

            bool operator!=(const iterator& out) {
                return !this->operator==(out);
            }

            iterator(IteratorFn<OutT>&& iter_fn) {
                if (iter_fn != nullptr) {
                    iter_fn_ = std::move(iter_fn);
                    this->advance();
                }
            }
        };

        using iter_type = iterator;
        using value_type = typename iterator::value_type;


    private:
        GeneratorFn<IteratorFn<OutT>> gen_fn_;
    };

    template <typename Range>
    using ElemT = typename Range::value_type;

    /// Transform the elements of a range using a function. The returned
    /// range will 'cosnume' the source iterator (a closure will be the sole
    /// owner of the range), thus the Range&&
    template <typename T, typename Range>
    GeneratingRange<T> transform(Range&& range, std::function<T(ElemT<Range>&)>&& fn) {
        // The outer function owns the range and function and will live
        // as long as the returned range does
        auto gen_fn = [&r = range, &f = fn]() {
            return [it = r.begin(), end = r.end(), cur = T(), &f](T& output) mutable {
                if (it == end) {
                    return false;
                }
                auto in = *it;
                cur = f(in);
                output = cur;
                ++it;
                return true;
            };
        };
        return GeneratingRange<T>(std::move(gen_fn));
    };
}
}


