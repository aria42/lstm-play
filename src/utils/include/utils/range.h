#pragma once

#include <cassert>

namespace utils {
namespace range {

    /// A range implicitly represents a sequence (or range) of values.
    /// A range does not necessarily store the elements but does provide a begin/end Iterator.
    /// A range must be re-usable, but not concurrency safe
    /// \tparam Iterator a sub-class std::iterator or at least having standard std::iterator_traits
    template <typename Iterator>
    class range {

        public:
        using iter_type = Iterator;
        using value_type = typename std::iterator_traits<Iterator>::value_type;

        static_assert(std::is_copy_constructible<value_type>::value,
                      "Elem type must be copyable");
        static_assert(std::is_default_constructible<value_type>::value,
                      "Elem type must be default constructible");


        template <typename OutputIterator>
        void copy(OutputIterator out) {
            std::copy(this->begin(), this->end(), out);
        }

        std::vector<value_type> to_vec() {
            std::vector<value_type> out;
            this->copy(std::back_inserter(out));
            return out;
        }

        virtual Iterator begin() = 0;
        virtual Iterator end() = 0;


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

    template <typename T>
    class GeneratingIterator : public std::iterator<std::forward_iterator_tag, T> {

        public:

        const T& operator*() {
            assert(value_);
            return *value_;
        }

        GeneratingIterator operator++() {
            this->advance();
            return *this;
        }

        void advance() {
            bool has_next = iter_fn_(*value_);
            if (!has_next) {
                iter_fn_ = nullptr;
            }
        }

        bool operator==(const GeneratingIterator<T>& out) {
            // if either is null, must both be
            if (!iter_fn_ || !out.iter_fn_) {
                return !iter_fn_ && !out.iter_fn_;
            }
            // if both are non-null, must be ref =
            return &iter_fn_ == &out.iter_fn_;
        }

        bool operator!=(const GeneratingIterator<T>& out) {
            return !this->operator==(out);
        }

        GeneratingIterator(IteratorFn<T>&& iter_fn) {
            if (iter_fn != nullptr) {
                iter_fn_ = std::move(iter_fn);
                this->advance();
            }
        }

        private:
        IteratorFn<T> iter_fn_;
        // Pointer to the value contained by `IteratorFn`
        T* value_;

    };

    /// The iterators here are `fat`, they contain
    /// a copy of a generator function and a pointer to the value
    template <typename T>
    class GeneratingRange : public range<GeneratingIterator<T>> {

        public:

        GeneratingIterator<T> begin() override {
            return GeneratingIterator<T>{std::move(gen_fn_())};
        }

        GeneratingIterator<T> end() override {
            return GeneratingIterator<T>{nullptr};
        }

        GeneratingRange(GeneratorFn<IteratorFn<T>>&& gen_fn)  {
            gen_fn_ = std::move(gen_fn);
        }

        private:
            GeneratorFn<IteratorFn<T>> gen_fn_;
    };

    template <typename Range>
    using ElemT = typename Range::value_type;

    /// Transform the elements of a range using a function. The returned
    /// range will 'cosnume' the source iterator (a closure will be the sole
    /// owner of the range), thus the Range&&
    /// \tparam T must be default constructable and copyable
    /// \tparam Range a valid range
    template <typename T, typename Range>
    GeneratingRange<T> transform(Range&& range, std::function<T(ElemT<Range>&)>&& fn) {
        // The outer function moves + owns the range and transform function
        return GeneratingRange<T>([&r = range, &f = fn]() {
            // The inner lambda owns a mutable iterator and a single T value
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
        });
    };
}
}


