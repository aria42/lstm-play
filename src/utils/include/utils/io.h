#pragma "once"

#include <iostream>
#include <istream>

namespace utils {
namespace io {

    class LineRange;

    namespace detail {
        class line_iter: public std::iterator<std::forward_iterator_tag, std::string> {
            public:
            line_iter(LineRange* parent_);
            std::string operator*();
            line_iter operator++();
            bool operator==(const line_iter& other);
            bool operator!=(const line_iter& other) { return !this->operator==(other); }

            private:
            LineRange *parent_;
        };
    };

    class LineRange  {

        using Iterator = detail::line_iter;
        using iter_type = detail::line_iter;
        using value_type = std::string;

        public:
        LineRange(std::unique_ptr<std::istream> in): in_(std::move(in)) {}
        detail::line_iter begin();
        detail::line_iter end();

        private:
        friend class detail::line_iter;
        std::unique_ptr<std::istream> in_;
        mutable std::string value_;
        bool advance();
    };

    template <typename StreamT, typename... Args>
    LineRange line_range(Args&&... args) {
        auto uptr = std::make_unique<StreamT>(std::forward<Args>(args)...);
        return LineRange(std::move(uptr));
    };
}
};