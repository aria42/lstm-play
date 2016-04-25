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

        public:
        LineRange(std::istream& in);
        detail::line_iter begin();
        detail::line_iter end();

        private:
        friend class detail::line_iter;
        std::istream& in_;
        mutable std::string value_;
        bool advance();
    };
}
};