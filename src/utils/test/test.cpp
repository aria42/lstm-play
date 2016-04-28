#include <iostream>
#include <sstream>
#define BOOST_TEST_MODULE UtilsExample
#include <boost/test/unit_test.hpp>

#include "io.h"
#include "range.h"

using namespace utils;

BOOST_AUTO_TEST_SUITE( IO )

BOOST_AUTO_TEST_CASE(LineRange) {
    auto range = io::line_range<std::stringstream>("This is line1\nLine2\nLine3");
    std::vector<std::string> actual;
    std::copy(range.begin(), range.end(), std::back_inserter(actual));
    auto expected = std::vector<std::string>({"This is line1", "Line2", "Line3"});
    BOOST_TEST(actual == expected);
}

BOOST_AUTO_TEST_CASE(GenRange) {
        auto gen_fn = []() {
            return [cnt = 0](int& value) mutable {
                if (cnt < 10) {
                    value = cnt++;
                    return true;
                }
                return false;
            };
        };
        auto range = range::GeneratingRange<int>(std::move(gen_fn));
        auto accum = range.to_vec();
        std::vector<int> expected;
        for (int idx=0; idx < 10; ++idx) {
            expected.push_back(idx);
        }
        BOOST_TEST(expected == accum);
}

BOOST_AUTO_TEST_CASE(MapRange) {
    auto lines = io::line_range<std::stringstream>("1\n2\n3\n");
    auto ints = range::transform<int>(std::move(lines), [](std::string& v) {
        int idx = atoi(v.c_str());
        return idx;
    });
    std::vector<int> expected({1,2,3});
    BOOST_TEST(expected == ints.to_vec());
}


BOOST_AUTO_TEST_SUITE_END()

