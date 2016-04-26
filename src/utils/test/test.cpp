#include <iostream>
#include <sstream>
#define BOOST_TEST_MODULE UtilsExample
#include <boost/test/unit_test.hpp>

#include "io.h"
#include "range.h"

using namespace utils;

BOOST_AUTO_TEST_SUITE( IO )

BOOST_AUTO_TEST_CASE(LineRange) {
    auto in_ptr = std::make_unique<std::stringstream>("This is line1\nLine2\nLine3");
    auto range = io::LineRange(std::move(in_ptr));
    std::vector<std::string> actual;
    std::copy(range.begin(), range.end(), std::back_inserter(actual));
    auto expected = std::vector<std::string>({"This is line1", "Line2", "Line3"});
    BOOST_TEST(actual == expected);
}

BOOST_AUTO_TEST_CASE(GenRange) {
        int cnt = 0;
        auto range = range::GeneratingRange<int>([&cnt](int& value) {
            value = cnt++;
            return value < 10;
        });
        auto accum = range::to_vec(std::move(range));
        std::vector<int> expected;
        for (int idx=0; idx < 10; ++idx) {
            expected.push_back(idx);
        }
        BOOST_TEST(expected == accum);
}

BOOST_AUTO_TEST_CASE(MapRange) {
    int cnt = 0;
    auto range = range::GeneratingRange<int>([&cnt](int& value) {
        value = cnt++;
        return value < 10;
    });
    auto fn = [](int idx) {
        return "" + idx;
    };
    auto out = range::transform<std::string>(std::move(range), std::move(fn));
    std::vector<std::string> expected;
    for (int idx=0; idx < 10; ++idx) {
        expected.push_back("" + idx);
    }
    auto actual = range::to_vec(std::move(out));
    BOOST_TEST(expected == actual);
}


BOOST_AUTO_TEST_SUITE_END()

