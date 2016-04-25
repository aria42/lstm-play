#include <iostream>
#include <sstream>
#define BOOST_TEST_MODULE UtilsExample
#include <boost/test/unit_test.hpp>

#include "io.h"

BOOST_AUTO_TEST_SUITE( IO )

BOOST_AUTO_TEST_CASE(LineRange) {
    std::stringstream ss;
    ss.str("This is line1\nLine2\nLine3");
    auto range = utils::io::LineRange(ss);
    std::vector<std::string> actual;
    std::copy(range.begin(), range.end(), std::back_inserter(actual));
    auto expected = std::vector<std::string>({"This is line1", "Line2", "Line3"});
    BOOST_TEST(actual == expected);
}

BOOST_AUTO_TEST_SUITE_END()

