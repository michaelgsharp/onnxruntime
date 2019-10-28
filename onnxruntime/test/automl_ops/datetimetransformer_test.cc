// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/automl/featurizers/src/FeaturizerPrep/Featurizers/DateTimeFeaturizer.h"

namespace dft = Microsoft::Featurizer::Featurizers;

using SysClock = std::chrono::system_clock;

namespace onnxruntime {
namespace test {

TEST(DateTimeTransformer, Past_1976_Nov_17__12_27_04) {

  const time_t date = 217081624;
  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSAutoMLDomain);

  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("X", {1}, {date});

  SysClock::time_point stp = SysClock::from_time_t(date);
  dft::TimePoint tp(stp);

  ASSERT_EQ(tp.year, 1976);
  ASSERT_EQ(tp.month, dft::TimePoint::NOVEMBER);
  ASSERT_EQ(tp.day, 17);
  ASSERT_EQ(tp.hour, 12);
  ASSERT_EQ(tp.minute, 27);
  ASSERT_EQ(tp.second, 4);
  ASSERT_EQ(tp.amPm, 1);
  ASSERT_EQ(tp.hour12, 12);
  ASSERT_EQ(tp.dayOfWeek, dft::TimePoint::WEDNESDAY);
  ASSERT_EQ(tp.dayOfQuarter, 48);
  ASSERT_EQ(tp.dayOfYear, 321);
  ASSERT_EQ(tp.weekOfMonth, 2);
  ASSERT_EQ(tp.quarterOfYear, 4);
  ASSERT_EQ(tp.halfOfYear, 2);
  ASSERT_EQ(tp.weekIso, 47);
  ASSERT_EQ(tp.yearIso, 1976);
  ASSERT_EQ(tp.monthLabel, "November");
  ASSERT_EQ(tp.amPmLabel, "pm");
  ASSERT_EQ(tp.dayOfWeekLabel, "Wednesday");
  ASSERT_EQ(tp.holidayName, "");
  ASSERT_EQ(tp.isPaidTimeOff, 0);

  // Expected output.
  test.AddOutput<dft::TimePoint>("Y", std::move(tp));
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(DateTimeTransformer, Past_1976_Nov_17__12_27_05) {
  const time_t date = 217081625;

  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSAutoMLDomain);
  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("X", {1}, {date});

  dft::DateTimeTransformer dt("", "");
  dft::TimePoint tp(dt.execute(date));
  ASSERT_EQ(tp.year, 1976);
  ASSERT_EQ(tp.month, dft::TimePoint::NOVEMBER);
  ASSERT_EQ(tp.day, 17);
  ASSERT_EQ(tp.hour, 12);
  ASSERT_EQ(tp.minute, 27);
  ASSERT_EQ(tp.second, 5);
  ASSERT_EQ(tp.amPm, 1);
  ASSERT_EQ(tp.hour12, 12);
  ASSERT_EQ(tp.dayOfWeek, dft::TimePoint::WEDNESDAY);
  ASSERT_EQ(tp.dayOfQuarter, 48);
  ASSERT_EQ(tp.dayOfYear, 321);
  ASSERT_EQ(tp.weekOfMonth, 2);
  ASSERT_EQ(tp.quarterOfYear, 4);
  ASSERT_EQ(tp.halfOfYear, 2);
  ASSERT_EQ(tp.weekIso, 47);
  ASSERT_EQ(tp.yearIso, 1976);
  ASSERT_EQ(tp.monthLabel, "November");
  ASSERT_EQ(tp.amPmLabel, "pm");
  ASSERT_EQ(tp.dayOfWeekLabel, "Wednesday");
  ASSERT_EQ(tp.holidayName, "");
  ASSERT_EQ(tp.isPaidTimeOff, 0);

  // Expected output.
  test.AddOutput<dft::TimePoint>("Y", std::move(tp));
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(DateTimeTransformer, Future_2025_June_30) {
  const time_t date = 1751241600;

  OpTester test("DateTimeTransformer", 1, onnxruntime::kMSAutoMLDomain);
  // We are adding a scalar Tensor in this instance
  test.AddInput<int64_t>("X", {1}, {date});

  SysClock::time_point stp = SysClock::from_time_t(date);

  dft::DateTimeTransformer dt("", "");
  dft::TimePoint tp = dt.execute(date);
  ASSERT_EQ(tp.year, 2025);
  ASSERT_EQ(tp.month, dft::TimePoint::JUNE);
  ASSERT_EQ(tp.day, 30);
  ASSERT_EQ(tp.hour, 0);
  ASSERT_EQ(tp.minute, 0);
  ASSERT_EQ(tp.second, 0);
  ASSERT_EQ(tp.dayOfWeek, dft::TimePoint::MONDAY);
  ASSERT_EQ(tp.dayOfYear, 180);
  ASSERT_EQ(tp.quarterOfYear, 2);
  ASSERT_EQ(tp.weekOfMonth, 4);
  ASSERT_EQ(tp.amPm, 0);
  ASSERT_EQ(tp.hour12, 0);
  ASSERT_EQ(tp.dayOfQuarter, 91);
  ASSERT_EQ(tp.halfOfYear, 1);
  ASSERT_EQ(tp.weekIso, 27);
  ASSERT_EQ(tp.yearIso, 2025);
  ASSERT_EQ(tp.monthLabel, "June");
  ASSERT_EQ(tp.amPmLabel, "am");
  ASSERT_EQ(tp.dayOfWeekLabel, "Monday");
  ASSERT_EQ(tp.holidayName, "");
  ASSERT_EQ(tp.isPaidTimeOff, 0);

  test.AddOutput<dft::TimePoint>("Y", std::move(tp));
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


}  // namespace test
}  // namespace onnxruntime
