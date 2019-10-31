// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This header contains shared definitions for reading and writing
// via C/C++ API Opaque data types that are registered within ORT
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// This structure is used to initialize and read
// OrtValue of opaque(com.microsoft.automl,DateTimeFeaturizer_TimePoint)
struct DateTimeFeaturizerTimePointData {
  std::int32_t year = 0;            // calendar year
  std::uint8_t month = 0;           // calendar month, 1 through 12
  std::uint8_t day = 0;             // calendar day of month, 1 through 31
  std::uint8_t hour = 0;            // hour of day, 0 through 23
  std::uint8_t minute = 0;          // minute of day, 0 through 59
  std::uint8_t second = 0;          // second of day, 0 through 59
  std::uint8_t amPm = 0;            // 0 if hour is before noon (12 pm), 1 otherwise
  std::uint8_t hour12 = 0;          // hour of day on a 12 basis, without the AM/PM piece
  std::uint8_t dayOfWeek = 0;       // day of week, 0 (Monday) through 6 (Sunday)
  std::uint8_t dayOfQuarter = 0;    // day of quarter, 1 through 92
  std::uint16_t dayOfYear = 0;      // day of year, 1 through 366
  std::uint16_t weekOfMonth = 0;    // week of the month, 0 - 4
  std::uint8_t quarterOfYear = 0;   // calendar quarter, 1 through 4
  std::uint8_t halfOfYear = 0;      // 1 if date is prior to July 1, 2 otherwise
  std::uint8_t weekIso = 0;         // ISO week, see below for details
  std::int32_t yearIso = 0;         // ISO year, see details later
  std::string monthLabel;      // calendar month as string, 'January' through 'December'
  std::string amPmLabel;       // 'am' if hour is before noon (12 pm), 'pm' otherwise
  std::string dayOfWeekLabel;  // day of week as string
  std::string holidayName;     // If a country is provided, we check if the date is a holiday
  std::uint8_t isPaidTimeOff = 0;   // If its a holiday, is it PTO
};

#ifdef __cplusplus
} // extern "C"
#endif
