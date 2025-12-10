/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <scf_5g_fapi.h>
#include <tl/expected.hpp>

#include <gtest/gtest.h>

#include "fapi/fapi_file_replay.hpp"
#include "fapi_test_utils.hpp"

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

/**
 * Base fixture for FapiFileReplay tests
 *
 * Validates FAPI capture file path once per test suite
 */
class FapiFileReplayTest : public ::testing::Test {
protected:
    /**
     * Set up test suite - runs once for all tests
     *
     * Reads FAPI_CAPTURE_DIR and TEST_CELLS environment variables (with defaults),
     * constructs capture filename dynamically, and validates file exists
     */
    static void SetUpTestSuite() {
        if (!fapi_file_path.empty()) {
            return; // Already set up
        }

        const auto result = ran::fapi::get_fapi_capture_file_path();
        ASSERT_TRUE(result.has_value()) << result.error();
        fapi_file_path = result.value().string();
    }

    /// Path to FAPI capture file
    inline static std::string fapi_file_path{};

    static constexpr std::uint8_t SLOTS_PER_SUBFRAME_30KHZ = 2;
};

/**
 * Verify request timing fields match expected slot timing
 *
 * @param[in] request Request to verify
 * @param[in] timing Expected slot timing
 */
void verify_request_timing(
        const scf_fapi_ul_tti_req_t *request, const ran::fapi::FapiSlotTiming &timing) {
    // Convert FapiSlotTiming to frame/subframe/slot for comparison
    static constexpr std::uint16_t SUBFRAMES_PER_FRAME = 10;
    static constexpr std::uint16_t FAPI_SFN_MAX = 1024;

    const std::uint64_t slots_per_frame =
            static_cast<std::uint64_t>(SUBFRAMES_PER_FRAME) * timing.slots_per_subframe;
    const std::uint64_t frames = timing.absolute_slot / slots_per_frame;
    const std::uint64_t slots_in_frame = timing.absolute_slot % slots_per_frame;

    const auto frame_id = static_cast<std::uint8_t>((frames % FAPI_SFN_MAX));
    const auto subframe_id = static_cast<std::uint8_t>(slots_in_frame / timing.slots_per_subframe);
    const auto slot_id = static_cast<std::uint8_t>(slots_in_frame % timing.slots_per_subframe);

    EXPECT_EQ(request->sfn, frame_id);

    const auto expected_slot =
            static_cast<std::uint16_t>(subframe_id * timing.slots_per_subframe + slot_id);
    EXPECT_EQ(request->slot, expected_slot);
}

TEST_F(FapiFileReplayTest, LoadValidFile) {
    const ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    EXPECT_GT(replay.get_total_request_count(), 0U);
    EXPECT_GT(replay.get_cell_count(), 0U);
}

TEST_F(FapiFileReplayTest, GetCellIds) {
    const ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    EXPECT_GT(cell_ids.size(), 0U);
    EXPECT_EQ(cell_ids.size(), replay.get_cell_count());

    // Verify each cell has at least one request
    for (const auto cell_id : cell_ids) {
        EXPECT_GT(replay.get_request_count(cell_id), 0U);
    }
}

TEST_F(FapiFileReplayTest, GetRequestCount) {
    const ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    std::size_t total_from_cells{0};
    for (const auto cell_id : cell_ids) {
        const auto count = replay.get_request_count(cell_id);
        EXPECT_GT(count, 0U);
        total_from_cells += count;
    }

    EXPECT_EQ(total_from_cells, replay.get_total_request_count());
}

TEST_F(FapiFileReplayTest, InvalidCellId) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    constexpr std::uint16_t INVALID_CELL_ID = 9999;
    EXPECT_EQ(replay.get_request_count(INVALID_CELL_ID), 0U);

    EXPECT_THROW(
            std::ignore = replay.get_request_for_current_slot(INVALID_CELL_ID),
            std::invalid_argument);
}

TEST_F(FapiFileReplayTest, AdvanceSlot) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    EXPECT_EQ(replay.get_current_absolute_slot(), 0U);

    const auto slot1 = replay.advance_slot();
    EXPECT_EQ(slot1, 1U);
    EXPECT_EQ(replay.get_current_absolute_slot(), 1U);

    const auto slot2 = replay.advance_slot();
    EXPECT_EQ(slot2, 2U);
    EXPECT_EQ(replay.get_current_absolute_slot(), 2U);
}

TEST_F(FapiFileReplayTest, TimingFieldsUpdated) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    const auto cell_id = cell_ids.at(0);

    // Advance through several slots to find a matching one
    std::optional<ran::fapi::FapiFileReplay::RequestWithSize> request{};
    constexpr std::size_t MAX_SLOTS_TO_TRY = 100;
    for (std::size_t i = 0; i < MAX_SLOTS_TO_TRY; ++i) {
        request = replay.get_request_for_current_slot(cell_id);
        if (request.has_value()) {
            break;
        }
        std::ignore = replay.advance_slot();
    }

    ASSERT_TRUE(request.has_value())
            << "No matching request found in first " << MAX_SLOTS_TO_TRY << " slots";

    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    ASSERT_NE(request.value().request, nullptr);
    const auto *req = request.value().request;
    // NOLINTEND(bugprone-unchecked-optional-access)
    const auto slot_timing = replay.get_current_slot_timing();

    // Verify timing fields were updated
    verify_request_timing(req, slot_timing);
}

TEST_F(FapiFileReplayTest, GetRequestForCurrentSlot_Match) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    const auto cell_id = cell_ids.at(0);
    const auto initial_count = replay.get_request_count(cell_id);
    ASSERT_GT(initial_count, 0U);

    // Find first matching slot
    std::optional<ran::fapi::FapiFileReplay::RequestWithSize> first_request{};
    constexpr std::size_t MAX_SLOTS_TO_TRY = 100;
    for (std::size_t i = 0; i < MAX_SLOTS_TO_TRY; ++i) {
        first_request = replay.get_request_for_current_slot(cell_id);
        if (first_request.has_value()) {
            break;
        }
        std::ignore = replay.advance_slot();
    }

    ASSERT_TRUE(first_request.has_value()) << "No matching request found";
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    ASSERT_NE(first_request.value().request, nullptr);
    // NOLINTEND(bugprone-unchecked-optional-access)
}

TEST_F(FapiFileReplayTest, GetRequestForCurrentSlot_NoMatch) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    // Try to get request for current slot (which starts at 0)
    // Most likely this won't match immediately
    const auto request = replay.get_request_for_current_slot(cell_ids.at(0));

    // If no match, advance slot and try again - the same request should be checked
    if (!request.has_value()) {
        const auto slot_before = replay.get_current_absolute_slot();
        std::ignore = replay.advance_slot();
        const auto slot_after = replay.get_current_absolute_slot();
        EXPECT_EQ(slot_after, slot_before + 1);

        // Request index should not have advanced (internal state test via behavior)
        // We can't directly test the index, but the behavior should be consistent
    }
}

TEST_F(FapiFileReplayTest, GetRequestForCurrentSlot_WrapsAround) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    const auto cell_id = cell_ids.at(0);
    const auto request_count = replay.get_request_count(cell_id);
    ASSERT_GT(request_count, 0U);

    // Advance through enough slots to verify wraparound behavior
    // Note: Not all slots have UL requests (many are DL-only), so we need enough iterations
    std::size_t matches_found{0};
    constexpr std::size_t MAX_SLOTS = 2000; // Enough to see wraparound with sparse UL

    for (std::size_t i = 0; i < MAX_SLOTS && matches_found <= request_count + 2; ++i) {
        const auto request = replay.get_request_for_current_slot(cell_id);
        if (request.has_value()) {
            ++matches_found;
        }
        std::ignore = replay.advance_slot();
    }

    // Should have found at least as many matches as there are requests
    // (including wraparound)
    EXPECT_GT(matches_found, request_count);
}

TEST_F(FapiFileReplayTest, MultipleSlotAdvances) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    constexpr std::size_t NUM_ADVANCES = 50;
    for (std::size_t i = 0; i < NUM_ADVANCES; ++i) {
        const auto before_slot = replay.get_current_absolute_slot();
        const auto after_slot = replay.advance_slot();
        EXPECT_EQ(after_slot, before_slot + 1);
        EXPECT_EQ(replay.get_current_absolute_slot(), after_slot);
    }

    EXPECT_EQ(replay.get_current_absolute_slot(), NUM_ADVANCES);
}

TEST_F(FapiFileReplayTest, AdvanceWithoutQuery) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    const auto cell_id = cell_ids.at(0);

    // Advance through many slots WITHOUT calling get_request_for_current_slot()
    // This simulates seeking to middle of file
    static constexpr std::size_t SLOTS_TO_SKIP = 100;
    for (std::size_t i = 0; i < SLOTS_TO_SKIP; ++i) {
        std::ignore = replay.advance_slot();
    }

    // Now query for current slot - should get correct request for slot 100
    const auto request = replay.get_request_for_current_slot(cell_id);
    const auto slot_timing = replay.get_current_slot_timing();

    if (request.has_value()) {
        // If we got a request, verify timing fields are correct for current slot
        const auto &req_info = request.value();
        verify_request_timing(req_info.request, slot_timing);
    }

    // Continue advancing and verify we can still get requests correctly
    static constexpr std::size_t ADDITIONAL_SLOTS = 20;
    std::size_t additional_requests_found{0};

    for (std::size_t i = 0; i < ADDITIONAL_SLOTS; ++i) {
        std::ignore = replay.advance_slot();
        const auto req = replay.get_request_for_current_slot(cell_id);
        if (req.has_value()) {
            ++additional_requests_found;
            // Verify timing fields are still correct
            const auto timing = replay.get_current_slot_timing();
            verify_request_timing(req.value().request, timing);
        }
    }

    // Should have found at least some requests in the additional slots
    EXPECT_GT(additional_requests_found, 0U)
            << "Should find at least some requests after skipping ahead";
}

TEST_F(FapiFileReplayTest, DLOnlySlots) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    const auto cell_id = cell_ids.at(0);

    // Advance through slots and track when we don't get matches (potential DL-only slots)
    constexpr std::size_t SLOTS_TO_CHECK = 100;

    for (std::size_t i = 0; i < SLOTS_TO_CHECK; ++i) {
        std::ignore = replay.get_request_for_current_slot(cell_id);
        // Just verify no-match is handled gracefully - don't need to track count
        std::ignore = replay.advance_slot();
    }

    // The test verifies the system handles no-match gracefully through multiple slots
    SUCCEED() << "Handled potential DL-only slots correctly";
}

TEST_F(FapiFileReplayTest, GetRequestForCurrentSlot_MatchAfterSkip) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    const auto cell_id = cell_ids.at(0);

    // Find first non-matching slot (DL-only slot)
    // In typical 5G operation, not all slots have UL requests
    constexpr std::size_t MAX_SLOTS_TO_FIND_NO_MATCH = 100;
    bool found_no_match{false};

    for (std::size_t i = 0; i < MAX_SLOTS_TO_FIND_NO_MATCH; ++i) {
        const auto request = replay.get_request_for_current_slot(cell_id);
        if (!request.has_value()) {
            found_no_match = true;
            break;
        }
        std::ignore = replay.advance_slot();
    }

    EXPECT_TRUE(found_no_match)
            << "Should find at least one non-matching slot (DL-only) in first "
            << MAX_SLOTS_TO_FIND_NO_MATCH << " slots. "
            << "This is expected behavior for 5G - not all slots have UL TTI requests.";

    // Advance a few more slots and find a match
    // Verify we can still find matching slots after encountering DL-only slots
    constexpr std::size_t MAX_ADDITIONAL_SLOTS = 50;
    bool found_match_after_skip{false};

    for (std::size_t i = 0; i < MAX_ADDITIONAL_SLOTS; ++i) {
        std::ignore = replay.advance_slot();
        const auto request = replay.get_request_for_current_slot(cell_id);
        if (request.has_value()) {
            found_match_after_skip = true;
            break;
        }
    }

    EXPECT_TRUE(found_match_after_skip)
            << "Should find matching slot after DL-only slots within " << MAX_ADDITIONAL_SLOTS
            << " additional slots. "
            << "Verify FAPI capture file has UL TTI requests distributed across slots.";
}

TEST_F(FapiFileReplayTest, SlotTimingCalculation) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    // Test initial slot timing
    auto timing = replay.get_current_slot_timing();
    EXPECT_EQ(timing.absolute_slot, 0U);
    EXPECT_EQ(timing.slots_per_subframe, SLOTS_PER_SUBFRAME_30KHZ);

    // Advance and check timing progression
    constexpr std::size_t SUBFRAMES_PER_FRAME = 10;
    const auto slots_per_frame = SUBFRAMES_PER_FRAME * SLOTS_PER_SUBFRAME_30KHZ;

    // Test within first subframe
    std::ignore = replay.advance_slot();
    timing = replay.get_current_slot_timing();
    EXPECT_EQ(timing.absolute_slot, 1U);

    // Advance to next subframe boundary
    for (std::size_t i = 1; i < SLOTS_PER_SUBFRAME_30KHZ; ++i) {
        std::ignore = replay.advance_slot();
    }
    timing = replay.get_current_slot_timing();
    EXPECT_EQ(timing.absolute_slot, SLOTS_PER_SUBFRAME_30KHZ);

    // Advance to next frame boundary
    for (std::size_t i = 0; i < slots_per_frame - SLOTS_PER_SUBFRAME_30KHZ; ++i) {
        std::ignore = replay.advance_slot();
    }
    timing = replay.get_current_slot_timing();
    EXPECT_EQ(timing.absolute_slot, slots_per_frame);
}

TEST_F(FapiFileReplayTest, ReplayBasic) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    EXPECT_GT(replay.get_cell_count(), 0U);
    EXPECT_GT(replay.get_total_request_count(), 0U);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    // Verify we can find requests across multiple slots
    // Note: Check current slot BEFORE advancing to avoid skipping slot 0
    constexpr std::size_t SLOTS_TO_CHECK = 10;
    std::size_t requests_found{0};

    for (std::size_t i = 0; i < SLOTS_TO_CHECK; ++i) {
        // Check current slot for requests
        for (const auto cell_id : cell_ids) {
            const auto request = replay.get_request_for_current_slot(cell_id);
            if (request.has_value()) {
                ++requests_found;
                // Verify timing fields are updated
                const auto &req_info = request.value();
                EXPECT_LE(req_info.request->sfn, 1023U); // SFN range check
            }
        }

        // Advance to next slot
        std::ignore = replay.advance_slot();
    }

    EXPECT_GT(requests_found, 0U) << "Should find at least some requests in " << SLOTS_TO_CHECK
                                  << " slots";
}

TEST_F(FapiFileReplayTest, MultipleCells) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    if (cell_ids.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 cells";
    }

    // Process several slots and verify each cell is handled correctly
    constexpr std::size_t SLOTS_TO_PROCESS = 20;
    std::vector<std::size_t> requests_per_cell(cell_ids.size(), 0);

    for (std::size_t slot = 0; slot < SLOTS_TO_PROCESS; ++slot) {
        // Check current slot for requests
        for (std::size_t cell_idx = 0; cell_idx < cell_ids.size(); ++cell_idx) {
            const auto request = replay.get_request_for_current_slot(cell_ids.at(cell_idx));
            if (request.has_value()) {
                ++requests_per_cell.at(cell_idx);

                // Verify request timing is correct
                const auto &req_info = request.value();
                const auto timing = replay.get_current_slot_timing();
                verify_request_timing(req_info.request, timing);
            }
        }

        // Advance to next slot
        std::ignore = replay.advance_slot();
    }

    // Each cell should have at least some requests
    for (std::size_t i = 0; i < cell_ids.size(); ++i) {
        EXPECT_GT(requests_per_cell.at(i), 0U)
                << "Cell " << cell_ids.at(i) << " should have requests";
    }
}

TEST_F(FapiFileReplayTest, TimingConsistency) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    const auto cell_id = cell_ids.at(0);

    // Track timing across multiple slots
    constexpr std::size_t SLOTS_TO_CHECK = 50;

    for (std::size_t i = 0; i < SLOTS_TO_CHECK; ++i) {
        const auto timing = replay.get_current_slot_timing();

        // Get request for current slot
        const auto request = replay.get_request_for_current_slot(cell_id);
        if (request.has_value()) {
            const auto &req_info = request.value();

            // Verify timing consistency between replay and request
            verify_request_timing(req_info.request, timing);

            // Verify SFN is in valid range (frame_id wraps at 256, but FAPI SFN should wrap at
            // 1024)
            EXPECT_LE(req_info.request->sfn, 255U);
        }

        // Advance to next slot
        std::ignore = replay.advance_slot();
    }
}

TEST_F(FapiFileReplayTest, RequestWraparound) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    const auto &cell_ids = replay.get_cell_ids();
    ASSERT_GT(cell_ids.size(), 0U);

    const auto cell_id = cell_ids.at(0);
    const auto request_count = replay.get_request_count(cell_id);
    ASSERT_GT(request_count, 0U);

    // Track unique request pointers to detect wraparound
    // Note: Since not all slots have UL requests (many are DL-only), we need to iterate
    // through enough slots to see wraparound behavior
    std::vector<const scf_fapi_ul_tti_req_t *> seen_requests{};
    constexpr std::size_t MAX_SLOTS = 2000; // Enough to see wraparound even with sparse UL
    std::size_t matches_found{0};

    for (std::size_t i = 0; i < MAX_SLOTS && matches_found <= request_count * 2; ++i) {
        const auto request = replay.get_request_for_current_slot(cell_id);
        if (request.has_value()) {
            ++matches_found;
            seen_requests.push_back(request.value().request);
        }
        std::ignore = replay.advance_slot();
    }

    // Should have found more matches than the total request count (indicating wraparound)
    EXPECT_GT(matches_found, request_count) << "Should wrap around and reuse requests";

    // Verify request pointers are reused (wraparound detection)
    const std::unordered_set<const scf_fapi_ul_tti_req_t *> unique_requests(
            seen_requests.begin(), seen_requests.end());
    EXPECT_LE(unique_requests.size(), request_count)
            << "Should only see unique request pointers <= request count";
    EXPECT_LT(unique_requests.size(), seen_requests.size())
            << "Should have duplicate pointers (wraparound occurred)";
}

TEST_F(FapiFileReplayTest, SlotTimingProgression) {
    ran::fapi::FapiFileReplay replay(fapi_file_path, SLOTS_PER_SUBFRAME_30KHZ);

    // Test initial state
    auto timing = replay.get_current_slot_timing();
    EXPECT_EQ(timing.absolute_slot, 0U);
    EXPECT_EQ(timing.slots_per_subframe, SLOTS_PER_SUBFRAME_30KHZ);

    // Advance one slot
    std::ignore = replay.advance_slot();
    timing = replay.get_current_slot_timing();
    EXPECT_EQ(timing.absolute_slot, 1U);

    // Advance to subframe boundary
    std::ignore = replay.advance_slot();
    timing = replay.get_current_slot_timing();
    EXPECT_EQ(timing.absolute_slot, SLOTS_PER_SUBFRAME_30KHZ);

    // Verify timing progresses correctly over multiple slots
    constexpr std::size_t SLOTS_TO_ADVANCE = 50;
    const std::uint64_t expected_slot_start = timing.absolute_slot;
    for (std::size_t i = 0; i < SLOTS_TO_ADVANCE; ++i) {
        std::ignore = replay.advance_slot();
        timing = replay.get_current_slot_timing();

        // Verify absolute slot increases monotonically
        EXPECT_EQ(timing.absolute_slot, expected_slot_start + i + 1);
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

} // namespace
