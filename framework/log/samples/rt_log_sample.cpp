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

/**
 * @file rt_log_sample.cpp
 * @brief RT logging framework demonstration application
 */

#include <cstdlib>      // for EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // for exception
#include <format>       // for format
#include <iostream>     // for cout, cerr
#include <string>       // for allocator, operator==, char_...
#include <system_error> // for error_code
#include <utility>      // for move
#include <vector>       // for vector

#include <pthread.h>             // for pthread_self, pthread_setnam...
#include <quill/LogMacros.h>     // for QUILL_LOG_INFO, QUILL_LOG_WA...
#include <wise_enum_detail.h>    // for WISE_ENUM_IMPL_IIF_0
#include <wise_enum_generated.h> // for WISE_ENUM_IMPL_LOOP_7, WISE_...

#include <CLI/CLI.hpp> // for App, IsMember, Option::expected
#include <wise_enum.h> // for wise_enum::from_string

// clang-format off
#include "internal_use_only/config.hpp" // for project_name, project_version

// example-begin include-required-headers-1
#include "log/rt_log_macros.hpp"  // For logging macros
#include "log/components.hpp"     // For component/event declarations
#include "log/rt_log.hpp"         // For logger configuration
// example-end include-required-headers-1
// clang-format on

namespace fl = ::framework::log;

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

// example-begin product-struct-1
// Simple user-defined struct for logging demonstration
struct Product {
    std::string name;            //!< Product name
    double price;                //!< Product price in dollars
    std::vector<int> quantities; //!< Available quantities per size/variant

    /**
     * Create a new product with specified details
     *
     * @param[in] product_name Name of the product
     * @param[in] product_price Price in dollars
     * @param[in] product_quantities Available quantities for different variants
     */
    Product(std::string product_name, double product_price, std::vector<int> product_quantities)
            : name(std::move(product_name)), price(product_price),
              quantities(std::move(product_quantities)) {}
};
// example-end product-struct-1

/// @cond HIDE_FROM_DOXYGEN
/**
 * Enable logging support for Product class
 *
 * Registers the Product class with the logging framework to enable
 * direct logging of Product objects. Uses deferred formatting since
 * Product contains only value types (no pointers or references).
 */
// cppcheck-suppress functionStatic
// example-begin product-struct-2
RT_LOGGABLE_DEFERRED_FORMAT(
        Product,
        "Product{{ name: '{}', price: ${:.2f}, quantities: {} }}",
        obj.name,
        obj.price,
        obj.quantities)
// example-end product-struct-2
/// @endcond

namespace {

/**
 * Application-specific logging components
 *
 * Defines the component categories used throughout this application
 * for organizing and filtering log messages by functional area.
 */
DECLARE_LOG_COMPONENT(
        AppComponent, Core, Network, Database, Security, Inventory, UserInterface, ProductManager);

/**
 * Application system events for normal operations
 *
 * Defines events that represent normal system operations and lifecycle
 * events that occur during application execution.
 */
DECLARE_LOG_EVENT(
        AppSystemEvent,
        AppStart,
        AppStop,
        ConfigLoaded,
        ConfigError,
        UserLogin,
        UserLogout,
        InventoryUpdate,
        ProductCreated,
        ProductUpdated);

/**
 * Application error events for exceptional conditions
 *
 * Defines events that represent error conditions and exceptional
 * situations that may occur during application execution.
 */
DECLARE_LOG_EVENT(
        AppErrorEvent,
        InvalidParam,
        NetworkError,
        DatabaseError,
        AuthenticationFailed,
        ProductNotFound,
        InventoryLow,
        ConnectionTimeout);

/**
 * Demonstrates basic logging functionality
 *
 * Shows usage of RT_LOG_* macros for different log levels including
 * DEBUG, INFO, WARN, and ERROR messages with and without parameters.
 */
void demonstrate_basic_logging() {
    RT_LOG_INFO("=== Basic Logging Demo ===");
    RT_LOG_DEBUG("Debug message with value: {}", 42);
    RT_LOG_INFO("Application started successfully");
    RT_LOG_WARN("This is a warning message");
    RT_LOG_ERROR("This is an error message (not a real error!)");
}

/**
 * Demonstrates component-based logging
 *
 * Shows usage of RT_LOGC_* macros to log messages associated with
 * specific system components like Core, Network, Database, and Security.
 */
void demonstrate_component_logging() {
    RT_LOG_INFO("=== Component Logging Demo ===");
    RT_LOGC_INFO(AppComponent::Core, "Core system initialized");
    RT_LOGC_WARN(AppComponent::Network, "Network latency is high: {}ms", 150);
    RT_LOGC_ERROR(AppComponent::Database, "Connection failed, retrying...");
    RT_LOGC_DEBUG(AppComponent::Security, "Security check passed for user: {}", "demo_user");
}

/**
 * Demonstrates event-based logging
 *
 * Shows usage of RT_LOGE_* and RT_LOGEC_* macros to log messages
 * associated with system events and combined component+event logging.
 */
void demonstrate_event_logging() {
    RT_LOG_INFO("=== Event Logging Demo ===");
    RT_LOGE_INFO(AppSystemEvent::AppStart, "Application startup completed in {}ms", 125);
    RT_LOGE_WARN(AppSystemEvent::ConfigLoaded, "Using default configuration");
    RT_LOGE_ERROR(AppErrorEvent::NetworkError, "Failed to connect to remote service");

    // Combined component + event logging
    RT_LOGEC_INFO(
            AppComponent::Security,
            AppSystemEvent::ConfigLoaded,
            "Security configuration loaded: {} rules",
            25);
}

/**
 * Demonstrates user-defined type logging
 *
 * Shows how custom types can be logged using RT_LOGGABLE_DEFERRED_FORMAT
 * macro to enable direct logging of complex objects.
 */
void demonstrate_custom_types() {
    RT_LOG_INFO("=== Custom Type Logging Demo ===");

    // example-begin custom-type-usage-1
    const Product laptop{"Gaming Laptop", 1299.99, {5, 3, 8}};
    const Product mouse{"Wireless Mouse", 29.99, {50, 25}};

    RT_LOG_INFO("Inventory item: {}", laptop);
    RT_LOG_WARN("Low stock alert: {}", mouse);
    RT_LOGC_INFO(AppComponent::Database, "Updated inventory: {}", laptop);

    // Demonstrate product-related events
    RT_LOGE_INFO(AppSystemEvent::ProductCreated, "New product added: {}", laptop.name);
    RT_LOGEC_WARN(
            AppComponent::Inventory,
            AppErrorEvent::InventoryLow,
            "Low stock for product: {}",
            mouse);
    // example-end custom-type-usage-1
}
} // namespace

/**
 * Main application entry point
 *
 * Parses command line arguments and demonstrates various RT logging
 * framework features including basic logging, component logging,
 * event logging, and custom type logging.
 *
 * @param[in] argc Number of command line arguments
 * @param[in] argv Array of command line argument strings
 * @return EXIT_SUCCESS on successful completion, EXIT_FAILURE on error
 */
int main(int argc, const char **argv) {
    try {
        CLI::App app{std::format(
                "RT Logging Demo - {} version {}",
                framework::cmake::project_name,
                framework::cmake::project_version)};

        std::string log_file;
        app.add_option("-f,--file", log_file, "Log to specified file (default: console)");

        std::string log_level = "INFO";
        app.add_option("-l,--level", log_level, "Log level (DEBUG, INFO, WARN, ERROR)")
                ->check(CLI::IsMember({"DEBUG", "INFO", "WARN", "ERROR"}));

        app.set_version_flag(
                "--version",
                std::string{framework::cmake::project_version},
                "Show version information");

        CLI11_PARSE(app, argc, argv);

        // Set thread name for better logging identification (Linux only)
        const int result = pthread_setname_np(pthread_self(), "log_demo_thread");
        if (result != 0) {
            const std::error_code ec(result, std::generic_category());
            std::cerr << std::format("Warning: Failed to set thread name: {}\n", ec.message());
        }

        // Configure logger based on command line options
        auto level = wise_enum::from_string<fl::LogLevel>(log_level).value_or(fl::LogLevel::Info);

        if (log_file.empty()) {
            // Console logging
            fl::Logger::configure(fl::LoggerConfig::console(level).with_colors(true));
            std::cout << std::format("Logging to console with level: {}\n", log_level);
        } else {
            // File logging
            fl::Logger::configure(
                    fl::LoggerConfig::file(log_file.c_str(), level).with_timestamps(true));
            std::cout << std::format("Logging to file: {} with level: {}\n", log_file, log_level);
        }

        // Register components for this demo
        fl::register_component<AppComponent>(fl::LogLevel::Debug);

        // Run demonstrations
        demonstrate_basic_logging();
        demonstrate_component_logging();
        demonstrate_event_logging();
        demonstrate_custom_types();

        RT_LOG_INFO("=== Demo Complete ===");
        fl::Logger::flush();

        if (!log_file.empty()) {
            std::cout << std::format(
                    "Log output saved to: {}\n", fl::Logger::get_actual_log_file());
        }

        return EXIT_SUCCESS;

    } catch (const std::exception &e) {
        std::cerr << std::format("Unhandled exception: {}\n", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown exception occurred\n";
        return EXIT_FAILURE;
    }
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
