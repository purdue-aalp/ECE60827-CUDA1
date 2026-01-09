/******************************************************************************
 * ECE 60827 CUDA Programming Lab 1 - Automated Grader
 *
 * ╔═══════════════════════════════════════════════════════════════════════╗
 * ║                          ⚠️  DO NOT EDIT  ⚠️                          ║
 * ║                                                                       ║
 * ║  This file is part of the automated grading system.                  ║
 * ║  Any modifications to this file may result in grading failures.      ║
 * ║                                                                       ║
 * ║  Students: Please implement your solutions in:                       ║
 * ║    - src/cudaLib.cu   (GPU implementations)                          ║
 * ║    - src/cpuLib.cpp   (CPU implementations)                          ║
 * ║                                                                       ║
 * ║  This grader will test your implementations automatically.           ║
 * ╚═══════════════════════════════════════════════════════════════════════╝
 *
 *****************************************************************************/

#include "../include/cudaLib.cuh"
#include "../include/cpuLib.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

// Test configuration
struct TestConfig {
    std::vector<int> saxpyVectorSizes;
    std::vector<uint64_t> mcpiThreadCounts;
    std::vector<uint64_t> mcpiSampleSizes;
    double piTolerance;

    TestConfig() {
        // Test sizes range from 2^5 to 2^20 (as per reminder: 2^0 to 2^29-1)
        saxpyVectorSizes = {32, 128, 1024, 4096, 16384, 65536, 262144, 1048576};

        // Monte Carlo test configurations
        mcpiThreadCounts = {256, 1024, 4096};
        mcpiSampleSizes = {1000, 10000, 100000};

        // Tolerance for Pi estimation (should be within reasonable error)
        piTolerance = 0.1;  // Allow 0.1 difference from actual Pi
    }
};

class Grader {
private:
    int totalTests;
    int passedTests;
    int failedTests;
    TestConfig config;

    void printTestHeader(const std::string& testName) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "  " << testName << "\n";
        std::cout << std::string(60, '=') << "\n";
    }

    void printTestResult(const std::string& testName, bool passed, const std::string& message = "") {
        totalTests++;
        if (passed) {
            passedTests++;
            std::cout << "[PASS] " << testName;
        } else {
            failedTests++;
            std::cout << "[FAIL] " << testName;
        }
        if (!message.empty()) {
            std::cout << " - " << message;
        }
        std::cout << "\n";
    }

    bool testSaxpySingleSize(int vectorSize) {
        std::cout << "\n  Testing SAXPY with vector size: " << vectorSize << "\n";

        // Call the student's runGpuSaxpy function
        // This function should:
        // - Allocate GPU memory
        // - Copy data to GPU
        // - Launch the saxpy_gpu kernel
        // - Verify results against CPU
        // - Return 0 on success, non-zero on failure
        int result = runGpuSaxpy(vectorSize);

        // Check for any CUDA errors that may have occurred
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "    [ERROR] CUDA error: " << cudaGetErrorString(err) << "\n";
            return false;
        }

        if (result == 0) {
            std::cout << "    runGpuSaxpy returned success\n";
            return true;
        } else {
            std::cout << "    runGpuSaxpy returned failure (code: " << result << ")\n";
            return false;
        }
    }

    bool testMonteCarloPi(uint64_t threadCount, uint64_t sampleSize) {
        std::cout << "\n  Testing Monte Carlo Pi with threads=" << threadCount
                  << ", samples=" << sampleSize << "\n";

        uint64_t reduceThreadCount = 256;
        uint64_t reduceSize = (threadCount + reduceThreadCount - 1) / reduceThreadCount;

        // Call the student's estimatePi function
        double estimatedPi = estimatePi(threadCount, sampleSize, reduceThreadCount, reduceSize);

        // Check if kernel was actually executed
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "    [ERROR] CUDA error after estimatePi: " << cudaGetErrorString(err) << "\n";
            return false;
        }

        // Check if Pi estimate is reasonable
        const double actualPi = 3.14159265358979323846;
        double error = std::abs(estimatedPi - actualPi);

        std::cout << "    Estimated Pi: " << std::setprecision(10) << estimatedPi << "\n";
        std::cout << "    Actual Pi:    " << std::setprecision(10) << actualPi << "\n";
        std::cout << "    Error:        " << std::setprecision(6) << error << "\n";

        if (estimatedPi == 0.0) {
            std::cout << "    [ERROR] Pi estimate is 0.0 - kernel likely not implemented\n";
            return false;
        }

        if (error > config.piTolerance) {
            std::cout << "    [ERROR] Pi estimate outside tolerance (" << config.piTolerance << ")\n";
            return false;
        }

        std::cout << "    Pi estimate within acceptable tolerance\n";
        return true;
    }

public:
    Grader() : totalTests(0), passedTests(0), failedTests(0) {}

    void checkCudaDevice() {
        // Check for CUDA device
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            std::cout << "\n[CRITICAL ERROR] No CUDA devices found!\n";
            abort();
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "\nCUDA Device: " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    }

    void runPartA() {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          ECE 60827 CUDA Lab 1 Grader - PART A             ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";

        checkCudaDevice();

        // Test SAXPY
        printTestHeader("PART A: SAXPY Tests");

        for (int size : config.saxpyVectorSizes) {
            std::string testName = "SAXPY (size=" + std::to_string(size) + ")";
            bool passed = testSaxpySingleSize(size);
            printTestResult(testName, passed);
        }

        printSummary();
    }

    void runPartB() {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          ECE 60827 CUDA Lab 1 Grader - PART B             ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";

        checkCudaDevice();

        // Test Monte Carlo Pi
        printTestHeader("PART B: Monte Carlo Pi Tests");

        for (size_t i = 0; i < config.mcpiThreadCounts.size() && i < config.mcpiSampleSizes.size(); i++) {
            uint64_t threads = config.mcpiThreadCounts[i];
            uint64_t samples = config.mcpiSampleSizes[i];
            std::string testName = "Monte Carlo Pi (threads=" + std::to_string(threads) +
                                   ", samples=" + std::to_string(samples) + ")";
            bool passed = testMonteCarloPi(threads, samples);
            printTestResult(testName, passed);
        }

        printSummary();
    }

    void runAllTests() {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          ECE 60827 CUDA Programming Lab 1 Grader          ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";

        checkCudaDevice();

        // Test SAXPY
        printTestHeader("PART A: SAXPY Tests");

        for (int size : config.saxpyVectorSizes) {
            std::string testName = "SAXPY (size=" + std::to_string(size) + ")";
            bool passed = testSaxpySingleSize(size);
            printTestResult(testName, passed);
        }

        // Test Monte Carlo Pi
        printTestHeader("PART B: Monte Carlo Pi Tests");

        for (size_t i = 0; i < config.mcpiThreadCounts.size() && i < config.mcpiSampleSizes.size(); i++) {
            uint64_t threads = config.mcpiThreadCounts[i];
            uint64_t samples = config.mcpiSampleSizes[i];
            std::string testName = "Monte Carlo Pi (threads=" + std::to_string(threads) +
                                   ", samples=" + std::to_string(samples) + ")";
            bool passed = testMonteCarloPi(threads, samples);
            printTestResult(testName, passed);
        }

        printSummary();
    }

    void printSummary() {
        // Print summary
        printTestHeader("Test Summary");
        std::cout << "  Total Tests:  " << totalTests << "\n";
        std::cout << "  Passed:       " << passedTests << "\n";
        std::cout << "  Failed:       " << failedTests << "\n";
        std::cout << "  Success Rate: " << std::fixed << std::setprecision(1)
                  << (100.0 * passedTests / totalTests) << "%\n";

        if (passedTests == totalTests) {
            std::cout << "\nALL TESTS PASSED!\n";
        } else {
            std::cout << "\nSOME TESTS FAILED - Please review implementation\n";
        }

        std::cout << "\n" << std::string(60, '=') << "\n";
    }

    int getExitCode() {
        return (failedTests == 0) ? 0 : 1;
    }
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [OPTION]\n";
    std::cout << "Run autograder for ECE 60827 CUDA Lab 1\n\n";
    std::cout << "Options:\n";
    std::cout << "  -a, --part-a     Run only Part A (SAXPY) tests\n";
    std::cout << "  -b, --part-b     Run only Part B (Monte Carlo Pi) tests\n";
    std::cout << "  -h, --help       Display this help message\n";
    std::cout << "  (no arguments)   Run all tests (Parts A and B)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << "           # Run all tests\n";
    std::cout << "  " << programName << " --part-a  # Run only SAXPY tests\n";
    std::cout << "  " << programName << " --part-b  # Run only Monte Carlo Pi tests\n";
}

int main(int argc, char** argv) {
    Grader grader;

    // Parse command-line arguments
    if (argc == 1) {
        // No arguments - run all tests
        grader.runAllTests();
    } else if (argc == 2) {
        std::string arg = argv[1];
        if (arg == "-a" || arg == "--part-a") {
            grader.runPartA();
        } else if (arg == "-b" || arg == "--part-b") {
            grader.runPartB();
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: Unknown option '" << arg << "'\n";
            printUsage(argv[0]);
            return 1;
        }
    } else {
        std::cerr << "Error: Too many arguments\n";
        printUsage(argv[0]);
        return 1;
    }

    return grader.getExitCode();
}
