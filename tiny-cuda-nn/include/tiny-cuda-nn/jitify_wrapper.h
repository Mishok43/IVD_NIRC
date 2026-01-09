/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   common.h
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */


#pragma once

// A macro is used such that external tools won't end up indenting entire files,
// resulting in wasted horizontal space.

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <filesystem>

#include <cuda_fp16.h>

#pragma comment(lib, "Dbghelp")
#define JITIFY_PRINT_ALL 0
#include "jitify.h"
#pragma comment(lib, "cudart")
#pragma comment(lib, "nvrtc")

// These live in _BootstrapUtils.cpp since they use Falcor includes / namespace,
//    which does not appear to play nice with the CUDA includes / namespace.
extern void logFatal(std::string str);
extern void logError(std::string str);
extern void logOptixWarning(unsigned int level, const char* tag, const char* message, void*);


#define LOCAL_DIR std::string(std::filesystem::canonical(std::filesystem::path(__FILE__).remove_filename()).u8string())

#ifdef _DEBUG
#define RUN_LINEAR_KERNEL(FILENAME, KERNEL_NAME, ...) tcnn::linear_kernel(KERNEL_NAME, __VA_ARGS__);
#else
#define RUN_LINEAR_KERNEL(FILENAME, KERNEL_NAME, ...) JitCacheManager::getInstance().runKernelLinear(LOCAL_DIR, FILENAME, #KERNEL_NAME, __VA_ARGS__);
#endif

template <typename... TemplateArgs>
struct KernelTemplate {};



class JitCacheManager
{
    public:
		static JitCacheManager& getInstance() {
			static JitCacheManager   instance;
			return instance;
		}

		template <typename T, typename... Types>
		void runKernelLinear(const std::string& path, const std::string& filename, const std::string& kernelName, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types... args) {
			runKernelLinearTemplate(path, filename, kernelName, shmem_size, stream, KernelTemplate<>(), n_elements, std::forward<Types>(args)...);
		}

		template <typename... InstantiateTempArgs, typename T, typename... Types>
		void runKernelLinearTemplate(const std::string& dirPath, const std::string& filename, const std::string& kernelName, uint32_t shmem_size, cudaStream_t stream, KernelTemplate<InstantiateTempArgs...> _a, T n_elements, Types... args) {
			if (n_elements <= 0) {
				return;
			}

			bool bExecuted = false;
			do {
				jitify::Program program = impl.program(dirPath + "/" + filename, 0, {"-I" + std::string(CUDA_INC_DIR), "-I"+std::string(dirPath)});
				dim3 grid((n_elements + 127) / 128, 1, 1);
				dim3 block(128, 1, 1);
				using jitify::reflection::type_of;

				try {
					auto t = program.kernel(kernelName)
						.instantiate(jitify::reflection::Type<InstantiateTempArgs>()...)
						.configure(grid, block, 0, stream)
						.launch(
							n_elements,
							args...
						);
				}
				catch (std::runtime_error e) {
					//impl = jitify::JitCache();
					logError(e.what());
					continue;
				}
				break;
			} while (!bExecuted);
		}

    private:
        JitCacheManager() {}                    // Constructor? (the {} brackets) are needed here.

        // C++ 03
        // ========
        // Don't forget to declare these two. You want to make sure they
        // are inaccessible(especially from outside), otherwise, you may accidentally get copies of
        // your singleton appearing.
        JitCacheManager(JitCacheManager const&);              // Don't Implement
        void operator=(JitCacheManager const&); // Don't implement

		jitify::JitCache impl;

        // C++ 11
        // =======
        // We can use the better technique of deleting the methods
        // we don't want.
    public:
        
        // Note: Scott Meyers mentions in his Effective Modern
        //       C++ book, that deleted functions should generally
        //       be public as it results in better error messages
        //       due to the compilers behavior to check accessibility
        //       before deleted status

};
