# unbundle_archive.cmake
# Extracts single-arch objects from a fat static archive.
#
# Handles the new-style HIP compilation model where device code is embedded
# in .hip_fatbin ELF sections (CCOB format) within regular host objects,
# rather than the old clang-offload-bundler fat binary format.
#
# Required -D arguments:
#   FAT_ARCHIVE   - Path to the fat .a file
#   ARCH          - GPU architecture (e.g., gfx90a)
#   OUTPUT        - Path for the output per-arch .a file
#   BUNDLER       - Path to clang-offload-bundler
#   AR            - Path to ar tool
#   LLVM_OBJCOPY  - Path to llvm-objcopy (NOT GNU objcopy, which rewrites ELFs)
#   PATCHER       - Path to patch_fatbin_inplace.py

cmake_minimum_required(VERSION 3.16)

# Find Python3 for the patcher script
find_program(_python3 python3)
if(NOT _python3)
    message(FATAL_ERROR "python3 not found, required for unbundle_archive")
endif()

get_filename_component(work_dir "${OUTPUT}" DIRECTORY)
set(work_dir "${work_dir}/_unbundle_${ARCH}")

# Clean and create work directory
file(REMOVE_RECURSE "${work_dir}")
file(MAKE_DIRECTORY "${work_dir}")

# Extract .o files from fat archive
execute_process(
    COMMAND ${AR} x "${FAT_ARCHIVE}"
    WORKING_DIRECTORY "${work_dir}"
    RESULT_VARIABLE ar_result)
if(NOT ar_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract ${FAT_ARCHIVE}")
endif()

file(GLOB obj_files "${work_dir}/*.o" "${work_dir}/*.obj")

set(thin_objs)
foreach(obj IN LISTS obj_files)
    get_filename_component(obj_name "${obj}" NAME)
    set(fatbin_file "${work_dir}/${obj_name}.fatbin")

    # Extract .hip_fatbin section using llvm-objcopy (GNU objcopy rewrites the
    # ELF in-place even for --dump-section, inflating it by ~500KB per object)
    execute_process(
        COMMAND ${LLVM_OBJCOPY} --dump-section .hip_fatbin=${fatbin_file} "${obj}"
        RESULT_VARIABLE extract_result
        ERROR_QUIET)

    if(NOT extract_result EQUAL 0)
        # No .hip_fatbin section — not a HIP object, skip
        continue()
    endif()

    # List targets in the extracted fatbin bundle
    execute_process(
        COMMAND ${BUNDLER} --type=o "--input=${fatbin_file}" -list
        OUTPUT_VARIABLE targets_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE list_result
        ERROR_QUIET)

    if(NOT list_result EQUAL 0)
        continue()
    endif()

    # Find the target triple for our architecture and collect all targets
    string(REPLACE "\n" ";" target_lines "${targets_output}")
    set(matched_target "")
    set(host_target "")
    set(all_targets "")
    foreach(line IN LISTS target_lines)
        string(STRIP "${line}" line)
        if(line STREQUAL "")
            continue()
        endif()
        list(APPEND all_targets "${line}")
        if(line MATCHES "^host-")
            set(host_target "${line}")
        elseif(line MATCHES "${ARCH}$" OR line MATCHES "${ARCH}:")
            set(matched_target "${line}")
        endif()
    endforeach()

    if(NOT matched_target)
        continue()
    endif()

    # Unbundle: extract all targets into separate files so we can pick just
    # the ones we need (bundler requires all targets to be listed)
    set(unbundle_targets "")
    set(unbundle_outputs "")
    set(device_output "")
    set(host_output "")
    foreach(target IN LISTS all_targets)
        string(REPLACE "/" "_" safe_target "${target}")
        set(out_file "${work_dir}/${obj_name}.${safe_target}")
        if(unbundle_targets)
            set(unbundle_targets "${unbundle_targets},${target}")
        else()
            set(unbundle_targets "${target}")
        endif()
        list(APPEND unbundle_outputs "--output=${out_file}")
        if(target STREQUAL matched_target)
            set(device_output "${out_file}")
        elseif(target STREQUAL host_target)
            set(host_output "${out_file}")
        endif()
    endforeach()

    execute_process(
        COMMAND ${BUNDLER} --type=o
            "--targets=${unbundle_targets}"
            "--input=${fatbin_file}"
            ${unbundle_outputs}
            --unbundle
        RESULT_VARIABLE unbundle_result)

    if(NOT unbundle_result EQUAL 0)
        message(WARNING "Failed to unbundle ${obj_name} for ${ARCH}")
        continue()
    endif()

    # Re-bundle with only host + target arch, compressed
    set(rebundle_targets "${host_target},${matched_target}")
    set(thin_fatbin "${work_dir}/${obj_name}.thin_fatbin")
    execute_process(
        COMMAND ${BUNDLER} --type=o
            "--targets=${rebundle_targets}"
            "--input=${host_output}" "--input=${device_output}"
            "--output=${thin_fatbin}"
            --compress
        RESULT_VARIABLE rebundle_result)

    if(NOT rebundle_result EQUAL 0)
        message(WARNING "Failed to re-bundle ${obj_name} for ${ARCH}")
        continue()
    endif()

    # Patch the .hip_fatbin section in-place (preserves ELF layout exactly,
    # unlike objcopy --update-section which duplicates string tables)
    set(thin_obj "${work_dir}/thin_${obj_name}")
    execute_process(
        COMMAND ${_python3} "${PATCHER}" "${obj}" "${thin_fatbin}" "${thin_obj}"
        RESULT_VARIABLE patch_result)

    if(patch_result EQUAL 0)
        list(APPEND thin_objs "${thin_obj}")
    else()
        message(WARNING "Failed to patch ${obj_name} for ${ARCH}")
    endif()
endforeach()

# Create per-arch archive from patched objects
if(thin_objs)
    execute_process(
        COMMAND ${AR} rcs "${OUTPUT}" ${thin_objs}
        RESULT_VARIABLE ar_result)
    if(NOT ar_result EQUAL 0)
        message(FATAL_ERROR "Failed to create ${OUTPUT}")
    endif()
    list(LENGTH thin_objs count)
    message(STATUS "Created ${OUTPUT} with ${count} objects")
else()
    message(WARNING "No objects found for ${ARCH} in ${FAT_ARCHIVE}, creating empty archive")
    # Create an empty archive so the build does not fail with a missing output.
    execute_process(
        COMMAND ${AR} rcs "${OUTPUT}"
        RESULT_VARIABLE ar_empty_result)
    if(NOT ar_empty_result EQUAL 0)
        message(FATAL_ERROR "Failed to create empty archive ${OUTPUT}")
    endif()
endif()

# Cleanup
file(REMOVE_RECURSE "${work_dir}")
