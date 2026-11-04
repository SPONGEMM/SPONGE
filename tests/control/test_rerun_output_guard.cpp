#include <chrono>
#include <fstream>
#include <iostream>

#include "utils/control/rerun_output_guard.hpp"

int main()
{
    namespace fs = std::filesystem;
    const auto dir =
        fs::temp_directory_path() /
        ("sponge-rerun-guard-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const auto input = dir / "input.dat";
    std::ofstream(input) << "trajectory bytes";
    std::error_code symlink_error;
    fs::create_symlink(input, dir / "symbolic", symlink_error);
    const bool symlink_unavailable =
        symlink_error == std::errc::operation_not_permitted ||
        symlink_error == std::errc::permission_denied ||
        symlink_error == std::errc::operation_not_supported;
    if (symlink_unavailable)
        std::cout << "Skipping symlink case: " << symlink_error.message()
                  << "\n";
    fs::create_hard_link(input, dir / "hard");
    bool ok =
        SpongeRerunOutput::Same_File(input.string(), input.string()) &&
        SpongeRerunOutput::Same_File((dir / "./input.dat").string(),
                                     input.string()) &&
        (symlink_unavailable ||
         (!symlink_error &&
          SpongeRerunOutput::Same_File((dir / "symbolic").string(),
                                       input.string()))) &&
        SpongeRerunOutput::Same_File((dir / "hard").string(), input.string()) &&
        !SpongeRerunOutput::Same_File((dir / "new.dat").string(),
                                      input.string());
    std::ofstream(dir / "other.dat") << "trajectory bytes";
    ok = ok && !SpongeRerunOutput::Same_File((dir / "other.dat").string(),
                                             input.string());
    fs::remove_all(dir);
    if (!ok) std::cerr << "rerun output identity check failed\n";
    return ok ? 0 : 1;
}
