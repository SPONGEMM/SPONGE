// Contract tests for the XTC trajectory writer.
//
// Covers the invariants that matter for a long, interruptible simulation:
// Angstrom-to-nanometre conversion, box layout, frame metadata, flush without
// close, frame index offsets, append semantics and the precision guard.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "io/xtc_trajectory_writer.h"

#include "xdrfile.h"
#include "xdrfile_xtc.h"

namespace
{
constexpr int kAtoms = 32;

struct Readback
{
    std::vector<int> steps;
    std::vector<float> times;
    std::vector<float> boxes;    // 9 values per frame, row major
    std::vector<float> coords;   // 3 * kAtoms per frame, nanometre
};

void Expect(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Fill_Coordinates(std::vector<float>& coordinates, int frame)
{
    coordinates.resize(static_cast<std::size_t>(kAtoms) * 3);
    for (int i = 0; i < kAtoms; ++i)
    {
        const std::size_t base = static_cast<std::size_t>(i) * 3;
        coordinates[base + 0] = 1.5f + 0.25f * static_cast<float>(i) +
                                static_cast<float>(frame);
        coordinates[base + 1] = 2.5f + 0.5f * static_cast<float>(i);
        coordinates[base + 2] = 3.5f - 0.125f * static_cast<float>(i);
    }
}

bool Read_Back(const std::string& path, Readback* out, std::string* error)
{
    int atoms = 0;
    if (read_xtc_natoms(path.c_str(), &atoms) != exdrOK)
    {
        *error = "read_xtc_natoms failed";
        return false;
    }
    if (atoms != kAtoms)
    {
        *error = "atom count mismatch: " + std::to_string(atoms);
        return false;
    }

    XDRFILE* handle = xdrfile_open(path.c_str(), "r");
    if (handle == nullptr)
    {
        *error = "xdrfile_open for read failed";
        return false;
    }

    std::vector<float> frame(static_cast<std::size_t>(kAtoms) * 3, 0.0f);
    matrix box;
    while (true)
    {
        int step = 0;
        float time = 0.0f;
        float precision = 0.0f;
        const int status = read_xtc(handle, kAtoms, &step, &time, box,
                                    reinterpret_cast<rvec*>(frame.data()),
                                    &precision);
        if (status == exdrENDOFFILE)
        {
            break;
        }
        if (status != exdrOK)
        {
            xdrfile_close(handle);
            *error = "read_xtc failed with status " + std::to_string(status);
            return false;
        }
        out->steps.push_back(step);
        out->times.push_back(time);
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                out->boxes.push_back(box[row][column]);
            }
        }
        out->coords.insert(out->coords.end(), frame.begin(), frame.end());
    }
    xdrfile_close(handle);
    return true;
}

// Largest deviation between written coordinates (Angstrom) and what came back
// (nanometre), expressed in nanometre.
float Max_Coordinate_Error(const Readback& readback,
                           const std::vector<std::vector<float>>& written)
{
    float max_error = 0.0f;
    const std::size_t per_frame = static_cast<std::size_t>(kAtoms) * 3;
    for (std::size_t frame = 0; frame < written.size(); ++frame)
    {
        for (std::size_t i = 0; i < per_frame; ++i)
        {
            const float expected = written[frame][i] * 0.1f;
            const float actual = readback.coords[frame * per_frame + i];
            max_error = std::max(max_error, std::fabs(actual - expected));
        }
    }
    return max_error;
}

void Test_Round_Trip(const std::string& directory)
{
    const std::string path = directory + "/xtc_roundtrip.xtc";
    std::remove(path.c_str());
    std::remove((path + ".idx").c_str());

    SpongeXTC::XtcTrajectoryWriter writer;
    std::string error;
    Expect(writer.Open(path, kAtoms, 1000.0f, 10, false, true, &error),
           "Open: " + error);

    std::vector<float> coordinates;
    std::vector<std::vector<float>> written;
    const float box[9] = {40.0f, 0.0f, 0.0f, 0.0f, 41.0f, 0.0f,
                          0.0f,  0.0f, 42.0f};
    for (int frame = 1; frame <= 4; ++frame)
    {
        Fill_Coordinates(coordinates, frame);
        written.push_back(coordinates);
        Expect(writer.Append(coordinates.data(), frame * 100,
                             static_cast<float>(frame) * 0.01f, box, &error),
               "Append: " + error);
        Expect(writer.Flush(&error), "Flush: " + error);
    }
    Expect(writer.Frame_Count() == 4, "frame count should be 4");
    Expect(writer.Index_Path() == path + ".idx",
           "index path should be path.idx");
    writer.Close();

    // One 8-byte offset per completed frame.
    std::FILE* index = std::fopen((path + ".idx").c_str(), "rb");
    Expect(index != nullptr, "frame index file should exist");
    std::fseek(index, 0, SEEK_END);
    const long index_bytes = std::ftell(index);
    std::fclose(index);
    Expect(index_bytes == 4 * static_cast<long>(sizeof(std::int64_t)),
           "index should hold one int64 per frame, got " +
               std::to_string(index_bytes) + " bytes");

    Readback readback;
    Expect(Read_Back(path, &readback, &error), "Read_Back: " + error);
    Expect(readback.steps.size() == 4,
           "read back 4 frames, got " +
               std::to_string(readback.steps.size()));
    for (std::size_t i = 0; i < readback.steps.size(); ++i)
    {
        Expect(readback.steps[i] == static_cast<int>((i + 1) * 100),
               "step " + std::to_string(readback.steps[i]));
        Expect(std::fabs(readback.times[i] -
                         static_cast<float>(i + 1) * 0.01f) < 1e-6f,
               "time " + std::to_string(readback.times[i]));
    }
    // Box travels in nanometre.
    Expect(std::fabs(readback.boxes[0] - 4.0f) < 1e-5f,
           "box[0][0] should be 4.0 nm");
    Expect(std::fabs(readback.boxes[4] - 4.1f) < 1e-5f,
           "box[1][1] should be 4.1 nm");
    Expect(std::fabs(readback.boxes[8] - 4.2f) < 1e-5f,
           "box[2][2] should be 4.2 nm");

    const float max_error = Max_Coordinate_Error(readback, written);
    // Quantization step is 0.001 nm, so the round trip must stay within it.
    Expect(max_error < 0.0011f, "coordinate error " +
                                    std::to_string(max_error) +
                                    " nm exceeds the 0.001 nm grid");
}

void Test_Append_Mode(const std::string& directory)
{
    const std::string path = directory + "/xtc_append.xtc";
    std::remove(path.c_str());

    std::vector<float> coordinates;
    const float box[9] = {30.0f, 0.0f, 0.0f, 0.0f, 30.0f, 0.0f,
                          0.0f,  0.0f, 30.0f};
    std::string error;

    {
        SpongeXTC::XtcTrajectoryWriter writer;
        Expect(writer.Open(path, kAtoms, 1000.0f, 10, false, false, &error),
               "Open first run: " + error);
        Fill_Coordinates(coordinates, 1);
        Expect(writer.Append(coordinates.data(), 100, 0.1f, box, &error),
               "Append first run: " + error);
        writer.Close();
    }
    {
        SpongeXTC::XtcTrajectoryWriter writer;
        Expect(writer.Open(path, kAtoms, 1000.0f, 10, true, false, &error),
               "Open append: " + error);
        Fill_Coordinates(coordinates, 2);
        Expect(writer.Append(coordinates.data(), 200, 0.2f, box, &error),
               "Append second run: " + error);
        writer.Close();
    }

    Readback readback;
    Expect(Read_Back(path, &readback, &error),
           "Read_Back after append: " + error);
    Expect(readback.steps.size() == 2,
           "append should yield 2 frames, got " +
               std::to_string(readback.steps.size()));
    Expect(readback.steps[0] == 100 && readback.steps[1] == 200,
           "append should preserve frame steps");

    // A different topology must not be concatenated.
    SpongeXTC::XtcTrajectoryWriter mismatched;
    Expect(!mismatched.Open(path, kAtoms + 1, 1000.0f, 10, true, false,
                            &error),
           "append with a different atom count must fail");
    mismatched.Close();
}

void Test_Precision_Guard(const std::string& directory)
{
    const std::string path = directory + "/xtc_precision.xtc";
    std::remove(path.c_str());

    SpongeXTC::XtcTrajectoryWriter writer;
    std::string error;
    // 1e7 precision with a 40 Angstrom extent scales to 4e7, past the XTC grid.
    Expect(writer.Open(path, kAtoms, 1.0e7f, 10, false, false, &error),
           "Open with large precision: " + error);
    std::vector<float> coordinates(static_cast<std::size_t>(kAtoms) * 3, 20.0f);
    const float box[9] = {40.0f, 0.0f, 0.0f, 0.0f, 40.0f, 0.0f,
                          0.0f,  0.0f, 40.0f};
    Expect(!writer.Append(coordinates.data(), 100, 0.1f, box, &error),
           "Append must reject a precision that overflows the XTC grid");
    writer.Close();

    SpongeXTC::XtcTrajectoryWriter invalid;
    Expect(!invalid.Open(path, kAtoms, 0.0f, 10, false, false, &error),
           "Open must reject a non-positive precision");
    invalid.Close();

    SpongeXTC::XtcTrajectoryWriter no_atoms;
    Expect(!no_atoms.Open(path, 0, 1000.0f, 10, false, false, &error),
           "Open must reject a non-positive atom count");
    no_atoms.Close();
}

void Test_Inert_Writer()
{
    SpongeXTC::XtcTrajectoryWriter writer;
    Expect(!writer.Is_Open(), "a fresh writer should be closed");
    std::string error;
    Expect(writer.Flush(&error), "flushing a closed writer is a no-op");
    writer.Close();
}
}  // namespace

int main(int argc, char** argv)
{
    const std::string directory = argc > 1 ? argv[1] : ".";
    Test_Round_Trip(directory);
    Test_Append_Mode(directory);
    Test_Precision_Guard(directory);
    Test_Inert_Writer();
    std::cout << "test_xtc_trajectory_writer: PASS\n";
    return 0;
}
