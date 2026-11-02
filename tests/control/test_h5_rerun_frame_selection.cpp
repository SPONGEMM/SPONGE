#include <stdexcept>

#include "utils/h5md/rerun_frame_selection.hpp"

static void Require(bool value)
{
    if (!value)
    {
        throw std::runtime_error("requirement failed");
    }
}

static void Test_Start_And_Strip_Advance_Like_A_File_Cursor()
{
    int next = 0;
    auto selection = SpongeH5MD::Select_Next_H5_Rerun_Frame(next, 2, 6);
    Require(selection.has_frame);
    Require(selection.frame_index == 2);
    Require(selection.next_frame_index == 3);
    Require(selection.skipped_frames == 2);

    next = selection.next_frame_index;
    selection = SpongeH5MD::Select_Next_H5_Rerun_Frame(next, 1, 6);
    Require(selection.has_frame);
    Require(selection.frame_index == 4);
    Require(selection.next_frame_index == 5);
    Require(selection.skipped_frames == 1);

    next = selection.next_frame_index;
    selection = SpongeH5MD::Select_Next_H5_Rerun_Frame(next, 0, 6);
    Require(selection.has_frame);
    Require(selection.frame_index == 5);
    Require(selection.next_frame_index == 6);
}

static void Test_Out_Of_Range_Stops_At_Frame_Count()
{
    auto selection = SpongeH5MD::Select_Next_H5_Rerun_Frame(5, 1, 6);
    Require(!selection.has_frame);
    Require(selection.frame_index == -1);
    Require(selection.next_frame_index == 6);
    Require(selection.skipped_frames == 1);

    selection = SpongeH5MD::Select_Next_H5_Rerun_Frame(6, 0, 6);
    Require(!selection.has_frame);
    Require(selection.next_frame_index == 6);
}

static void Test_Negative_Inputs_Are_Clamped()
{
    auto selection = SpongeH5MD::Select_Next_H5_Rerun_Frame(-3, -2, 1);
    Require(selection.has_frame);
    Require(selection.frame_index == 0);
    Require(selection.next_frame_index == 1);
    Require(selection.skipped_frames == 0);
}

int main()
{
    try
    {
        Test_Start_And_Strip_Advance_Like_A_File_Cursor();
        Test_Out_Of_Range_Stops_At_Frame_Count();
        Test_Negative_Inputs_Are_Clamped();
    }
    catch (const std::exception&)
    {
        return 1;
    }
    return 0;
}
