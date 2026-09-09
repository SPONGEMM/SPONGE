#include <fstream>

#include "h5_bundle_test_common.hpp"
#include "utils/h5md/output_route_helpers.hpp"

using namespace SpongeH5Test;
using namespace SpongeH5OutputRoute;

static void Test_Output_Name_Sanitization()
{
    REQUIRE_EQ(Sanitize_Output_Name("TEMP(K)"), std::string("TEMP_K_"));
    REQUIRE_EQ(Sanitize_Output_Name("1TEMP"), std::string("_1TEMP"));
    REQUIRE_EQ(Sanitize_Output_Name(""), std::string("_"));
    REQUIRE_EQ(Sanitize_Output_Name("A B/C"), std::string("A_B_C"));
    REQUIRE_EQ(Sanitize_Output_Name("already_ok_9"),
               std::string("already_ok_9"));
}

static void Test_Unique_Output_Name_Collision_Resolution()
{
    const auto names =
        Make_Unique_Output_Names({"A-B", "A B", "A_B", "A_B", "1", ""});

    REQUIRE_EQ(names.size(), static_cast<std::size_t>(6));
    REQUIRE_EQ(names[0], std::string("A_B"));
    REQUIRE_EQ(names[1], std::string("A_B_1"));
    REQUIRE_EQ(names[2], std::string("A_B_2"));
    REQUIRE_EQ(names[3], std::string("A_B_3"));
    REQUIRE_EQ(names[4], std::string("_1"));
    REQUIRE_EQ(names[5], std::string("_"));

    const auto names_with_reserved_suffixes =
        Make_Unique_Output_Names({"A", "A_1", "A", "A-1", "A"});
    REQUIRE_EQ(names_with_reserved_suffixes.size(),
               static_cast<std::size_t>(5));
    REQUIRE_EQ(names_with_reserved_suffixes[0], std::string("A"));
    REQUIRE_EQ(names_with_reserved_suffixes[1], std::string("A_1"));
    REQUIRE_EQ(names_with_reserved_suffixes[2], std::string("A_2"));
    REQUIRE_EQ(names_with_reserved_suffixes[3], std::string("A_1_1"));
    REQUIRE_EQ(names_with_reserved_suffixes[4], std::string("A_3"));
}

static void Test_Reserved_H5MD_Output_Name_Remap()
{
    const auto names = Make_Unique_Output_Names(
        {"step", "time", "mdout_step", "mdout_time", "step"});
    REQUIRE_EQ(names.size(), static_cast<std::size_t>(5));
    REQUIRE_EQ(names[0], std::string("mdout_step"));
    REQUIRE_EQ(names[1], std::string("mdout_time"));
    REQUIRE_EQ(names[2], std::string("mdout_step_1"));
    REQUIRE_EQ(names[3], std::string("mdout_time_1"));
    REQUIRE_EQ(names[4], std::string("mdout_step_2"));
}

static void Test_Output_Double_Parsing()
{
    double value = 0.0;
    REQUIRE_TRUE(Parse_Output_Double(" 1.25 ", &value));
    REQUIRE_EQ(value, 1.25);
    REQUIRE_TRUE(Parse_Output_Double("-3.0e+2", &value));
    REQUIRE_EQ(value, -300.0);

    value = 7.0;
    REQUIRE_TRUE(!Parse_Output_Double("", &value));
    REQUIRE_EQ(value, 7.0);
    REQUIRE_TRUE(!Parse_Output_Double("   ", &value));
    REQUIRE_EQ(value, 7.0);
    REQUIRE_TRUE(!Parse_Output_Double("abc", &value));
    REQUIRE_EQ(value, 7.0);
    REQUIRE_TRUE(!Parse_Output_Double("1.0x", &value));
    REQUIRE_EQ(value, 7.0);
    REQUIRE_TRUE(!Parse_Output_Double("1.0", nullptr));
}

static void Test_Reaxff_Output_Key_Recognition()
{
    REQUIRE_TRUE(Is_Reaxff_Output_Key("REAXFF"));
    REQUIRE_TRUE(Is_Reaxff_Output_Key("REAXFF_BOND"));
    REQUIRE_TRUE(Is_Reaxff_Output_Key("REAXFF_"));
    REQUIRE_TRUE(!Is_Reaxff_Output_Key("REAXFFBOND"));
    REQUIRE_TRUE(!Is_Reaxff_Output_Key("X_REAXFF"));
    REQUIRE_TRUE(!Is_Reaxff_Output_Key("REAX"));
    REQUIRE_TRUE(!Is_Reaxff_Output_Key("reaxff"));
}

static void Test_Output_Key_Exists()
{
    const std::vector<std::string> keys = {"TEMP", "LJ", "REAXFF_BOND"};
    REQUIRE_TRUE(Output_Key_Exists(keys, "TEMP"));
    REQUIRE_TRUE(Output_Key_Exists(keys, "LJ"));
    REQUIRE_TRUE(!Output_Key_Exists(keys, "PRESS"));
    REQUIRE_TRUE(!Output_Key_Exists(keys, nullptr));
    REQUIRE_TRUE(!Output_Key_Exists({}, "TEMP"));
}

static void Test_Text_File_Read_If_Present()
{
    const auto dir = SpongeH5Test::Unique_Temp_Path("output_route_helpers");
    std::filesystem::create_directories(dir);
    const auto file_path = dir / "sidecar.txt";
    {
        std::ofstream output(file_path, std::ios::out | std::ios::binary);
        output << "line one\nline two\n";
    }

    std::string text;
    REQUIRE_TRUE(Read_Text_File_If_Present(file_path.string().c_str(), &text));
    REQUIRE_EQ(text, std::string("line one\nline two\n"));

    text = "unchanged";
    REQUIRE_TRUE(!Read_Text_File_If_Present(nullptr, &text));
    REQUIRE_EQ(text, std::string("unchanged"));
    REQUIRE_TRUE(!Read_Text_File_If_Present("", &text));
    REQUIRE_EQ(text, std::string("unchanged"));
    REQUIRE_TRUE(!Read_Text_File_If_Present(
        (dir / "missing.txt").string().c_str(), &text));
    REQUIRE_EQ(text, std::string("unchanged"));
    REQUIRE_TRUE(
        !Read_Text_File_If_Present(file_path.string().c_str(), nullptr));

    std::filesystem::remove_all(dir);
}

int main()
{
    return Run_Test(
        []
        {
            Test_Output_Name_Sanitization();
            Test_Unique_Output_Name_Collision_Resolution();
            Test_Reserved_H5MD_Output_Name_Remap();
            Test_Output_Double_Parsing();
            Test_Reaxff_Output_Key_Recognition();
            Test_Output_Key_Exists();
            Test_Text_File_Read_If_Present();
        });
}
