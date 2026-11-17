// Directly exercise the production local-pair packing kernel on the CPU.
// This is an identity/parameter transport test, not an MPI communication test.
#include <stdexcept>

#include "../../SPONGE/nb14/nb14.cpp"

static void Require(bool condition)
{
    if (!condition) throw std::runtime_error("nb14 local remap mismatch");
}

int main()
{
    int a[] = {0, 1, 1, 2, 3, 0};
    int b[] = {1, 2, 3, 3, 4, 2};
    float A[] = {12, 24, 36, 48, 60, 72};
    float B[] = {6, 12, 18, 24, 30, 36};
    float q[] = {.1f, .2f, .3f, .4f, .5f, .6f};
    float delta[] = {0, 1, 2, 3, 4, 5};
    int aa[6], bb[6], count;
    float AA[6], BB[6], qq[6], dd[6];
    char local[] = {1, 0, 1, 0, 0};
    int ids[] = {1, 3, 0, 2, 4};
    const int kept[] = {0, 1, 3, 5};
    for (int iteration = 0; iteration < 3; ++iteration)
    {
        for (int i = 0; i < 6; ++i) dd[i] = -999;
        get_local_device(6, a, b, local, ids, aa, bb, A, B, q, AA, BB, qq,
                         iteration == 1 ? nullptr : delta, dd, &count);
        Require(count == 4);
        for (int k = 0; k < count; ++k)
        {
            int i = kept[k];
            Require(aa[k] == ids[a[i]] && bb[k] == ids[b[i]]);
            Require(AA[k] == A[i] && BB[k] == B[i] && qq[k] == q[i]);
            Require(dd[k] == (iteration == 1 ? -999 : delta[i]));
        }
        // Change local ordering and parameters to detect stale buffers.
        std::swap(ids[0], ids[2]);
        for (int i = 0; i < 6; ++i) delta[i] += 7;
    }
    // A domain owning no atom must clear its previous pair count.
    for (int i = 0; i < 5; ++i) local[i] = 0;
    get_local_device(6, a, b, local, ids, aa, bb, A, B, q, AA, BB, qq, delta,
                     dd, &count);
    Require(count == 0);
    return 0;
}
