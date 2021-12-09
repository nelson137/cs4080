#include <cstring>
#include <memory>
#include <vector>

using namespace std;

void golden_selection_sort(unsigned char arr[], int length)
{
    for (int i=0; i<length; i++)
    {
        unsigned char min_val = arr[i];
        int min_idx = i;

        for (int j=i+1; j<length; j++)
        {
            unsigned char val_j = arr[j];
            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        if (min_idx != i)
        {
            arr[min_idx] = arr[i];
            arr[i] = min_val;
        }
    }
}

unique_ptr<unsigned char[]> gold_standard(
    unsigned char *img_in,
    unsigned width,
    unsigned height,
    unsigned radius)
{
    size_t size = sizeof(unsigned char) * width * height;
    size_t n_side = 2*radius + 1;
    size_t n_area = n_side * n_side;

    unique_ptr<unsigned char[]> img_out(new unsigned char[size]);
    memset(img_out.get(), 0xff, size);

    vector<unsigned char> neighborhood(n_area);

    for (size_t y = radius; y < height - radius; y++)
    {
        for (size_t x = radius; x < width - radius; x++)
        {
            size_t i = y*width + x;
            // Index of the first row in the neighborhood (top left pixel)
            size_t n_i     = i - radius - radius*width;
            // Index of the last row in the neighborhood (bot left pixel)
            size_t n_i_end = i - radius + radius*width;
            // Pointer to row in neighborhood
            unsigned char *row = &neighborhood[0];
            // Copy pixels of surrounding radius into neighborhood array
            while (n_i <= n_i_end)
            {
                memcpy(row, img_in + n_i, n_side);
                row += n_side;
                n_i += width;
            }

            golden_selection_sort(&neighborhood[0], n_area);
            *(img_out.get() + i) = neighborhood[radius*n_side + radius];
        }
    }

    return img_out;
}
