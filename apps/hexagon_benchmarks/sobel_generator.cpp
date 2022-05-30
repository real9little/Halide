#include "Halide.h"

using namespace Halide;

Expr sum3x3(Func f, Var x, Var y) {
    return f(x - 1, y - 1) + f(x - 1, y) + f(x - 1, y + 1) +
           f(x, y - 1) + f(x, y) + f(x, y + 1) +
           f(x + 1, y - 1) + f(x + 1, y) + f(x + 1, y + 1);
}

class Sobel : public Generator<Sobel> {
public:
    Input<Buffer<uint8_t, 2>> input{"input"};
    Output<Buffer<float, 2>> output{"output"};

    GeneratorParam<bool> use_parallel_sched{"use_parallel_sched", true};
    GeneratorParam<bool> use_prefetch_sched{"use_prefetch_sched", true};

    void generate() {
        bounded_input(x, y) = BoundaryConditions::repeat_edge(input)(x, y);

        Iy(x, y) = bounded_input(x - 1, y - 1) * (-1.0f / 12) + bounded_input(x - 1, y + 1) * (1.0f / 12) +
                   bounded_input(x, y - 1) * (-2.0f / 12) + bounded_input(x, y + 1) * (2.0f / 12) +
                   bounded_input(x + 1, y - 1) * (-1.0f / 12) + bounded_input(x + 1, y + 1) * (1.0f / 12);
        Ix(x, y) = bounded_input(x - 1, y - 1) * (-1.0f / 12) + bounded_input(x + 1, y - 1) * (1.0f / 12) +
                   bounded_input(x - 1, y) * (-2.0f / 12) + bounded_input(x + 1, y) * (2.0f / 12) +
                   bounded_input(x - 1, y + 1) * (-1.0f / 12) + bounded_input(x + 1, y + 1) * (1.0f / 12);
        Ixx(x, y) = Ix(x, y) * Ix(x, y);
        Iyy(x, y) = Iy(x, y) * Iy(x, y);
        Ixy(x, y) = Ix(x, y) * Iy(x, y);
        Sxx(x, y) = sum3x3(Ixx, x, y);
        Syy(x, y) = sum3x3(Iyy, x, y);
        Sxy(x, y) = sum3x3(Ixy, x, y);
        det(x, y) = Sxx(x, y) * Syy(x, y) - Sxy(x, y) * Sxy(x, y);
        trace(x, y) = Sxx(x, y) + Syy(x, y);
        output(x, y) = det(x, y) - 0.04f * trace(x, y) * trace(x, y);

//        Func input_16{"input_16"};
//        input_16(x, y) = cast<uint16_t>(bounded_input(x, y));
//
//        sobel_x_avg(x, y) = input_16(x - 1, y) + 2 * input_16(x, y) + input_16(x + 1, y);
//        sobel_x(x, y) = absd(sobel_x_avg(x, y - 1), sobel_x_avg(x, y + 1));
//
//        sobel_y_avg(x, y) = input_16(x, y - 1) + 2 * input_16(x, y) + input_16(x, y + 1);
//        sobel_y(x, y) = absd(sobel_y_avg(x - 1, y), sobel_y_avg(x + 1, y));
//
//        // This sobel implementation is non-standard in that it doesn't take the square root
//        // of the gradient.
//        output(x, y) = cast<uint8_t>(clamp(sobel_x(x, y) + sobel_y(x, y), 0, 255));
    }

    void schedule() {
        Var xi{"xi"}, yi{"yi"};

        input.dim(0).set_min(0);
        input.dim(1).set_min(0);

        if (get_target().has_feature(Target::HVX)) {
            const int vector_size = 128;
            Expr input_stride = input.dim(1).stride();
            input.dim(1).set_stride((input_stride / vector_size) * vector_size);

            Expr output_stride = output.dim(1).stride();
            output.dim(1).set_stride((output_stride / vector_size) * vector_size);
            bounded_input
                .compute_at(Func(output), y)
                .align_storage(x, 128)
                .vectorize(x, vector_size, TailStrategy::RoundUp);
            output
                .hexagon()
                .tile(x, y, xi, yi, vector_size, 4, TailStrategy::RoundUp)
                .vectorize(xi)
                .unroll(yi);
            if (use_prefetch_sched) {
                output.prefetch(input, y, y, 2);
            }
            if (use_parallel_sched) {
                Var yo;
                output.split(y, yo, y, 128).parallel(yo);
            }
        } else {
            const int vector_size = natural_vector_size<uint8_t>();
            output
                .vectorize(x, vector_size)
                .parallel(y, 16);
        }
    }

private:
    Var x{"x"}, y{"y"};
    Func Iy{"Iy"}, Ix{"Ix"}, Ixx{"Ixx"}, Iyy{"Iyy"}, Ixy{"Ixy"};
    Func Sxx{"Sxx"};
    Func Syy{"Syy"};
    Func Sxy{"Sxy"};
    Func det{"det"};
    Func trace{"trace"};
//    Func sobel_x_avg{"sobel_x_avg"}, sobel_y_avg{"sobel_y_avg"};
//    Func sobel_x{"sobel_x"}, sobel_y{"sobel_y"};
    Func bounded_input{"bounded_input"};
};

HALIDE_REGISTER_GENERATOR(Sobel, sobel)
