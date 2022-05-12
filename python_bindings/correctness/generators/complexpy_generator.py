import halide as hl

x = hl.Var('x')
y = hl.Var('y')
c = hl.Var('c')

@hl.generator(name = "complexpy")
class ComplexPy:
    vectorize = hl.GeneratorParam(True)

    typed_buffer_input = hl.InputBuffer(hl.UInt(8), 3)
    untyped_buffer_input = hl.InputBuffer(None, 3)
    simple_input = hl.InputBuffer(None, 3)
    float_arg = hl.InputScalar(hl.Float(32))
    int_arg = hl.InputScalar(hl.Int(32))

    simple_output = hl.OutputBuffer(hl.Float(32), 3)
    tuple_output = hl.OutputBuffer(None, 3)
    typed_buffer_output = hl.OutputBuffer(hl.Float(32), 3)
    untyped_buffer_output = hl.OutputBuffer(None, None)
    static_compiled_buffer_output = hl.OutputBuffer(hl.UInt(8), 3)
    scalar_output = hl.OutputScalar(hl.Float(32))

    # Just an intermediate Func we need to share between generate() and schedule()
    intermediate = hl.Func()

    def configure(self):
        g = self
        g.add_input("extra_input", hl.InputBuffer(hl.UInt(16), 3))
        g.add_output("extra_output", hl.OutputBuffer(hl.Float(64), 2))

    def generate(self):
        g = self

        g.simple_output[x, y, c] = hl.f32(g.simple_input[x, y, c])
        g.typed_buffer_output[x, y, c] = hl.f32(g.typed_buffer_input[x, y, c])
        g.untyped_buffer_output[x, y, c] = hl.cast(g.untyped_buffer_output.output_type(), g.untyped_buffer_input[x, y, c])

        g.intermediate[x, y, c] = g.simple_input[x, y, c] * g.float_arg

        g.tuple_output[x, y, c] = (g.intermediate[x, y, c], g.intermediate[x, y, c] + g.int_arg)

        # This should be compiled into the Generator product itself,
        # and not produce another input for the Stub or AOT filter.
        static_compiled_buffer = hl.Buffer(hl.UInt(8), [4, 4, 1])
        for xx in range(4):
            for yy in range(4):
                for cc in range(1):
                    static_compiled_buffer[xx, yy, cc] = xx + yy + cc + 42

        g.static_compiled_buffer_output[x, y, c] = static_compiled_buffer[x, y, c]
        g.extra_output[x, y] = hl.f64(g.extra_input[x, y, 0] + 1)

        g.scalar_output[()] = g.float_arg + g.int_arg

        g.intermediate.compute_at(g.tuple_output, y);
        g.intermediate.specialize(g.vectorize).vectorize(x, g.natural_vector_size(hl.Float(32)));

if __name__ == "__main__":
    hl.main()

