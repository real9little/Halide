include(PythonExtensionHelpers)

set(SCRIPTS
    bilateral_grid_shell.py
    blur.py
    erode.py
    interpolate.py
    local_laplacian.py)

add_python_aot_extension(py_aot_bilateral_grid
                         GENERATOR bilateral_grid
                         FUNCTION_NAME bilateral_grid
                         PARAMS s_sigma=8 auto_schedule=false
                         SOURCES bilateral_grid_generator.py)

# TODO: can't use autoscheduling from python until the glue code
# is set up to enable the autoscheduler extensions at build time
# add_python_aot_extension(py_aot_bilateral_grid_autoschedule
#                          GENERATOR bilateral_grid
#                          FUNCTION_NAME bilateral_grid_autoschedule
#                          PARAMS s_sigma=8 auto_schedule=true
#                          SOURCES bilateral_grid_generator.py)

set(PY_APPS_DEPS_bilateral_grid_shell py_aot_bilateral_grid)
set(PY_APPS_ARGS_bilateral_grid_shell ${CMAKE_CURRENT_SOURCE_DIR}/../../apps/images/gray.png /tmp/out.png 0.1 10)

make_shell_path(PY_APPS_PYTHONPATH
                "$<TARGET_FILE_DIR:py_aot_bilateral_grid>"
                "$<TARGET_FILE_DIR:Halide::Python>"
                )

foreach (SCRIPT IN LISTS SCRIPTS)
    get_filename_component(BASE ${SCRIPT} NAME_WE)
    add_test(NAME python_apps_${BASE}
             COMMAND Python3::Interpreter "$<SHELL_PATH:${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT}>" ${PY_APPS_ARGS_${BASE}})
    set_tests_properties(python_apps_${BASE} PROPERTIES
                         LABELS python
                         DEPENDS "${PY_APPS_DEPS_${BASE}}"
                         ENVIRONMENT "PYTHONPATH=${PY_APPS_PYTHONPATH};HL_TARGET=${Halide_TARGET}")
endforeach ()
