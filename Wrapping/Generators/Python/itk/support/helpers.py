# ==========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ==========================================================================*/

import os
import re
import functools

import numpy as np

_HAVE_XARRAY = False
try:
    import xarray as xr

    _HAVE_XARRAY = True
except ImportError:
    pass
_HAVE_TORCH = False
try:
    import torch

    _HAVE_TORCH = True
except ImportError:
    pass


def camel_to_snake_case(name):
    snake = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake)
    return snake.replace("__", "_").lower()


def is_arraylike(arr):
    return (
        hasattr(arr, "shape")
        and hasattr(arr, "dtype")
        and hasattr(arr, "__array__")
        and hasattr(arr, "ndim")
    )


def move_first_dimension_to_last(arr):
    import numpy as np

    dest = list(range(arr.ndim))
    source = dest.copy()
    end = source.pop()
    source.insert(0, end)
    arr_contiguous_channels = np.moveaxis(arr, source, dest).copy()
    return arr_contiguous_channels


def move_last_dimension_to_first(arr):
    import numpy as np

    dest = list(range(arr.ndim))
    source = dest.copy()
    end = source.pop()
    source.insert(0, end)
    arr_interleaved_channels = np.moveaxis(arr, dest, source).copy()
    return arr_interleaved_channels


def accept_array_like_xarray_torch(image_filter):
    """Decorator that allows itk.ProcessObject snake_case functions to accept
    NumPy array-like, PyTorch Tensor's or xarray DataArray inputs for itk.Image inputs.

    If a NumPy array-like is passed as an input, output itk.Image's are converted to numpy.ndarray's.
    If a torch.Tensor is passed as an input, output itk.Image's are converted to torch.Tensors.
    If a xarray DataArray is passed as an input, output itk.Image's are converted to xarray.DataArray's."""
    import numpy as np
    import itk

    @functools.wraps(image_filter)
    def image_filter_wrapper(*args, **kwargs):
        have_array_input = False
        have_xarray_input = False
        have_torch_input = False

        args_list = list(args)
        for index, arg in enumerate(args):
            if _HAVE_XARRAY and isinstance(arg, xr.DataArray):
                have_xarray_input = True
                image = itk.image_from_xarray(arg)
                args_list[index] = image
            elif _HAVE_TORCH and isinstance(arg, torch.Tensor):
                have_torch_input = True
                channels = arg.shape[0]  # assume first dimension is channels
                arr = np.asarray(arg)
                if channels > 1:  # change from contiguous to interleaved channel order
                    arr = move_last_dimension_to_first(arr)
                image = itk.image_view_from_array(arr, is_vector=channels > 1)
                args_list[index] = image
            elif not isinstance(arg, itk.Object) and is_arraylike(arg):
                have_array_input = True
                array = np.asarray(arg)
                image = itk.image_view_from_array(array)
                args_list[index] = image

        potential_image_input_kwargs = ("input", "input1", "input2", "input3")
        for key, value in kwargs.items():
            if key.lower() in potential_image_input_kwargs or "image" in key.lower():
                if _HAVE_XARRAY and isinstance(value, xr.DataArray):
                    have_xarray_input = True
                    image = itk.image_from_xarray(value)
                    kwargs[key] = image
                elif _HAVE_TORCH and isinstance(value, torch.Tensor):
                    have_torch_input = True
                    channels = value.shape[0]  # assume first dimension is channels
                    arr = np.asarray(value)
                    if (
                        channels > 1
                    ):  # change from contiguous to interleaved channel order
                        arr = move_last_dimension_to_first(arr)
                    image = itk.image_view_from_array(arr, is_vector=channels > 1)
                    kwargs[key] = image
                elif not isinstance(value, itk.Object) and is_arraylike(value):
                    have_array_input = True
                    array = np.asarray(value)
                    image = itk.image_view_from_array(array)
                    kwargs[key] = image

        if have_xarray_input or have_torch_input or have_array_input:
            # Convert output itk.Image's to numpy.ndarray's
            output = image_filter(*tuple(args_list), **kwargs)
            if isinstance(output, tuple):
                output_list = list(output)
                for index, value in enumerate(output_list):
                    if isinstance(value, itk.Image):
                        if have_xarray_input:
                            data_array = itk.xarray_from_image(value)
                            output_list[index] = data_array
                        elif have_torch_input:
                            channels = value.GetNumberOfComponentsPerPixel()
                            data_array = itk.array_view_from_image(value)
                            if (
                                channels > 1
                            ):  # change from interleaved to contiguous channel order
                                data_array = move_first_dimension_to_last(data_array)
                            torch_tensor = torch.from_numpy(data_array)
                            output_list[index] = torch_tensor
                        else:
                            array = itk.array_view_from_image(value)
                            output_list[index] = array
                return tuple(output_list)
            else:
                if isinstance(output, itk.Image):
                    if have_xarray_input:
                        output = itk.xarray_from_image(output)
                    elif have_torch_input:
                        channels = output.GetNumberOfComponentsPerPixel()
                        output = itk.array_view_from_image(output)
                        if (
                            channels > 1
                        ):  # change from interleaved to contiguous channel order
                            output = move_first_dimension_to_last(output)
                        output = torch.from_numpy(output)
                    else:
                        output = itk.array_view_from_image(output)
                return output
        else:
            return image_filter(*args, **kwargs)

    return image_filter_wrapper


def image_type_from_wasm_type(jstype):
    import itk

    _pixelType_to_prefix = {
        1: '',
        2: 'RGB',
        3: 'RGBA',
        4: 'O',
        5: 'V',
        7: 'CV',
        8: 'SSRT',
        11: 'FA'
    }
    pixelType = jstype['pixelType']
    dimension = jstype['dimension']
    if pixelType == 10:
        if jstype['componentType'] == 'float':
            return itk.Image[itk.complex, itk.F], np.float32
        else:
            return itk.Image[itk.complex, itk.D], np.float64

    def _long_type():
        if os.name == 'nt':
            return 'LL'
        else:
            return 'L'
    prefix = _pixelType_to_prefix[pixelType]
    _js_to_python = {
        'int8_t': 'SC',
        'uint8_t': 'UC',
        'int16_t': 'SS',
        'uint16_t': 'US',
        'int32_t': 'SI',
        'uint32_t': 'UI',
        'int64_t': 'S' + _long_type(),
        'uint64_t': 'U' + _long_type(),
        'float': 'F',
        'double': 'D'
    }
    _js_to_numpy_dtype = {
        'int8_t': np.int8,
        'uint8_t': np.uint8,
        'int16_t': np.int16,
        'uint16_t': np.uint16,
        'int32_t': np.int32,
        'uint32_t': np.uint32,
        'int64_t': np.int64,
        'uint64_t': np.uint64,
        'float': np.float32,
        'double': np.float64
    }
    dtype = _js_to_numpy_dtype[jstype['componentType']]
    if pixelType != 4:
        prefix += _js_to_python[jstype['componentType']]
    if pixelType not in (1, 2, 3, 10):
        prefix += str(dimension)
    prefix += str(dimension)
    return getattr(itk.Image, prefix), dtype
