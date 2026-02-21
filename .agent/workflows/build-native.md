---
description: How to build the CDL C++ native extension
---

# Building the CDL C++ Extension

The CDL package works in pure Python by default, but compiling the
C++ extension provides significant performance improvements.

## Prerequisites

- MSYS2 MinGW64 (at `D:\msys64\mingw64\bin`) — already installed
- `pip install cmake pybind11 setuptools`

## Build (single command)

// turbo
```cmd
set PATH=D:\msys64\mingw64\bin;%PATH%
```

// turbo
```cmd
del "d:\THESIS\CHC_C++_OP\CHC_AI\CeSIPS\CDL\src\cdl\_cdl_native.cp313-win_amd64.pyd" 2>nul && rd /s /q d:\THESIS\CHC_C++_OP\CHC_AI\CeSIPS\CDL\build 2>nul && mkdir d:\THESIS\CHC_C++_OP\CHC_AI\CeSIPS\CDL\build && cmake -S d:\THESIS\CHC_C++_OP\CHC_AI\CeSIPS\CDL -B d:\THESIS\CHC_C++_OP\CHC_AI\CeSIPS\CDL\build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCDL_BUILD_TESTS=OFF -DCDL_BUILD_PYTHON=ON -DCMAKE_CXX_COMPILER="D:/msys64/mingw64/bin/g++.exe" -Dpybind11_DIR="C:\Users\Anthony\AppData\Local\Programs\Python\Python313\Lib\site-packages\pybind11\share\cmake\pybind11" -DCMAKE_MAKE_PROGRAM="D:/msys64/mingw64/bin/mingw32-make.exe" -DPYTHON_EXECUTABLE="C:/Users/Anthony/AppData/Local/Programs/Python/Python313/python.exe" -DPYTHON_LIBRARY="C:/Users/Anthony/AppData/Local/Programs/Python/Python313/libs/python313.lib" -DPYTHON_INCLUDE_DIR="C:/Users/Anthony/AppData/Local/Programs/Python/Python313/include" -Wno-dev && cmake --build d:\THESIS\CHC_C++_OP\CHC_AI\CeSIPS\CDL\build --config Release -j4
```

## Verify

// turbo
```cmd
set PYTHONPATH=d:\THESIS\CHC_C++_OP\CHC_AI\CeSIPS\CDL\src && "C:\Users\Anthony\AppData\Local\Programs\Python\Python313\python.exe" -c "from cdl.shuffle import USING_NATIVE; print('Native backend:', USING_NATIVE)"
```

Expected output: `Native backend: True`

## Run Tests

// turbo
```cmd
set PYTHONPATH=d:\THESIS\CHC_C++_OP\CHC_AI\CeSIPS\CDL\src && "C:\Users\Anthony\AppData\Local\Programs\Python\Python313\python.exe" -m pytest d:\THESIS\CHC_C++_OP\CHC_AI\CeSIPS\CDL\tests -v --tb=short
```

## Key Notes

- **MSYS2 bin must be on PATH** before running cmake (for compiler detection)
- **Static linking** is enabled (`-static-libgcc -static-libstdc++`) so the `.pyd` doesn't need MSYS2 DLLs at runtime
- **Python paths must be explicit** to avoid picking up MSYS2's own Python 3.12
- The `.pyd` is output to `src/cdl/` so `import cdl._cdl_native` works
