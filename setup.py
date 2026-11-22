"""Setuptools native C++20 shared-library build for CausalSieve wheels.

Compiles the multi-ISA C++20 core directly into a platform-native ctypes
shared library inside the installed ``causalsieve`` package:

  - Linux:   ``causalsieve/libcausalsieve.so``
  - macOS:   ``causalsieve/libcausalsieve.dylib``
  - Windows: ``causalsieve/causalsieve.dll``

Architecture & SIMD multi-ISA runtime dispatch:
  - x86_64 / AMD64 (Linux, macOS x86_64, Windows MSVC — INCLUDING portable
    ``CAUSALSIEVE_PORTABLE=1`` distribution wheels!):
      * ``csrc/causalsieve.cpp`` + ``csrc/causalsieve_kernels_scalar.cpp`` are
        compiled for portable baseline x86-64 (zero AVX2 instructions).
      * ``csrc/causalsieve_kernels_avx2.cpp`` is compiled with ``-mavx2 -mfma``
        (or ``/arch:AVX2 /D__FMA__=1`` on MSVC) and ``CAUSALSIEVE_BUILD_AVX2=1``.
      * All translation units link into ONE shared library that selects
        ``avx2+fma`` vs ``scalar`` once at startup via CPUID + OSXSAVE + XGETBV.
  - arm64 / aarch64: ``-O3`` (NEON is baseline on AArch64; no x86 SIMD flags).
  - macOS universal2: ``-O3`` with ``ARCHFLAGS`` and no x86-only flags.

Because ``libcausalsieve`` exposes a pure C99 ABI loaded via ``ctypes`` with
zero Python C-API linkage, built wheels are tagged ``py3-none-<platform>`` so
one wheel per OS/architecture works across Python 3.10+.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

ROOT = Path(__file__).resolve().parent
CSRC_DIR = ROOT / "csrc"
SOURCE_CORE_CPP = CSRC_DIR / "causalsieve.cpp"
SOURCE_SCALAR_CPP = CSRC_DIR / "causalsieve_kernels_scalar.cpp"
SOURCE_AVX2_CPP = CSRC_DIR / "causalsieve_kernels_avx2.cpp"

# Exact C99 ABI exports from csrc/causalsieve.h (used to generate .def on MSVC).
_C99_EXPORTS: tuple[str, ...] = (
    "cs_config_default",
    "cs_create",
    "cs_destroy",
    "cs_reset",
    "cs_simd_isa",
    "cs_last_error",
    "cs_tick_count",
    "cs_dim",
    "cs_lag",
    "cs_feature_dim",
    "cs_push",
    "cs_push_batch",
    "cs_granger_matrix",
    "cs_precision_matrix",
    "cs_contemporaneous_orientation",
    "cs_root_attribution",
    "cs_counterfactual_do",
    "cs_get_raw_slice",
    "cs_compress_channel_pla",
    "cs_inspect_state",
    "cs_force_refresh",
    "cs_screened_create",
    "cs_screened_destroy",
    "cs_screened_push_batch",
    "cs_screened_active_graph",
)


def _target_lib_name() -> str:
    if sys.platform == "darwin":
        return "libcausalsieve.dylib"
    if os.name == "nt" or sys.platform == "win32":
        return "causalsieve.dll"
    return "libcausalsieve.so"


def _is_portable_build() -> bool:
    return os.environ.get("CAUSALSIEVE_PORTABLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _detect_target_arch(plat_name: str | None = None) -> str:
    """Detect target CPU architecture, respecting macOS ARCHFLAGS and plat_name."""
    if sys.platform == "darwin":
        archflags = os.environ.get("ARCHFLAGS", "")
        tokens = shlex.split(archflags)
        arches = [
            tokens[i + 1].lower()
            for i in range(len(tokens) - 1)
            if tokens[i] == "-arch"
        ]
        if len(arches) == 1:
            raw = arches[0]
        elif len(arches) > 1:
            return "universal2"
        else:
            raw = (plat_name or platform.machine()).lower()
    else:
        raw = (plat_name or platform.machine()).lower()

    if "universal" in raw:
        return "universal2"
    if any(tok in raw for tok in ("arm64", "aarch64")):
        return "arm64"
    if any(tok in raw for tok in ("x86_64", "amd64", "x64")):
        return "x86_64"
    return platform.machine().lower()


def _macos_arch_flags() -> list[str]:
    if sys.platform != "darwin":
        return []
    archflags = os.environ.get("ARCHFLAGS", "").strip()
    if not archflags:
        return []
    tokens = shlex.split(archflags)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "-arch" and i + 1 < len(tokens):
            out.extend(["-arch", tokens[i + 1]])
            i += 2
        else:
            i += 1
    return out


def _find_unix_cxx() -> list[str]:
    env_cxx = os.environ.get("CXX", "").strip()
    if env_cxx:
        return shlex.split(env_cxx)
    for candidate in ("c++", "g++", "clang++"):
        found = shutil.which(candidate)
        if found:
            return [found]
    raise RuntimeError(
        "No C++20 compiler found (tried $CXX, c++, g++, clang++). "
        "Install g++ or clang++ to build causalsieve from source."
    )


class BuildNativeLib(build_ext):
    """Compile CausalSieve multi-ISA C++20 core directly into ``causalsieve/<lib>``."""

    def initialize_options(self) -> None:
        super().initialize_options()
        self._built_outputs: list[str] = []
        self._output_mapping: dict[str, str] = {}

    def run(self) -> None:
        for src in (SOURCE_CORE_CPP, SOURCE_SCALAR_CPP, SOURCE_AVX2_CPP):
            if not src.is_file():
                raise FileNotFoundError(f"Missing C++20 core source file: {src}")

        lib_name = _target_lib_name()
        build_lib = Path(self.build_lib).resolve()
        pkg_out_dir = build_lib / "causalsieve"
        pkg_out_dir.mkdir(parents=True, exist_ok=True)
        dest_path = pkg_out_dir / lib_name
        inplace_path = ROOT / "python" / "causalsieve" / lib_name

        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        plat_name = getattr(self, "plat_name", None)
        arch = _detect_target_arch(plat_name)
        portable = _is_portable_build()

        compiler_type = getattr(self.compiler, "compiler_type", None)
        if os.name == "nt" and compiler_type in (None, "msvc"):
            self._compile_msvc(dest_path, build_temp, arch=arch, portable=portable)
        else:
            self._compile_unix(dest_path, build_temp, arch=arch, portable=portable)

        self._built_outputs = [str(dest_path)]
        self._output_mapping = {str(dest_path): str(inplace_path)}

        if self.inplace:
            inplace_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest_path, inplace_path)

    def _compile_unix(
        self,
        dest_path: Path,
        build_temp: Path,
        *,
        arch: str,
        portable: bool,
    ) -> None:
        cxx = _find_unix_cxx()
        base_flags: list[str] = [
            "-std=c++20",
            "-O3",
            "-fno-math-errno",
            "-fno-trapping-math",
            "-fPIC",
            f"-I{CSRC_DIR}",
        ]
        if sys.platform == "darwin":
            base_flags.extend(_macos_arch_flags())

        # Enable runtime CPUID AVX2+FMA dispatch on all x86_64 builds (including portable wheels).
        use_avx2_dispatch = arch == "x86_64"

        core_obj = build_temp / "causalsieve.o"
        scalar_obj = build_temp / "causalsieve_kernels_scalar.o"
        avx2_obj = build_temp / "causalsieve_kernels_avx2.o"

        core_flags = list(base_flags)
        if use_avx2_dispatch:
            core_flags.append("-DCAUSALSIEVE_HAS_AVX2_TU=1")

        objects: list[str] = []

        # 1. Core engine TU (portable baseline instructions)
        cmd_core = [*cxx, *core_flags, "-c", str(SOURCE_CORE_CPP), "-o", str(core_obj)]
        self.announce(
            f"Compiling CausalSieve core TU ({arch}, portable={portable}, avx2_dispatch={use_avx2_dispatch}): {' '.join(cmd_core)}",
            level=2,
        )
        subprocess.run(cmd_core, check=True)
        objects.append(str(core_obj))

        # 2. Scalar / ARM64 NEON kernel TU (portable baseline instructions)
        cmd_scalar = [*cxx, *base_flags, "-c", str(SOURCE_SCALAR_CPP), "-o", str(scalar_obj)]
        self.announce(f"Compiling CausalSieve scalar/NEON kernel TU: {' '.join(cmd_scalar)}", level=2)
        subprocess.run(cmd_scalar, check=True)
        objects.append(str(scalar_obj))

        # 3. AVX2+FMA kernel TU (x86_64 only; safely guarded at runtime via CPUID+OSXSAVE+XGETBV)
        if use_avx2_dispatch:
            avx2_flags = [*base_flags, "-mavx2", "-mfma", "-DCAUSALSIEVE_BUILD_AVX2=1"]
            cmd_avx2 = [*cxx, *avx2_flags, "-c", str(SOURCE_AVX2_CPP), "-o", str(avx2_obj)]
            self.announce(f"Compiling CausalSieve AVX2+FMA kernel TU: {' '.join(cmd_avx2)}", level=2)
            subprocess.run(cmd_avx2, check=True)
            objects.append(str(avx2_obj))

        if sys.platform == "darwin":
            link_flags = [
                *_macos_arch_flags(),
                "-dynamiclib",
                "-install_name",
                "@rpath/libcausalsieve.dylib",
                "-lm",
            ]
        else:
            link_flags = ["-shared", "-lm"]

        cmd_link = [*cxx, *objects, "-o", str(dest_path), *link_flags]
        self.announce(f"Linking CausalSieve native library: {' '.join(cmd_link)}", level=2)
        subprocess.run(cmd_link, check=True)

    def _compile_msvc(
        self,
        dest_path: Path,
        build_temp: Path,
        *,
        arch: str,
        portable: bool,
    ) -> None:
        from setuptools._distutils.ccompiler import new_compiler

        compiler = self.compiler or new_compiler(compiler="msvc")
        if not getattr(compiler, "initialized", False):
            compiler.initialize()

        use_avx2_dispatch = arch == "x86_64"
        base_args: list[str] = [
            "/std:c++20",
            "/O2",
            "/D__restrict__=__restrict",
        ]
        core_args = list(base_args)
        if use_avx2_dispatch:
            core_args.append("/DCAUSALSIEVE_HAS_AVX2_TU=1")

        self.announce(
            f"Compiling CausalSieve native core (MSVC {arch}, portable={portable}, avx2_dispatch={use_avx2_dispatch})",
            level=2,
        )

        objects: list[str] = []
        objects.extend(
            compiler.compile(
                [str(SOURCE_CORE_CPP)],
                output_dir=str(build_temp),
                include_dirs=[str(CSRC_DIR)],
                extra_postargs=core_args,
            )
        )
        objects.extend(
            compiler.compile(
                [str(SOURCE_SCALAR_CPP)],
                output_dir=str(build_temp),
                include_dirs=[str(CSRC_DIR)],
                extra_postargs=base_args,
            )
        )
        if use_avx2_dispatch:
            avx2_args = [*base_args, "/arch:AVX2", "/D__FMA__=1", "/DCAUSALSIEVE_BUILD_AVX2=1"]
            objects.extend(
                compiler.compile(
                    [str(SOURCE_AVX2_CPP)],
                    output_dir=str(build_temp),
                    include_dirs=[str(CSRC_DIR)],
                    extra_postargs=avx2_args,
                )
            )

        compiler.link_shared_object(
            objects,
            str(dest_path),
            export_symbols=list(_C99_EXPORTS),
            build_temp=str(build_temp),
            target_lang="c++",
        )

    def get_outputs(self) -> list[str]:
        return list(self._built_outputs)

    def get_output_mapping(self) -> dict[str, str]:
        return dict(self._output_mapping)

    def copy_extensions_to_source(self) -> None:
        for built, inplace in self._output_mapping.items():
            built_p = Path(built)
            if built_p.is_file():
                inplace_p = Path(inplace)
                inplace_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built_p, inplace_p)


class BdistWheel(_bdist_wheel):
    """Mark wheel as platform-specific and Python-ABI-independent (``py3-none-<plat>``)."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, plat = super().get_tag()
        return ("py3", "none", plat)


setup(
    ext_modules=[
        Extension(
            name="causalsieve._libcausalsieve",
            sources=[
                "csrc/causalsieve.cpp",
                "csrc/causalsieve_kernels_scalar.cpp",
                "csrc/causalsieve_kernels_avx2.cpp",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildNativeLib,
        "bdist_wheel": BdistWheel,
    },
)
