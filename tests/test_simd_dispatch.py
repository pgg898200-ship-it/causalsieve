"""Runtime multi-ISA CPUID dispatch, CAUSALSIEVE_FORCE_SCALAR parity, and portable wheel tests."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import numpy as np
from _cabi import DEFAULT_LIB_PATH, REPO_ROOT, RawEngine


def _host_has_avx2_fma() -> bool:
    """Detect whether the current x86_64 host CPU advertises AVX2 + FMA."""
    mach = platform.machine().lower()
    if mach not in ("x86_64", "amd64", "x64"):
        return False
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        text = cpuinfo.read_text(encoding="utf-8", errors="ignore").lower()
        for line in text.splitlines():
            if line.startswith("flags"):
                tokens = set(line.split())
                return "avx2" in tokens and "fma" in tokens
    return True


def _expected_native_isa() -> str:
    mach = platform.machine().lower()
    if mach in ("arm64", "aarch64"):
        return "neon"
    if _host_has_avx2_fma():
        return "avx2+fma"
    return "scalar"


def _run_lib_isa_probe(lib_path: Path, env_overrides: dict[str, str | None]) -> str:
    code = (
        "import ctypes, sys; "
        "lib = ctypes.CDLL(sys.argv[1]); "
        "lib.cs_simd_isa.restype = ctypes.c_char_p; "
        "print(lib.cs_simd_isa().decode('ascii'))"
    )
    env = os.environ.copy()
    for key, val in env_overrides.items():
        if val is None:
            env.pop(key, None)
        else:
            env[key] = val
    proc = subprocess.run(
        [sys.executable, "-c", code, str(lib_path)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def test_default_shared_lib_reports_native_isa() -> None:
    """Default build/libcausalsieve shared library dispatches native SIMD ISA at runtime."""
    assert DEFAULT_LIB_PATH.is_file(), f"Missing native shared library: {DEFAULT_LIB_PATH}"
    isa = _run_lib_isa_probe(DEFAULT_LIB_PATH, {"CAUSALSIEVE_FORCE_SCALAR": None})
    expected = _expected_native_isa()
    assert isa == expected, f"Expected native ISA {expected!r}, got {isa!r}"


def test_runtime_force_scalar_override_same_shared_lib() -> None:
    """CAUSALSIEVE_FORCE_SCALAR=1/true/yes/on forces scalar dispatch from the SAME shared lib."""
    expected_native = _expected_native_isa()
    for truthy in ("1", "true", "TRUE", "yes", "on", " 1 "):
        isa = _run_lib_isa_probe(DEFAULT_LIB_PATH, {"CAUSALSIEVE_FORCE_SCALAR": truthy})
        assert isa == "scalar", f"CAUSALSIEVE_FORCE_SCALAR={truthy!r} gave {isa!r}, expected 'scalar'"

    for falsy in ("0", "false", "", "no"):
        isa = _run_lib_isa_probe(DEFAULT_LIB_PATH, {"CAUSALSIEVE_FORCE_SCALAR": falsy})
        assert isa == expected_native, (
            f"CAUSALSIEVE_FORCE_SCALAR={falsy!r} gave {isa!r}, expected {expected_native!r}"
        )


def test_runtime_avx2_vs_force_scalar_exact_parity(tmp_path: Path) -> None:
    """Verify < 1e-13 parity between native SIMD and CAUSALSIEVE_FORCE_SCALAR=1 on the SAME .so."""
    worker = r"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, sys.argv[1])
from _cabi import DEFAULT_LIB_PATH, RawEngine, RawScreenedEngine, load_lib

lib = load_lib(DEFAULT_LIB_PATH)
isa = lib.cs_simd_isa().decode("ascii")

# Stream A: non-multiple-of-4 D=17, p=2, T=350 (multiple slab flushes + partial tail + fault + LiNGAM)
rng = np.random.default_rng(202604)
D_a, p_a, T_a = 17, 2, 350
X_a = np.zeros((T_a, D_a), dtype=np.float64)
for t in range(T_a):
    e = rng.exponential(1.0, size=D_a) - 1.0
    row = e.copy()
    row[1] += 0.65 * row[0]
    row[2] += 0.60 * row[1]
    if t >= 1:
        row[3] += 0.55 * X_a[t - 1, 0]
        row[4] += 0.45 * X_a[t - 1, 3]
        row += 0.25 * X_a[t - 1]
    if t >= 2:
        row[5] -= 0.20 * X_a[t - 2, 1]
    if t >= 290:
        row[0] += 7.5
    X_a[t] = row
X_a = np.ascontiguousarray(X_a, dtype=np.float64)

with RawEngine(dim=D_a, lag=p_a, forgetting=0.99, warmup_ticks=50, normalize_scale=1, lib_path=DEFAULT_LIB_PATH) as eng:
    rc, res_a, exo_a, innov_a = eng.push_batch(X_a, return_results=True, return_matrices=True)
    assert rc == 0
    B_a, freiv_a = eng.inspect_state()
    F_a, r2_a = eng.granger_matrix()
    Th_a, pc_a = eng.precision_matrix()
    lin_a = eng.contemporaneous_orientation()
    rca_a = eng.root_attribution(0)
    rscores_a = rca_a["root_scores"]
    pexo_a = rca_a["peak_exo_z"]
    praw_a = rca_a["raw_z"]
    onset_a = rca_a["onset_tick"]
    ranked_a = rca_a["ranked_channels"]
    maha_a = np.array([r.mahalanobis_sq for r in res_a], dtype=np.float64)
    eval_a = np.array([r.e_value for r in res_a], dtype=np.float64)
    w_a = np.array([r.shock_weight for r in res_a], dtype=np.float64)
    top_a = np.array([r.top_root_channel for r in res_a], dtype=np.int32)
    alarm_a = np.array([r.alarm for r in res_a], dtype=np.uint32)

# Stream B: hinge basis + exact expanding mode (D=13, p=2, K=53, T=131 across slab boundaries)
rng_b = np.random.default_rng(202605)
D_b, p_b, T_b = 13, 2, 131
X_b = np.zeros((T_b, D_b), dtype=np.float64)
for t in range(T_b):
    row = rng_b.standard_normal(D_b) * 0.7
    if t >= 1:
        row += 0.25 * X_b[t - 1]
        row[1:] += 0.15 * np.maximum(0.0, X_b[t - 1, :-1] - 1.0)
    X_b[t] = row
X_b = np.ascontiguousarray(X_b, dtype=np.float64)

with RawEngine(
    dim=D_b, lag=p_b, forgetting=1.0, ridge=0.1, precision_ridge=0.1, shock_threshold=0.0,
    warmup_ticks=p_b, nonlinear_hinge=1, normalize_scale=0, lib_path=DEFAULT_LIB_PATH
) as eng_b:
    rc, _, exo_b, innov_b = eng_b.push_batch(X_b, return_results=False, return_matrices=True)
    assert rc == 0
    assert eng_b.force_refresh() == 0
    B_b, freiv_b = eng_b.inspect_state()
    F_b, r2_b = eng_b.granger_matrix()
    Th_b, pc_b = eng_b.precision_matrix()

# Stream C: Tier-2 screened engine (non-multiple-of-4 D=257, M=16, p=2, T=180)
rng_c = np.random.default_rng(202606)
D_c, M_c, T_c = 257, 16, 180
X_c = rng_c.standard_normal((T_c, D_c)).astype(np.float64)
X_c[150:, 19] += 8.0
X_c = np.ascontiguousarray(X_c, dtype=np.float64)
with RawScreenedEngine(dim=D_c, active_m=M_c, lag=2, lib_path=DEFAULT_LIB_PATH) as sc:
    last_z_c, top_ch_c, top_sc_c = sc.push_batch(X_c, top_k=5)

np.savez(
    sys.argv[2],
    exo_a=exo_a, innov_a=innov_a, maha_a=maha_a, eval_a=eval_a, w_a=w_a,
    top_a=top_a, alarm_a=alarm_a, B_a=B_a, F_a=F_a, r2_a=r2_a, Th_a=Th_a,
    pc_a=pc_a, lin_a=lin_a, rscores_a=rscores_a, pexo_a=pexo_a, praw_a=praw_a,
    onset_a=onset_a, ranked_a=ranked_a,
    exo_b=exo_b, innov_b=innov_b, B_b=B_b, F_b=F_b, r2_b=r2_b, Th_b=Th_b, pc_b=pc_b,
    last_z_c=last_z_c, top_ch_c=top_ch_c, top_sc_c=top_sc_c,
)
print(json.dumps({"isa": isa, "freiv_a": freiv_a, "freiv_b": freiv_b}))
"""
    native_npz = tmp_path / "native.npz"
    scalar_npz = tmp_path / "scalar.npz"
    tests_dir = str(REPO_ROOT / "tests")

    env_native = os.environ.copy()
    env_native.pop("CAUSALSIEVE_FORCE_SCALAR", None)
    p_native = subprocess.run(
        [sys.executable, "-c", worker, tests_dir, str(native_npz)],
        env=env_native,
        capture_output=True,
        text=True,
        check=True,
    )
    meta_native = json.loads(p_native.stdout.strip())

    env_scalar = os.environ.copy()
    env_scalar["CAUSALSIEVE_FORCE_SCALAR"] = "1"
    p_scalar = subprocess.run(
        [sys.executable, "-c", worker, tests_dir, str(scalar_npz)],
        env=env_scalar,
        capture_output=True,
        text=True,
        check=True,
    )
    meta_scalar = json.loads(p_scalar.stdout.strip())

    assert meta_native["isa"] == _expected_native_isa()
    assert meta_scalar["isa"] == "scalar"
    assert meta_native["freiv_a"] < 1e-11
    assert meta_native["freiv_b"] < 1e-11

    d1 = np.load(native_npz)
    d2 = np.load(scalar_npz)

    def rel_max(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))

    def abs_max(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    tol = 1e-13
    # Stream A (D=17 non-mult-of-4, normalize_scale=1, forgetting=0.99, shock + LiNGAM + RCA)
    assert rel_max(d1["exo_a"], d2["exo_a"]) < tol
    assert rel_max(d1["innov_a"], d2["innov_a"]) < tol
    assert rel_max(d1["maha_a"], d2["maha_a"]) < tol
    assert rel_max(d1["eval_a"], d2["eval_a"]) < tol
    assert abs_max(d1["w_a"], d2["w_a"]) < tol
    np.testing.assert_array_equal(d1["top_a"], d2["top_a"])
    np.testing.assert_array_equal(d1["alarm_a"], d2["alarm_a"])
    assert rel_max(d1["B_a"], d2["B_a"]) < tol
    assert rel_max(d1["F_a"], d2["F_a"]) < tol
    assert abs_max(d1["r2_a"], d2["r2_a"]) < tol
    assert rel_max(d1["Th_a"], d2["Th_a"]) < tol
    assert abs_max(d1["pc_a"], d2["pc_a"]) < tol
    assert abs_max(d1["lin_a"], d2["lin_a"]) < tol
    assert rel_max(d1["rscores_a"], d2["rscores_a"]) < tol
    assert rel_max(d1["pexo_a"], d2["pexo_a"]) < tol
    assert rel_max(d1["praw_a"], d2["praw_a"]) < tol
    np.testing.assert_array_equal(d1["onset_a"], d2["onset_a"])
    np.testing.assert_array_equal(d1["ranked_a"], d2["ranked_a"])

    # Stream B (D=13, p=3, hinge=1, K=79, expanding mode)
    assert rel_max(d1["exo_b"], d2["exo_b"]) < tol
    assert rel_max(d1["innov_b"], d2["innov_b"]) < tol
    assert rel_max(d1["B_b"], d2["B_b"]) < tol
    assert rel_max(d1["F_b"], d2["F_b"]) < tol
    assert abs_max(d1["r2_b"], d2["r2_b"]) < tol
    assert rel_max(d1["Th_b"], d2["Th_b"]) < tol
    assert abs_max(d1["pc_b"], d2["pc_b"]) < tol

    # Stream C (Tier-2 screened D=257)
    assert rel_max(d1["last_z_c"], d2["last_z_c"]) < tol
    np.testing.assert_array_equal(d1["top_ch_c"], d2["top_ch_c"])
    assert rel_max(d1["top_sc_c"], d2["top_sc_c"]) < tol


def test_simd_all_finite_pd_every_lane_offset() -> None:
    """SIMD all_finite_pd rejects NaN, +Inf, and -Inf at every lane and tail offset atomically."""
    rng = np.random.default_rng(101)
    base = np.ascontiguousarray(rng.standard_normal((11, 3)), dtype=np.float64)  # 33 doubles
    bad_vals = (float("nan"), float("inf"), float("-inf"))

    with RawEngine(dim=3, lag=1, lib_path=DEFAULT_LIB_PATH) as eng:
        assert eng.push_batch(base[:4], return_results=False, return_matrices=False)[0] == 0
        t0 = eng.tick_count
        for offset in range(base.size):
            bad = base.copy()
            bad.ravel()[offset] = bad_vals[offset % 3]
            rc, _, _, _ = eng.push_batch(bad, return_results=False, return_matrices=False)
            assert rc == -1, f"Failed to reject non-finite value at flat offset {offset}"
            assert eng.tick_count == t0, f"Mutated state on non-finite value at offset {offset}"

        # Clean batch still succeeds right after rejected batches
        assert eng.push_batch(base, return_results=False, return_matrices=False)[0] == 0
        assert eng.tick_count == t0 + len(base)


def _copy_clean_package_tree(dst: Path) -> None:
    """Copy minimal clean source tree (excluding any prebuilt .so/.dylib/.dll) for wheel build."""
    dst.mkdir(parents=True, exist_ok=True)
    for fname in (
        "pyproject.toml",
        "setup.py",
        "MANIFEST.in",
        "LICENSE",
        "NOTICE",
        "README.md",
        "BENCHMARKS.md",
        "SKILL.md",
        "llms.txt",
        "llms-full.txt",
        "CITATION.cff",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "server.json",
        "Makefile",
    ):
        src_f = REPO_ROOT / fname
        if src_f.is_file():
            shutil.copy2(src_f, dst / fname)

    ignore = shutil.ignore_patterns(
        "*.so", "*.dylib", "*.dll", "*.pyc", "*.pyo", "__pycache__", "*.egg-info", ".DS_Store"
    )
    shutil.copytree(REPO_ROOT / "csrc", dst / "csrc", ignore=ignore)
    shutil.copytree(REPO_ROOT / "python", dst / "python", ignore=ignore)
    for extra_dir in ("skills", ".claude-plugin"):
        if (REPO_ROOT / extra_dir).is_dir():
            shutil.copytree(REPO_ROOT / extra_dir, dst / extra_dir, ignore=ignore)


def _build_portable_wheel(pkg_src: Path, wheel_dir: Path) -> Path:
    """Build a CAUSALSIEVE_PORTABLE=1 wheel using python -m build, uv build, or pip wheel."""
    env = os.environ.copy()
    env["CAUSALSIEVE_PORTABLE"] = "1"
    env.pop("PYTHONPATH", None)

    # Prefer `python -m build --wheel` if installed, else `uv build --wheel`, else `pip wheel`.
    has_build = (
        subprocess.run(
            [sys.executable, "-c", "import build.__main__"],
            cwd=str(wheel_dir),
            env=env,
            capture_output=True,
        ).returncode
        == 0
    )
    uv_bin = shutil.which("uv")
    if has_build:
        cmd = [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir), str(pkg_src)]
    elif uv_bin:
        cmd = [uv_bin, "build", "--wheel", "--out-dir", str(wheel_dir), str(pkg_src)]
    else:
        cmd = [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(wheel_dir), str(pkg_src)]

    subprocess.run(cmd, env=env, cwd=str(wheel_dir), check=True, capture_output=True, text=True)
    wheels = sorted(wheel_dir.glob("causalsieve-*.whl"))
    assert len(wheels) == 1, f"Expected 1 built wheel in {wheel_dir}, found {wheels}"
    return wheels[0]


def test_portable_wheel_dispatches_avx2_in_bare_venv(tmp_path: Path) -> None:
    """A CAUSALSIEVE_PORTABLE=1 wheel installed in a bare venv must dispatch native AVX2+FMA."""
    pkg_src = tmp_path / "pkg_src"
    wheel_dir = tmp_path / "wheels"
    venv_dir = tmp_path / "bare_venv"
    wheel_dir.mkdir(parents=True, exist_ok=True)

    _copy_clean_package_tree(pkg_src)
    wheel_path = _build_portable_wheel(pkg_src, wheel_dir)

    uv_bin = shutil.which("uv")
    if uv_bin:
        subprocess.run(
            [uv_bin, "venv", "--python", sys.executable, str(venv_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        venv_py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [uv_bin, "pip", "install", "--python", str(venv_py), "--no-deps", str(wheel_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        venv_py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(venv_py), "-m", "pip", "install", "--no-deps", str(wheel_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    probe = (
        "import json, causalsieve as cs; "
        "eng = cs.CausalSieve(dim=5, lag=2, warmup_ticks=12); "
        "eng.push_batch([[float((t + j) % 7) * 0.25 for j in range(5)] for t in range(45)]); "
        "gm = eng.granger_matrix(); "
        "pm = eng.precision_matrix(); "
        "lr = eng.contemporaneous_orientation(); "
        "rca = eng.root_attribution(); "
        "print(json.dumps({"
        "'isa': cs.simd_isa(), "
        "'lib': str(cs.library_path()), "
        "'dim': eng.dim, "
        "'ticks': eng.ticks, "
        "'f_rows': len(gm.f_stat), "
        "'pcorr_rows': len(pm.pcorr), "
        "'lr_rows': len(lr), "
        "'ranked_len': len(rca.ranked_channels)"
        "}))"
    )

    bare_env = os.environ.copy()
    bare_env.pop("PYTHONPATH", None)
    bare_env.pop("CAUSALSIEVE_LIB", None)
    bare_env.pop("CAUSALSIEVE_FORCE_SCALAR", None)

    p_native = subprocess.run(
        [str(venv_py), "-c", probe],
        cwd=str(tmp_path),
        env=bare_env,
        capture_output=True,
        text=True,
        check=True,
    )
    info_native = json.loads(p_native.stdout.strip())
    assert info_native["isa"] == _expected_native_isa()
    assert str(venv_dir.resolve()) in str(Path(info_native["lib"]).resolve())
    assert info_native["dim"] == 5
    assert info_native["ticks"] == 45
    assert info_native["f_rows"] == 5
    assert info_native["ranked_len"] == 5

    # Same installed portable wheel must switch cleanly to scalar when CAUSALSIEVE_FORCE_SCALAR=1
    bare_env["CAUSALSIEVE_FORCE_SCALAR"] = "1"
    p_scalar = subprocess.run(
        [str(venv_py), "-c", "import causalsieve as cs; print(cs.simd_isa())"],
        cwd=str(tmp_path),
        env=bare_env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert p_scalar.stdout.strip() == "scalar"
