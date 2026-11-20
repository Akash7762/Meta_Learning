"""
phase0_health_check.py
----------------------
Phase 0 of the v2 plan (docs/planning/phase0/phase0_plan.md): show with
evidence that the code imports, connects and runs, and compare it with
README.md. Decision D1: it reports; it fixes nothing.

It changes no code. It writes only to results/phase0/ and to a temporary
folder (deleted at the end unless --keep-temp). It never calls an LLM API:
API keys are blanked in this process and its subprocesses, and the one README
example that names a key runs with the network switched off.

Checks
  C1  imports                 C5  tiny working runs
  C2  dependencies            C6  known-defect register (16 items)
  C3  call map                C7  test suites
  C4  fit between modules     C8  README check

Run from the repository root with the main environment:
    .venv\\Scripts\\python.exe phase0_health_check.py              # everything
    .venv\\Scripts\\python.exe phase0_health_check.py --only C1,C4 # a subset
    .venv\\Scripts\\python.exe phase0_health_check.py --list       # describe, run nothing

Outputs
  results/phase0/health_report.txt   readable report
  results/phase0/health_report.json  every result, with evidence
  results/phase0/call_map.csv        C3: every public function, class and method
"""

import os
import sys

# Before anything imports aad: no bytecode files in the repository, a headless
# plotting backend, and no API key in this process (aad's load_dotenv() never
# overrides a variable that is already set, even to an empty string).
sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
SECRET_NAMES = ("GEMINI_API_KEY", "OPENAI_API_KEY")
for _name in SECRET_NAMES:
    os.environ[_name] = ""

import argparse
import ast
import collections
import contextlib
import csv
import datetime
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
OUT = ROOT / "results" / "phase0"
PFLACCO_PY = ROOT / ".venv-pflacco" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TEMP = None                      # set in main()
RESULTS = []                     # one dict per recorded item
STATE = {}                       # evidence shared between checks
TIMES = {}                       # seconds per check

CHECKS = {
    "C1": "imports: 12 modules in fresh processes, aad.__all__, every file compiles and its imports resolve",
    "C2": "dependencies: imports covered by requirements.txt, installed = pinned in both environments, "
          "pflacco imports, BBOB fingerprint identical in both environments",
    "C3": "call map: every public function/class/method in aad, who refers to it, whether the tests run it",
    "C4": "fit between modules: algorithm lists, build_algorithm, operator settings, 30 problem classes, "
          "feature keys, surrogate shapes",
    "C5": "tiny working runs: 25 algorithms twice each, EA_Designer, LLMDesigner offline, "
          "EvaluationFramework, Visualizer, experiment_runner.py on one tiny problem",
    "C6": "known-defect register: confirm or rule out the 16 items in the plan",
    "C7": "test suites: tests/ (pytest) and test_phase0_config_consistency.py",
    "C8": "README check: names, code examples run as written (timed), claims compared with the code",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def record(check, item, status, detail="", evidence=""):
    RESULTS.append({"check": check, "item": item, "status": status,
                    "detail": str(detail), "evidence": str(evidence)})
    print(f"  [{status}] {item}: {detail}"[:300], flush=True)


def rel(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def last_line(text):
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def tail(text, n=15):
    return "\n".join((text or "").rstrip().splitlines()[-n:])


def seed_all(seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def sub_env(keep_secret_names_out=False):
    """Environment for subprocesses: API keys blanked, unless the caller needs
    the names absent (the .env check), in which case they are removed."""
    env = dict(os.environ)
    env.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               MPLBACKEND="Agg", PYTHONPATH=str(ROOT))
    if keep_secret_names_out:
        for name in SECRET_NAMES:
            env.pop(name, None)
    return env


def run_py(code, cwd=None, timeout=600, python=None, env=None, args=()):
    t0 = time.perf_counter()
    cmd = [str(python or sys.executable), "-c", code, *[str(a) for a in args]]
    try:
        r = subprocess.run(cmd, cwd=str(cwd or ROOT), env=env or sub_env(),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return SimpleNamespace(rc=r.returncode, out=r.stdout, err=r.stderr,
                               secs=time.perf_counter() - t0, timeout=False)
    except subprocess.TimeoutExpired as exc:
        def _s(x):
            return x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")
        return SimpleNamespace(rc=None, out=_s(exc.stdout), err=_s(exc.stderr),
                               secs=time.perf_counter() - t0, timeout=True)


NO_NET_PRELUDE = (
    "import socket as _s\n"
    "def _phase0_no_net(*a, **k):\n"
    "    raise OSError('network disabled by phase0_health_check')\n"
    "_s.socket.connect = _phase0_no_net\n"
    "_s.socket.connect_ex = _phase0_no_net\n"
    "_s.create_connection = _phase0_no_net\n"
    "_s.getaddrinfo = _phase0_no_net\n"
)


@contextlib.contextmanager
def no_network():
    import socket
    saved = (socket.socket.connect, socket.socket.connect_ex,
             socket.create_connection, socket.getaddrinfo)

    def blocked(*a, **k):
        raise OSError("network disabled by phase0_health_check")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    try:
        yield
    finally:
        (socket.socket.connect, socket.socket.connect_ex,
         socket.create_connection, socket.getaddrinfo) = saved


@contextlib.contextmanager
def captured():
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield out, err, caught


class Counter:
    """Counts the evaluations made on a real problem instance, and how many of
    them were outside the problem's bounds."""

    def __init__(self, problem):
        import numpy as np
        self.n = 0
        self.out_of_bounds = 0
        lb, ub = problem.bounds()
        self._lb = np.asarray(lb, dtype=float)
        self._ub = np.asarray(ub, dtype=float)
        original = problem.evaluate

        def evaluate(solution, _orig=original):
            self.n += 1
            x = np.asarray(solution, dtype=float)
            if np.any(x < self._lb) or np.any(x > self._ub):
                self.out_of_bounds += 1
            return _orig(solution)

        problem.evaluate = evaluate


def project_files():
    files = sorted(ROOT.glob("*.py")) + sorted((ROOT / "aad").glob("*.py")) \
        + sorted((ROOT / "tests").glob("*.py"))
    return [f for f in files if f.is_file()]


def iter_imports(tree, path):
    """Yield (module, names, inside_try, lineno) for every import in *tree*."""
    in_try = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    in_try.add(id(sub))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, [], id(node) in in_try, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = rel(path).split("/")[:-1]
                base = pkg[: len(pkg) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module
            yield mod, [a.name for a in node.names], id(node) in in_try, node.lineno


def resolve_import(mod, names):
    try:
        if importlib.util.find_spec(mod) is None:
            return False, "module not found"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    missing = []
    for name in names:
        if name == "*":
            continue
        try:
            module = importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            return False, f"import failed: {type(exc).__name__}: {exc}"
        if hasattr(module, name):
            continue
        try:
            if importlib.util.find_spec(f"{mod}.{name}") is not None:
                continue
        except Exception:  # noqa: BLE001
            pass
        missing.append(name)
    return (not missing), ("missing: " + ", ".join(missing) if missing else "")


def norm_dist(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path):
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[norm_dist(name)] = ver.strip()
    return pins


def problem_classes():
    import aad
    classic = [aad.SphereProblem, aad.RastriginProblem, aad.RosenbrockProblem,
               aad.AckleyProblem, aad.ZDT1Problem, aad.ZDT2Problem]
    bbob = [getattr(aad, n) for n in aad.__all__ if n.startswith("BBOB_F")]
    bbob.sort(key=lambda c: int(re.match(r"BBOB_F(\d+)_", c.__name__).group(1)))
    return classic + bbob


def f1_sphere():
    from aad import BBOB_F1_Sphere
    return BBOB_F1_Sphere(dim=5, seed=100)


# ---------------------------------------------------------------------------
# C1  imports
# ---------------------------------------------------------------------------

def c1_imports():
    modules = ["aad"] + [f"aad.{p.stem}" for p in sorted((ROOT / "aad").glob("*.py"))
                         if p.stem != "__init__"]
    for mod in modules:
        code = ("import time, importlib\nt = time.perf_counter()\n"
                f"importlib.import_module({mod!r})\nprint(round(time.perf_counter() - t, 2))")
        r = run_py(code, timeout=300)
        if r.rc == 0:
            record("C1", f"import {mod}", "PASS", f"{r.out.strip()} s in a fresh process")
        else:
            record("C1", f"import {mod}", "FAIL", last_line(r.err), tail(r.err, 25))
    record("C1", "modules imported", "INFO", f"{len(modules)} (aad and its {len(modules) - 1} modules)")

    import aad
    missing = [n for n in aad.__all__ if not hasattr(aad, n)]
    dupes = sorted({n for n in aad.__all__ if aad.__all__.count(n) > 1})
    record("C1", "aad.__all__", "PASS" if not (missing or dupes) else "FAIL",
           f"{len(aad.__all__)} names; missing: {missing or 'none'}; duplicated: {dupes or 'none'}")

    for path in project_files():
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(path))
            compile(tree, str(path), "exec")            # compiles in memory, writes nothing
        except SyntaxError as exc:
            record("C1", f"compile {rel(path)}", "FAIL", f"SyntaxError line {exc.lineno}: {exc.msg}")
            continue
        imports = list(iter_imports(tree, path))
        bad = []
        for mod, names, in_try, lineno in imports:
            ok, why = resolve_import(mod, names)
            if not ok:
                bad.append((lineno, mod, names, why, in_try))
        hard = [b for b in bad if not b[4]]
        status = "FAIL" if hard else ("WARN" if bad else "PASS")
        detail = f"compiles; {len(imports)} imports, " + (
            "all resolve" if not bad else
            "; ".join(f"line {b[0]} {b[1]} {b[2] or ''} ({b[3]}{', inside try' if b[4] else ''})" for b in bad))
        record("C1", f"compile + imports {rel(path)}", status, detail)


# ---------------------------------------------------------------------------
# C2  dependencies
# ---------------------------------------------------------------------------

PFLACCO_ENV_CODE = r'''
import json, re, sys, importlib, importlib.metadata as md
def norm(n): return re.sub(r"[-_.]+", "-", n).lower()
pins = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.split("#", 1)[0].strip()
    if "==" in line:
        n, v = line.split("==", 1); pins[norm(n)] = v.strip()
out = {"pins": len(pins), "mismatch": [], "missing": []}
for n, v in pins.items():
    try:
        i = md.version(n)
    except md.PackageNotFoundError:
        out["missing"].append(n); continue
    if i != v:
        out["mismatch"].append(f"{n} {i} (pinned {v})")
for mod in ("pflacco", "pflacco.classical_ela_features"):
    try:
        importlib.import_module(mod); out[mod] = "ok"
    except Exception as e:
        out[mod] = f"{type(e).__name__}: {e}"
out["python"] = sys.version.split()[0]
print(json.dumps(out))
'''

FINGERPRINT_CODE = r'''
import hashlib, importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("aad_problems_fp", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
pts = np.load(sys.argv[2])
out = {}
for f in range(1, 25):
    cls = next(getattr(mod, n) for n in dir(mod) if n.startswith(f"BBOB_F{f}_"))
    for d in (5, 10, 20):
        p = cls(dim=d, seed=100)
        vals = np.array([float(p.evaluate(x)) for x in pts[f"d{d}"]], dtype=np.float64)
        out[f"F{f}_d{d}"] = [hashlib.sha256(vals.tobytes()).hexdigest(), vals.tolist()]
print(json.dumps({"numpy": np.__version__, "fp": out}))
'''


def c2_dependencies():
    import numpy as np
    pins = read_pins(ROOT / "requirements.txt")
    record("C2", "requirements.txt", "INFO", f"{len(pins)} pinned packages")

    stdlib = set(sys.stdlib_module_names)
    local = {"aad", "tests", "conftest"} | {p.stem for p in ROOT.glob("*.py")}
    tops = collections.defaultdict(set)
    for path in project_files():
        if path.resolve() == THIS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for mod, _names, _in_try, _ln in iter_imports(tree, path):
            top = (mod or "").split(".")[0]
            if top and top not in stdlib and top not in local:
                tops[top].add(rel(path))
    providers = importlib.metadata.packages_distributions()
    for top in sorted(tops):
        dists = sorted(set(providers.get(top, [])))
        if not dists:
            record("C2", f"import '{top}' covered", "FAIL", "no installed distribution provides it",
                   ", ".join(sorted(tops[top])))
            continue
        unpinned = [d for d in dists if norm_dist(d) not in pins]
        record("C2", f"import '{top}' covered", "FAIL" if len(unpinned) == len(dists) else "PASS",
               f"provided by {', '.join(dists)}" + (f"; not pinned: {', '.join(unpinned)}" if unpinned else ""),
               "used in " + ", ".join(sorted(tops[top])))

    mismatch, missing = [], []
    for name, ver in pins.items():
        try:
            inst = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if inst != ver:
            mismatch.append(f"{name} {inst} (pinned {ver})")
    record("C2", "main .venv: installed = pinned", "FAIL" if (mismatch or missing) else "PASS",
           f"{len(pins)} pins; differing: {mismatch or 'none'}; not installed: {missing or 'none'}")
    installed = {norm_dist(d.metadata["Name"]) for d in importlib.metadata.distributions()}
    extra = sorted(installed - set(pins) - {"pip", "setuptools", "wheel"})
    record("C2", "main .venv: installed but not pinned", "INFO" if not extra else "WARN",
           ", ".join(extra) if extra else "none")

    if not PFLACCO_PY.exists():
        record("C2", ".venv-pflacco", "FAIL", f"interpreter not found: {rel(PFLACCO_PY)}")
        return
    r = run_py(PFLACCO_ENV_CODE, python=PFLACCO_PY, args=[ROOT / "requirements-pflacco.txt"], timeout=300)
    if r.rc != 0:
        record("C2", ".venv-pflacco", "FAIL", last_line(r.err), tail(r.err, 25))
        return
    info = json.loads(last_line(r.out))
    ok = not (info["mismatch"] or info["missing"])
    record("C2", ".venv-pflacco: installed = pinned", "PASS" if ok else "FAIL",
           f"Python {info['python']}; {info['pins']} pins; differing: {info['mismatch'] or 'none'}; "
           f"not installed: {info['missing'] or 'none'}")
    imp_ok = info["pflacco"] == "ok" and info["pflacco.classical_ela_features"] == "ok"
    record("C2", ".venv-pflacco: pflacco imports", "PASS" if imp_ok else "FAIL",
           f"pflacco: {info['pflacco']}; pflacco.classical_ela_features: {info['pflacco.classical_ela_features']}")

    # Fingerprint: the same 20 points per (function, dimension), evaluated in
    # both environments, must give bit-identical values.
    rng = np.random.default_rng(20261120)
    pts_file = TEMP / "fingerprint_points.npz"
    np.savez(pts_file, **{f"d{d}": rng.uniform(-5.0, 5.0, (20, d)) for d in (5, 10, 20)})
    prob_file = ROOT / "aad" / "problems.py"
    runs = {}
    for label, py in (("main", sys.executable), ("pflacco", PFLACCO_PY)):
        rr = run_py(FINGERPRINT_CODE, python=py, args=[prob_file, pts_file], timeout=600)
        if rr.rc != 0:
            record("C2", f"fingerprint ({label} environment)", "FAIL", last_line(rr.err), tail(rr.err, 25))
            return
        runs[label] = json.loads(last_line(rr.out))
    a, b = runs["main"]["fp"], runs["pflacco"]["fp"]
    same = [k for k in a if a[k][0] == b.get(k, [None])[0]]
    diff = [k for k in a if k not in same]
    worst = 0.0
    for k in diff:
        va, vb = np.array(a[k][1]), np.array(b[k][1])
        worst = max(worst, float(np.max(np.abs(va - vb) / np.maximum(np.abs(va), 1e-300))))
    STATE["fingerprint"] = {"identical": len(same), "total": len(a), "different": diff}
    record("C2", "BBOB fingerprint, both environments", "PASS" if not diff else "FAIL",
           f"{len(same)} of {len(a)} (24 functions x dims 5/10/20, seed 100, 20 points each) bit-identical; "
           f"numpy {runs['main']['numpy']} vs {runs['pflacco']['numpy']}"
           + (f"; largest relative difference {worst:.3g} in {diff}" if diff else ""))


# ---------------------------------------------------------------------------
# Shared pytest run (C3 call map and C7 test suites)
# ---------------------------------------------------------------------------

class _PytestCollector:
    def __init__(self):
        self.reports = []
        self.warnings = []

    def pytest_runtest_logreport(self, report):
        self.reports.append((report.nodeid, report.when, report.outcome))

    def pytest_warning_recorded(self, warning_message, when, nodeid, location):
        self.warnings.append({"nodeid": nodeid, "when": when,
                              "category": warning_message.category.__name__,
                              "message": str(warning_message.message),
                              "file": rel(warning_message.filename),
                              "line": warning_message.lineno})


def pytest_run():
    """Run tests/ once, in this process, under cProfile; cached."""
    if "pytest" in STATE:
        return STATE["pytest"]
    import cProfile
    import pytest
    collector = _PytestCollector()
    prof = cProfile.Profile()
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        rc = prof.runcall(pytest.main, ["-q", "-p", "no:cacheprovider", str(ROOT / "tests")],
                          plugins=[collector])
    secs = time.perf_counter() - t0
    prof.create_stats()
    called = collections.defaultdict(set)
    for (fname, lineno, funcname) in prof.stats:
        if fname and not fname.startswith("~"):
            called[(os.path.normcase(os.path.abspath(fname)), funcname)].add(lineno)
    reps = collector.reports
    STATE["pytest"] = {
        "rc": int(rc), "secs": secs, "output": buf.getvalue(), "called": called,
        "passed": sum(1 for _, w, o in reps if w == "call" and o == "passed"),
        "failed": sorted({n for n, _, o in reps if o == "failed"}),
        "skipped": sorted({n for n, _, o in reps if o == "skipped"}),
        "warnings": collector.warnings,
    }
    (TEMP / "pytest_output.txt").write_text(buf.getvalue(), encoding="utf-8")
    return STATE["pytest"]


# ---------------------------------------------------------------------------
# C3  call map
# ---------------------------------------------------------------------------

def _inventory():
    items = []

    def first_line(node):
        return min([d.lineno for d in node.decorator_list] + [node.lineno])

    for path in sorted((ROOT / "aad").glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                items.append({"file": path, "qualname": node.name, "name": node.name, "kind": "function",
                              "line": node.lineno, "spans": [(node.name, first_line(node), node.lineno)]})
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                defs = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                items.append({"file": path, "qualname": node.name, "name": node.name, "kind": "class",
                              "line": node.lineno,
                              "spans": [(d.name, first_line(d), d.lineno) for d in defs]})
                for d in defs:
                    if not d.name.startswith("_"):
                        items.append({"file": path, "qualname": f"{node.name}.{d.name}", "name": d.name,
                                      "kind": "method", "line": d.lineno,
                                      "spans": [(d.name, first_line(d), d.lineno)]})
    return items


def _references():
    refs = collections.defaultdict(list)
    for path in project_files():
        if path.resolve() == THIS or (path.name == "__init__.py" and path.parent.name == "aad"):
            continue
        if path.parent.name == "tests" or path.name == "test_phase0_config_consistency.py":
            kind = "test"
        elif path.name in ("user_model_runner.py", "test_runner.py"):
            kind = "script"
        else:
            kind = "pipeline"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                refs[node.id].append((rel(path), node.lineno, kind))
            elif isinstance(node, ast.Attribute):
                refs[node.attr].append((rel(path), node.lineno, kind))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    refs[alias.name].append((rel(path), node.lineno, kind))
    return refs


def c3_call_map():
    items = _inventory()
    refs = _references()
    run = pytest_run()
    called = run["called"]
    name_count = collections.Counter(i["name"] for i in items)
    rows = []
    for it in items:
        locs = refs.get(it["name"], [])
        n_pipe = sum(1 for _, _, k in locs if k == "pipeline")
        n_script = sum(1 for _, _, k in locs if k == "script")
        n_test = sum(1 for _, _, k in locs if k == "test")
        fkey = os.path.normcase(str(it["file"].resolve()))
        ran = any(any(lo <= ln <= hi for ln in called.get((fkey, name), ()))
                  for name, lo, hi in it["spans"])
        if not it["spans"]:
            ran_text = "n/a"
        else:
            ran_text = "yes" if ran else "no"
        if n_pipe:
            status = "used, tested" if ran else "used, not run by tests"
        elif n_script or n_test:
            status = "used only by " + ("reference scripts" if n_script and not n_test else
                                        "tests" if n_test and not n_script else "scripts and tests")
        else:
            status = "unused"
        examples = "; ".join(f"{f}:{ln}" for f, ln, k in locs if k == "pipeline")[:200]
        note = (f"name shared by {name_count[it['name']]} definitions: references counted by name"
                if name_count[it["name"]] > 1 else "")
        rows.append({"file": rel(it["file"]), "qualname": it["qualname"], "kind": it["kind"],
                     "line": it["line"], "pipeline_refs": n_pipe, "script_refs": n_script,
                     "test_refs": n_test, "pipeline_examples": examples, "run_by_tests": ran_text,
                     "status": status, "note": note})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "call_map.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    STATE["call_map"] = rows
    counts = collections.Counter(r["status"] for r in rows)
    kinds = collections.Counter(r["kind"] for r in rows)
    record("C3", "call map written", "INFO",
           f"{len(rows)} items in aad ({kinds['function']} functions, {kinds['class']} classes, "
           f"{kinds['method']} methods) -> results/phase0/call_map.csv; " +
           ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    for status in ("unused", "used only by tests", "used only by reference scripts",
                   "used only by scripts and tests", "used, not run by tests"):
        names = [r["qualname"] for r in rows if r["status"] == status]
        if names:
            record("C3", f"items {status}", "WARN" if status != "used, not run by tests" else "INFO",
                   f"{len(names)}: " + ", ".join(names))


# ---------------------------------------------------------------------------
# C4  fit between modules
# ---------------------------------------------------------------------------

def operator_evidence():
    """mealpy algorithms ignore the operator keys; ea uses them. Cached."""
    if "operators" in STATE:
        return STATE["operators"]
    import numpy as np
    from aad import mealpy_algorithms as ma
    from aad.evolutionary_algorithm import build_algorithm
    base = {"algorithm": "PSO", "population_size": 20, "max_fe": 500, "seed": 3}
    cfg_a = dict(base, selection="tournament", crossover="sbx", mutation="gaussian")
    cfg_b = dict(base, selection="roulette", crossover="uniform", mutation="polynomial")
    seed_all(3)
    sol_a, fit_a = build_algorithm(cfg_a).run(f1_sphere())
    seed_all(3)
    sol_b, fit_b = build_algorithm(cfg_b).run(f1_sphere())
    ea_a = {"algorithm": "ea", "population_size": 20, "max_generations": 20,
            "selection": "tournament", "crossover": "sbx", "mutation": "gaussian"}
    ea_b = dict(ea_a, selection="roulette", crossover="uniform", mutation="polynomial")
    seed_all(3)
    _, ea_fit_a = build_algorithm(ea_a).run(f1_sphere())
    seed_all(3)
    _, ea_fit_b = build_algorithm(ea_b).run(f1_sphere())
    src = Path(ma.__file__).read_text(encoding="utf-8").splitlines()
    hits = [i for i, ln in enumerate(src, 1)
            if re.search(r"[\"'](selection|crossover|mutation)[\"']", ln)]
    STATE["operators"] = {
        "mealpy_same": bool(fit_a == fit_b and np.array_equal(sol_a, sol_b)),
        "pso_fits": (fit_a, fit_b), "ea_differs": bool(ea_fit_a != ea_fit_b),
        "ea_fits": (ea_fit_a, ea_fit_b), "static_hits": hits}
    return STATE["operators"]


def feature_cost():
    """Evaluations used by one feature extraction on F1 at dims 5 and 10. Cached."""
    if "feature_cost" in STATE:
        return STATE["feature_cost"]
    from aad import BBOB_F1_Sphere
    from aad.landscape_features import LandscapeFeatureExtractor
    cost = {}
    for d in (5, 10):
        p = BBOB_F1_Sphere(dim=d, seed=100)
        c = Counter(p)
        seed_all(0)
        LandscapeFeatureExtractor().extract_features(p)
        cost[d] = c.n
    STATE["feature_cost"] = cost
    return cost


def c4_fit():
    import numpy as np
    import aad
    from aad import config_space as cs, mealpy_algorithms as ma, llm_designer as ld, problems
    from aad.evolutionary_algorithm import build_algorithm, EvolutionaryAlgorithm
    from aad.mealpy_algorithms import MealpyAlgorithm

    # Algorithm lists
    reg = sorted(ma.MEALPY_REGISTRY)
    dup = list(cs.MEALPY_ALGORITHM_KEYS)
    ext = list(cs.EXTENDED_CONFIG_SPACE["algorithm"])
    llm = sorted(k for k in ld.ALGORITHM_NAMES if k != "ea")
    problems_found = []
    if sorted(dup) != reg:
        problems_found.append(f"config_space list vs registry: only in list {sorted(set(dup) - set(reg))}, "
                              f"only in registry {sorted(set(reg) - set(dup))}")
    if len(set(dup)) != len(dup):
        problems_found.append("config_space list has duplicates")
    if ext != ["ea"] + dup:
        problems_found.append("EXTENDED_CONFIG_SPACE['algorithm'] is not ['ea'] + the list")
    if llm != reg:
        problems_found.append(f"LLM names vs registry: only in LLM {sorted(set(llm) - set(reg))}, "
                              f"only in registry {sorted(set(reg) - set(llm))}")
    if "ea" not in ld.ALGORITHM_NAMES:
        problems_found.append("LLM names have no 'ea'")
    STATE["algorithms"] = ext
    STATE["lists_ok"] = not problems_found
    record("C4", "algorithm lists agree", "PASS" if not problems_found else "FAIL",
           f"registry {len(reg)}, config_space list {len(dup)}, EXTENDED options {len(ext)}, "
           f"LLM names {len(llm)} + 'ea'" + ("; " + "; ".join(problems_found) if problems_found else ""))

    # build_algorithm reaches every algorithm
    unreachable = []
    for a in ext:
        try:
            obj = build_algorithm({"algorithm": a, "population_size": 20, "max_generations": 10})
            want = EvolutionaryAlgorithm if a == "ea" else MealpyAlgorithm
            if not isinstance(obj, want):
                unreachable.append(f"{a}: got {type(obj).__name__}")
            elif isinstance(obj, MealpyAlgorithm):
                obj._make_model()
        except Exception as exc:  # noqa: BLE001
            unreachable.append(f"{a}: {type(exc).__name__}: {exc}")
    record("C4", "build_algorithm reaches every algorithm", "PASS" if not unreachable else "FAIL",
           f"{len(ext) - len(unreachable)} of {len(ext)} build and load their class"
           + (f"; failed: {unreachable}" if unreachable else ""))

    # Operator settings are read only by ea
    op = operator_evidence()
    ok = op["mealpy_same"] and op["ea_differs"] and not op["static_hits"]
    record("C4", "operator settings read only by ea", "PASS" if ok else "FAIL",
           f"PSO with two different operator sets, same seed: identical result = {op['mealpy_same']}; "
           f"ea with two operator sets, same seed: results differ = {op['ea_differs']}; "
           f"operator keys named in mealpy_algorithms.py at lines {op['static_hits'] or 'none'}")

    # Problem classes
    classes = problem_classes()
    concrete = sorted(n for n, c in vars(problems).items()
                      if inspect.isclass(c) and issubclass(c, problems.OptimizationProblem)
                      and not inspect.isabstract(c) and not n.startswith("_"))
    record("C4", "problem classes exported", "PASS" if len(concrete) == len(classes) == 30 else "FAIL",
           f"{len(classes)} exported (6 classic + 24 BBOB); {len(concrete)} concrete classes in problems.py")
    broken, warned = [], []
    for cls in classes:
        try:
            with captured() as (_o, _e, caught):
                p = cls(dim=5)
                lb, ub = p.bounds()
                mo = p.is_multiobjective()
                name = p.name
                pts = np.random.default_rng(1).uniform(lb, ub, (10, 5))
                vals = [p.evaluate(x) for x in pts]
            errs = []
            if len(lb) != 5 or len(ub) != 5 or not all(l < u for l, u in zip(lb, ub)):
                errs.append("bad bounds")
            if not isinstance(mo, bool):
                errs.append("is_multiobjective not bool")
            if not isinstance(name, str):
                errs.append("name not str")
            for v in vals:
                if mo and not (isinstance(v, (list, tuple)) and len(v) == 2 and np.all(np.isfinite(v))):
                    errs.append(f"bad multi-objective value {v!r}")
                    break
                if not mo and not (isinstance(v, float) and np.isfinite(v)):
                    errs.append(f"bad value {v!r} ({type(v).__name__})")
                    break
            if errs:
                broken.append(f"{cls.__name__}: {', '.join(errs)}")
            if caught:
                warned.append(f"{cls.__name__}: {caught[0].category.__name__}: {caught[0].message}")
        except Exception as exc:  # noqa: BLE001
            broken.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
    record("C4", "30 problem classes: evaluate, bounds, is_multiobjective, name",
           "PASS" if not broken else "FAIL",
           f"{len(classes) - len(broken)} of {len(classes)} pass (dim 5, 10 points inside the bounds)"
           + (f"; failed: {broken}" if broken else "") + (f"; warnings: {warned}" if warned else ""))

    # Feature extractor: same 9 keys everywhere
    from aad.landscape_features import LandscapeFeatureExtractor
    fx = LandscapeFeatureExtractor()
    keysets, nonfinite, failed = {}, {}, []
    for cls in classes:
        try:
            p = cls(dim=5)
            seed_all(0)
            with captured():
                feats = fx.extract_features(p)
            keysets[cls.__name__] = tuple(sorted(feats))
            bad = [k for k, v in feats.items() if not np.isfinite(v)]
            if bad:
                nonfinite[cls.__name__] = bad
            if cls.__name__ == "BBOB_F1_Sphere":
                STATE["f1_features"] = fx.features_to_vector(feats)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
    distinct = set(keysets.values())
    ok = not failed and len(distinct) == 1 and len(next(iter(distinct))) == 9
    record("C4", "feature extractor returns the same 9 keys", "PASS" if ok else "FAIL",
           f"{len(keysets)} of {len(classes)} problems; distinct key sets: {len(distinct)}"
           + (f"; keys: {list(next(iter(distinct)))}" if len(distinct) == 1 else "")
           + (f"; failed: {failed}" if failed else ""))
    if nonfinite:
        record("C4", "non-finite feature values", "WARN", json.dumps(nonfinite))

    # Surrogate input and output shapes
    from aad.config_space import config_to_vector, vector_to_config, random_config, EXTENDED_CONFIG_SPACE
    from aad.surrogate_model import SurrogateModel
    random.seed(0)
    cfgs = [random_config(EXTENDED_CONFIG_SPACE) for _ in range(30)]
    vecs = [config_to_vector(c, EXTENDED_CONFIG_SPACE) for c in cfgs]
    expect = sum(1 if isinstance(v[0], (int, float)) else len(v) for v in EXTENDED_CONFIG_SPACE.values())
    lengths = {len(v) for v in vecs}
    roundtrip = all(vector_to_config(v, EXTENDED_CONFIG_SPACE) == c for v, c in zip(vecs, cfgs))
    feat = STATE.get("f1_features", np.zeros(9))
    X = np.array([np.concatenate([feat, v]) for v in vecs])
    y = np.random.default_rng(0).random(len(X))
    shape_notes, shapes_ok = [], True
    for mt in ("rf", "gp"):
        try:
            m = SurrogateModel(model_type=mt, n_jobs=1)
            with captured():
                m.fit(X, y)
                p3 = np.asarray(m.predict(X[:3]))
                p1 = np.asarray(m.predict(X[0]))
                mu, sd = m.predict_with_uncertainty(X[:3])
            ok_shapes = p3.shape == (3,) and p1.shape == (1,) and np.shape(mu) == (3,) and np.shape(sd) == (3,)
            shape_notes.append(f"{mt}: predict(3 rows)->{p3.shape}, predict(1-D row)->{p1.shape}, "
                               f"uncertainty->{np.shape(mu)},{np.shape(sd)}" + ("" if ok_shapes else " WRONG"))
            shapes_ok = shapes_ok and ok_shapes
        except Exception as exc:  # noqa: BLE001
            shape_notes.append(f"{mt}: {type(exc).__name__}: {exc}")
            shapes_ok = False
    ok = lengths == {expect} and roundtrip and shapes_ok
    record("C4", "surrogate input and output shapes", "PASS" if ok else "FAIL",
           f"config vector length {sorted(lengths)} (expected {expect}); +9 features = {expect + 9} inputs; "
           f"config -> vector -> config round trip: {roundtrip}; " + "; ".join(shape_notes))


# ---------------------------------------------------------------------------
# C5  tiny working runs
# ---------------------------------------------------------------------------

def c5_tiny_runs():
    import numpy as np
    from aad import config_space as cs, mealpy_algorithms as ma
    from aad.evolutionary_algorithm import build_algorithm
    budget, pop, seed = 2000, 20, 7
    algs = STATE.get("algorithms") or ["ea"] + list(cs.MEALPY_ALGORITHM_KEYS)
    rows = []
    for a in algs:
        if a == "ea":
            cfg = {"algorithm": "ea", "population_size": pop, "max_generations": 50, "selection": "tournament",
                   "crossover": "sbx", "mutation": "gaussian", "max_fe": budget, "seed": seed}
        else:
            cfg = {"algorithm": a, "population_size": pop, "max_fe": budget, "seed": seed}
            ma._COST_CACHE.pop((a, pop), None)      # run 1 pays the calibration probes, run 2 does not
        runs = []
        for _rep in (1, 2):
            p = f1_sphere()
            c = Counter(p)
            seed_all(seed)
            t0 = time.perf_counter()
            try:
                with captured():
                    sol, fit = build_algorithm(cfg).run(p)
                err = None
            except Exception as exc:  # noqa: BLE001
                sol, fit, err = None, None, f"{type(exc).__name__}: {exc}"
            runs.append({"sol": sol, "fit": fit, "n": c.n, "oob": c.out_of_bounds,
                         "secs": time.perf_counter() - t0, "err": err})
        r1, r2 = runs
        row = {"algorithm": a, "best_run1": r1["fit"], "best_run2": r2["fit"],
               "evals_run1": r1["n"], "evals_run2": r2["n"], "out_of_bounds_evals": r2["oob"],
               "secs_run2": round(r2["secs"], 2)}
        if r1["err"] or r2["err"]:
            status, detail = "FAIL", r1["err"] or r2["err"]
        elif not (np.isfinite(r1["fit"]) and np.isfinite(r2["fit"])):
            status, detail = "FAIL", f"non-finite result {r1['fit']}, {r2['fit']}"
        elif r1["fit"] != r2["fit"] or not np.array_equal(r1["sol"], r2["sol"]):
            status, detail = "FAIL", f"not repeatable: {r1['fit']!r} vs {r2['fit']!r}"
        elif a == "ea":
            status = "EXCEPTION"
            detail = (f"ignores max_fe={budget}: used {r2['n']} evaluations "
                      f"(pop x (generations + 1) = {pop * 51})")
        elif r2["n"] > budget:
            status = "EXCEPTION"
            detail = f"over budget: {r2['n']} search evaluations for max_fe={budget} (+{r2['n'] - budget})"
        else:
            status, detail = "PASS", f"{r2['n']} search evaluations for max_fe={budget}"
        if status != "FAIL":
            detail += (f"; best {r2['fit']:.4g}, identical twice; calibration probes "
                       f"{r1['n'] - r2['n']} extra evaluations; {row['secs_run2']} s")
            if r2["oob"]:
                detail += f"; {r2['oob']} evaluations outside the bounds"
        row["status"] = status
        rows.append(row)
        record("C5", f"tiny run {a}", status, detail)
    STATE["c5_rows"] = rows
    n_ok = sum(1 for r in rows if r["status"] in ("PASS", "EXCEPTION"))
    record("C5", "algorithms summary", "PASS" if n_ok == len(rows) else "FAIL",
           f"{n_ok} of {len(rows)} finish, are finite and repeat exactly (BBOB F1, dim 5, seed 100, "
           f"pop {pop}, max_fe {budget}, seed {seed}); budget exceptions: "
           f"{[r['algorithm'] for r in rows if r['status'] == 'EXCEPTION'] or 'none'}; "
           f"evaluate outside the bounds: {[r['algorithm'] for r in rows if r['out_of_bounds_evals']] or 'none'}")

    # Feature extraction (C4 runs it on all 30 problems; here its cost)
    cost = feature_cost()
    record("C5", "feature extraction", "PASS", f"evaluations per extraction: dim 5 -> {cost[5]}, dim 10 -> {cost[10]}")

    # EA_Designer, tiny
    from aad import EA_Designer, EXTENDED_CONFIG_SPACE
    space = dict(EXTENDED_CONFIG_SPACE, population_size=[20], max_generations=[20])
    seed_all(1)
    try:
        d = EA_Designer(config_space=space, population_size=4, generations=1, runs_per_eval=1)
        t0 = time.perf_counter()
        with captured():
            best_cfg, best_perf = d.evolve(f1_sphere(), verbose=False)
        mem = np.array(d._mem_X, dtype=float)
        const = bool(mem.shape[0] > 1 and np.all(mem[:, :9] == mem[0, :9]))
        STATE["designer_mem"] = {"rows": int(mem.shape[0]), "feature_cols_constant": const,
                                 "runs_used": d.runs_used}
        ok = np.isfinite(best_perf) and best_cfg["algorithm"] in space["algorithm"]
        record("C5", "EA_Designer, tiny", "PASS" if ok else "FAIL",
               f"best {best_cfg['algorithm']} {best_perf:.4g}; {d.runs_used} runs; surrogate memory "
               f"{mem.shape[0]} rows; meta_convergence {len(d.meta_convergence)} points; "
               f"{time.perf_counter() - t0:.1f} s (space reduced to pop 20, 20 generations)")
    except Exception as exc:  # noqa: BLE001
        record("C5", "EA_Designer, tiny", "FAIL", f"{type(exc).__name__}: {exc}", traceback.format_exc())

    # LLMDesigner, offline
    from aad import LLMDesigner
    try:
        with no_network(), captured() as (out, _e, _w):
            des = LLMDesigner()
            text = des.generate_description(f1_sphere())
            cfg = des.propose_algorithm(text)
            build_algorithm(cfg)
        valid = all(k in EXTENDED_CONFIG_SPACE and cfg[k] in EXTENDED_CONFIG_SPACE[k] for k in cfg)
        record("C5", "LLMDesigner offline (no key)", "PASS" if valid else "FAIL",
               f"rule-based proposal {cfg}; every value is in EXTENDED_CONFIG_SPACE: {valid}")
        try:
            with no_network(), captured():
                des.select_algorithm(text)
            record("C5", "LLMDesigner.select_algorithm offline", "WARN", "returned without a key")
        except Exception as exc:  # noqa: BLE001
            record("C5", "LLMDesigner.select_algorithm offline", "PASS",
                   f"refuses without a key, as documented: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        record("C5", "LLMDesigner offline (no key)", "FAIL", f"{type(exc).__name__}: {exc}", traceback.format_exc())

    # EvaluationFramework and Visualizer
    from aad import EvaluationFramework, Visualizer
    try:
        cfg = {"algorithm": "PSO", "population_size": 20, "max_fe": 1000, "seed": 7}
        with captured() as (out, _e, _w):
            metrics = EvaluationFramework(n_runs=2).evaluate(cfg, f1_sphere())
            EvaluationFramework.summarize(metrics)
        STATE["metrics_keys"] = sorted(metrics)
        STATE["summary_text"] = out.getvalue()
        ok = all(np.isfinite(metrics[k]) for k in ("best_fitness", "mean_fitness", "std_fitness"))
        record("C5", "EvaluationFramework, 2 runs", "PASS" if ok else "FAIL",
               f"keys {sorted(metrics)}; summarize() printed {len(out.getvalue().splitlines())} lines")
        vdir = TEMP / "visualizer"
        v = Visualizer(output_dir=str(vdir))
        with captured():
            made = [v.plot_fitness_curves(metrics["histories"], filename="fitness_test.png"),
                    v.plot_meta_convergence([3.0, 2.0, 1.5], filename="meta_test.png"),
                    v.plot_pareto_front([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]], filename="pareto_test.png"),
                    v.plot_algorithm_comparison([(cfg, metrics)], filename="comparison_test.png")]
        good = [m for m in made if m and Path(m).exists() and Path(m).stat().st_size > 0]
        record("C5", "Visualizer, 4 plots", "PASS" if len(good) == 4 else "FAIL",
               f"{len(good)} of 4 files written to the temporary folder")
    except Exception as exc:  # noqa: BLE001
        record("C5", "EvaluationFramework / Visualizer", "FAIL", f"{type(exc).__name__}: {exc}", traceback.format_exc())

    # experiment_runner.py, for real, on one tiny problem (decision D2)
    cwd = TEMP / "experiment_runner"
    cwd.mkdir(exist_ok=True)
    code = ("import time\nimport experiment_runner as er\nfrom aad import SphereProblem\n"
            "t = time.perf_counter()\n"
            "er.run_experiment(problems=[SphereProblem(dim=5)], meta_generations=1, "
            "meta_population_size=4, runs_per_eval=2, verbose=False)\n"
            "print('PHASE0_SECONDS', round(time.perf_counter() - t, 1))\n")
    r = run_py(code, cwd=cwd, timeout=1800)
    files = sorted(p.relative_to(cwd).as_posix() for p in cwd.rglob("*") if p.is_file())
    STATE["runner_files"] = files
    ok = r.rc == 0 and "results/experiment_results.json" in files
    record("C5", "experiment_runner.py, tiny (SphereProblem dim 5, 1 meta-generation, 4 configs)",
           "PASS" if ok else "FAIL",
           (f"{r.secs:.0f} s; files written: {files}" if r.rc == 0 else
            ("timed out" if r.timeout else last_line(r.err))), tail(r.err, 20))


# ---------------------------------------------------------------------------
# C6  known-defect register
# ---------------------------------------------------------------------------

def _src_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def _grep(path, pattern):
    return [i for i, ln in enumerate(_src_lines(path), 1) if re.search(pattern, ln)]


def c6_defects():
    import numpy as np
    import aad
    from aad import (EvaluationFramework, LLMDesigner, OptimizationProblem, ZDT1Problem,
                     EXTENDED_CONFIG_SPACE)
    from aad.config_space import config_to_vector
    from aad.evolutionary_algorithm import build_algorithm, EvolutionaryAlgorithm
    from aad.mealpy_algorithms import MealpyAlgorithm
    from aad.landscape_features import LandscapeFeatureExtractor as LFE

    def item(n, title, fn):
        try:
            status, detail, evidence = fn()
        except Exception as exc:  # noqa: BLE001
            status, detail, evidence = "ERROR", f"{type(exc).__name__}: {exc}", traceback.format_exc()
        record("C6", f"#{n} {title}", status, detail, evidence)

    def d1():
        op = operator_evidence()
        base = {"algorithm": "PSO", "population_size": 20, "selection": "tournament",
                "crossover": "sbx", "mutation": "gaussian", "max_generations": 100}
        differ = config_to_vector(base) != config_to_vector(dict(base, crossover="uniform"))
        masking = _grep(ROOT / "aad" / "meta_learner.py", r"selection|crossover|mutation")
        confirmed = op["mealpy_same"] and differ and not masking
        return ("CONFIRMED" if confirmed else "RULED OUT",
                f"PSO ignores the operator keys (identical results: {op['mealpy_same']}), yet two PSO "
                f"configurations that differ only in crossover encode differently: {differ}; "
                f"operator keys mentioned in meta_learner.py: {masking or 'nowhere'}", "")

    def d2():
        cost = feature_cost()
        per_dim = (cost[10] - cost[5]) / 5
        base = cost[5] - 5 * per_dim
        return ("CONFIRMED" if cost[5] > 300 else "RULED OUT",
                f"one extraction (300 samples) used {cost[5]} evaluations at dim 5 and {cost[10]} at dim 10, "
                f"i.e. {base:g} + {per_dim:g} x dimension",
                f"double evaluation in sample_landscape: aad/landscape_features.py lines "
                f"{_grep(ROOT / 'aad' / 'landscape_features.py', r'problem.evaluate')}")

    def d3():
        p = f1_sphere()
        c = Counter(p)
        seed_all(7)
        cfg = {"algorithm": "ea", "population_size": 20, "max_generations": 50, "max_fe": 2000,
               "selection": "tournament", "crossover": "sbx", "mutation": "gaussian"}
        build_algorithm(cfg).run(p)
        return ("CONFIRMED" if c.n != 2000 else "RULED OUT",
                f"ea with max_fe=2000, pop 20, 50 generations used {c.n} evaluations "
                f"(pop x (generations + 1) = {20 * 51})", "")

    def d4():
        class NaNProblem(OptimizationProblem):
            def evaluate(self, solution):
                return float("nan")

            def bounds(self):
                return [-1.0] * 3, [1.0] * 3

            def is_multiobjective(self):
                return False

        cfg = {"algorithm": "ea", "population_size": 6, "max_generations": 2,
               "selection": "tournament", "crossover": "sbx", "mutation": "gaussian"}
        with captured() as (out, err, caught):
            m = EvaluationFramework(n_runs=2).evaluate(cfg, NaNProblem())
        said = (out.getvalue() + err.getvalue()).strip()
        confirmed = m["best_fitness"] == 1e12 and not caught and not said
        return ("CONFIRMED" if confirmed else "RULED OUT",
                f"a problem that always returns NaN gives best_fitness={m['best_fitness']:g}, "
                f"warnings raised: {len(caught)}, text printed: {'yes' if said else 'no'}",
                f"aad/evaluation.py lines {_grep(ROOT / 'aad' / 'evaluation.py', r'_FITNESS_PENALTY|1e12')}; "
                f"aad/meta_learner.py lines {_grep(ROOT / 'aad' / 'meta_learner.py', r'_FITNESS_PENALTY|1e12')}")

    def d5():
        from aad.surrogate_model import SurrogateModel
        rng = np.random.default_rng(5)
        X, y, Xt = rng.random((200, 40)), rng.random(200), rng.random((50, 40))
        preds = []
        for _ in range(5):
            m = SurrogateModel(model_type="rf")           # default n_jobs=-1
            with captured():
                m.fit(X, y)
            preds.append(np.asarray(m.predict(Xt)))
        same = all(np.array_equal(preds[0], p) for p in preds[1:])
        spread = max(float(np.max(np.abs(preds[0] - p))) for p in preds[1:])
        return ("NOT REPRODUCED" if same else "CONFIRMED",
                f"5 fits with the default n_jobs=-1 on identical data ({os.cpu_count()} CPUs): "
                f"predictions bit-identical: {same}; largest difference {spread:.3g}", "")

    def d6():
        code = ("import json, os\nbefore = set(os.environ)\nimport aad\n"
                "print(json.dumps(sorted(set(os.environ) - before)))\n")
        r = run_py(code, env=sub_env(keep_secret_names_out=True), timeout=300)
        if r.rc != 0:
            return "ERROR", last_line(r.err), tail(r.err)
        added = json.loads(last_line(r.out))
        has_env = (ROOT / ".env").exists()
        return ("CONFIRMED" if added else ("RULED OUT" if has_env else "NOT CHECKED"),
                f"variable names added to the environment by 'import aad' (names only, values never read): "
                f"{added or 'none'}; .env present: {has_env}",
                f"aad/llm_designer.py lines {_grep(ROOT / 'aad' / 'llm_designer.py', r'^load_dotenv')}")

    def d7():
        from scipy.stats import skew
        f = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        fv = LFE.fitness_variance(f)
        is_cv = bool(np.isclose(fv, np.std(f) / abs(np.mean(f))))
        is_var = bool(np.isclose(fv, np.var(f)))
        mi = LFE.modality_indicator(f)
        is_skew = bool(np.isclose(mi, skew(f)) or np.isclose(mi, abs(skew(f))))
        params = list(inspect.signature(LFE.ruggedness).parameters)
        uses_points = any(p in params for p in ("samples", "points", "problem", "X"))
        found = sum([is_cv and not is_var, is_skew, not uses_points])
        return ("CONFIRMED" if found == 3 else ("PARTLY CONFIRMED" if found else "RULED OUT"),
                f"fitness_variance = std/|mean|: {is_cv} (= variance: {is_var}); modality = skewness: "
                f"{is_skew} (value {mi:.4g}, skew {skew(f):.4g}); ruggedness parameters {params}: "
                f"cannot see where points are: {not uses_points}", "")

    def d8():
        path = ROOT / "experiment_runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_experiment")
        seg = ast.get_source_segment(path.read_text(encoding="utf-8"), fn)
        args = [a.arg for a in fn.args.args]
        defaults = dict(zip(args[-len(fn.args.defaults):], [ast.literal_eval(d) for d in fn.args.defaults]))
        seeded = bool(re.search(r"\bseed\b", seg))
        budgeted = "max_fe" in seg
        llm_lines = [ln for ln in _grep(path, r"LLMDesigner\(")]
        llm_with_key = any("api_key" in _src_lines(path)[ln - 1] for ln in llm_lines)
        label_lines = _grep(path, r"[\"']llm_designer[\"']")
        confirmed = (not seeded) and (not budgeted) and defaults.get("runs_per_eval") == 2 and not llm_with_key
        return ("CONFIRMED" if confirmed else "PARTLY CONFIRMED",
                f"run_experiment: any seed: {seeded}; any max_fe (equal budget): {budgeted}; default "
                f"runs_per_eval={defaults.get('runs_per_eval')}; LLMDesigner built without a key at lines "
                f"{llm_lines} (so it is the rule-based designer) and stored as 'llm_designer' at lines {label_lines}",
                "")

    def d9():
        if "lists_ok" not in STATE:
            c4_lists = None
        else:
            c4_lists = STATE["lists_ok"]
        if c4_lists is None:
            from aad import config_space as cs, mealpy_algorithms as ma, llm_designer as ld
            c4_lists = (sorted(cs.MEALPY_ALGORITHM_KEYS) == sorted(ma.MEALPY_REGISTRY)
                        == sorted(k for k in ld.ALGORITHM_NAMES if k != "ea"))
        claim = _grep(ROOT / "user_model_runner.py", r"\b24\b")
        return ("RULED OUT" if c4_lists else "CONFIRMED",
                f"registry, config_space list and LLM names hold the same 24: {c4_lists}; "
                f"user_model_runner.py mentions 24 at lines {claim or 'none'}", "")

    def d10():
        des = LLMDesigner(api_key="phase0-fake-key-not-real")

        def fail():
            raise RuntimeError("simulated API failure")

        des._get_client = fail
        with no_network(), captured() as (out, err, caught):
            cfg = des.propose_algorithm("A 5-dimensional multimodal problem.")
        said = (out.getvalue() + err.getvalue()).strip()
        return ("CONFIRMED", f"with the API failing, propose_algorithm returned {cfg} without raising; "
                f"the only signal is printed text: {said!r}; warnings: {len(caught)}", "")

    def d11():
        rows = STATE.get("call_map")
        if rows is None:
            return "NOT CHECKED", "needs C3 (run without --only, or include C3)", ""
        unused = [r["qualname"] for r in rows if r["status"] == "unused"]
        named = {"EvaluationFramework.compare_configs", "LLMDesigner.describe_for_selection"}
        return ("CONFIRMED" if named <= set(unused) else "PARTLY CONFIRMED",
                f"unused by pipeline, scripts and tests: {unused or 'none'}", "")

    def d12():
        z = ZDT1Problem(dim=5)
        x = np.full(5, 0.3)
        v = z.evaluate(x)
        s1 = MealpyAlgorithm._scalar(v)
        s2 = EvolutionaryAlgorithm._scalar(v)
        same = np.isclose(s1, sum(v)) and np.isclose(s2, sum(v))
        return ("CONFIRMED" if same else "RULED OUT",
                f"ZDT1 returns {len(v)} objectives; both algorithm families optimise their sum "
                f"({s1:.4g} = {v[0]:.4g} + {v[1]:.4g})", "")

    def d13():
        path = ROOT / "experiment_runner.py"
        rnd = _grep(path, r"np\.random\.random\(\(200")
        plot = _grep(path, r"plot_pareto_front")
        return ("CONFIRMED" if rnd and plot else "RULED OUT",
                f"200 random points drawn at lines {rnd}, plotted by plot_pareto_front at lines {plot}", "")

    def d14():
        mem = STATE.get("designer_mem")
        if mem is None:
            from aad import EA_Designer
            space = dict(EXTENDED_CONFIG_SPACE, population_size=[20], max_generations=[20])
            seed_all(1)
            d = EA_Designer(config_space=space, population_size=4, generations=1, runs_per_eval=1)
            with captured():
                d.evolve(f1_sphere(), verbose=False)
            arr = np.array(d._mem_X, dtype=float)
            mem = {"rows": int(arr.shape[0]), "feature_cols_constant": bool(np.all(arr[:, :9] == arr[0, :9]))}
        path = ROOT / "experiment_runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        inside_loop = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "EA_Designer":
                        inside_loop.append(sub.lineno)
        confirmed = mem["feature_cols_constant"] and bool(inside_loop)
        return ("CONFIRMED" if confirmed else "PARTLY CONFIRMED",
                f"the 9 feature columns are identical in all {mem['rows']} rows of the surrogate memory: "
                f"{mem['feature_cols_constant']}; experiment_runner.py builds a new EA_Designer inside the "
                f"per-problem loop at lines {inside_loop}", "")

    def d15():
        out = {}
        for a in ("CMA_ES", "AOA", "DE", "PSO"):
            res = []
            for fe in (2000, 20000):
                seed_all(7)
                with captured():
                    _, fit = build_algorithm({"algorithm": a, "population_size": 20,
                                              "max_fe": fe, "seed": 7}).run(f1_sphere())
                res.append(fit)
            out[a] = res
        stuck = [a for a in ("CMA_ES", "AOA", "DE") if out[a][1] > out[a][0] / 10]
        text = "; ".join(f"{a}: {v[0]:.3g} at 2,000 -> {v[1]:.3g} at 20,000" for a, v in out.items())
        status = "CONFIRMED" if len(stuck) == 3 else ("PARTLY CONFIRMED" if stuck else "RULED OUT")
        return (status, f"BBOB F1 dim 5, pop 20, seed 7 (PSO as a working control). {text}. "
                f"Improved less than 10x with 10x the budget: {stuck or 'none'}", "")

    def d16():
        with captured() as (_o, _e, caught):
            ZDT1Problem(dim=5).evaluate(np.array([-0.5, 0.2, 0.2, 0.2, 0.2]))
        direct = [str(w.message) for w in caught]
        run = STATE.get("pytest")
        if run is None:
            return ("NOT CHECKED", f"needs the pytest run (C3 or C7). Direct check: ZDT1 at x1 = -0.5 "
                    f"(outside its [0, 1] bounds) warns: {direct or 'nothing'}", "")
        hits = [w for w in run["warnings"] if w["category"] == "RuntimeWarning"]
        oob = [r["algorithm"] for r in STATE.get("c5_rows", []) if r["out_of_bounds_evals"]]
        detail = (f"RuntimeWarnings in this test run: "
                  + ("; ".join(f"{w['nodeid']} ({w['file']}:{w['line']}: {w['message']})" for w in hits)
                     if hits else "none")
                  + f". ZDT1 evaluated at x1 = -0.5, outside its bounds, warns: {direct or 'nothing'}. "
                  + f"Algorithms that evaluated points outside the bounds in C5: {oob or 'none (or C5 not run)'}")
        return ("CONFIRMED" if hits else "NOT REPRODUCED", detail, "")

    defects = [
        (1, "operator settings not masked", d1),
        (2, "feature points evaluated twice", d2),
        (3, "ea ignores max_fe", d3),
        (4, "failed runs replaced by 1e12", d4),
        (5, "surrogate n_jobs=-1 not repeatable", d5),
        (6, "importing aad loads .env", d6),
        (7, "three feature names mislead", d7),
        (8, "experiment_runner.py: budgets, seeds, runs, LLM label", d8),
        (9, "algorithm lists hold the same 24", d9),
        (10, "LLM failure silently becomes a rule-based pick", d10),
        (11, "unused methods", d11),
        (12, "ZDT1/ZDT2 optimised as f1 + f2", d12),
        (13, "pareto plot shows 200 random points", d13),
        (14, "EA_Designer's surrogate learns inside one problem only", d14),
        (15, "CMA_ES, AOA, DE do not optimise", d15),
        (16, "ZDT1 square-root warning", d16),
    ]
    for n, title, fn in defects:
        item(n, title, fn)


# ---------------------------------------------------------------------------
# C7  test suites
# ---------------------------------------------------------------------------

def c7_tests():
    run = pytest_run()
    ok = run["rc"] == 0 and not run["failed"]
    record("C7", "pytest tests/", "PASS" if ok else "FAIL",
           f"{run['passed']} passed, {len(run['failed'])} failed, {len(run['skipped'])} skipped, "
           f"{len(run['warnings'])} warnings; {run['secs']:.0f} s (under the profiler used by C3)"
           + (f"; failed: {run['failed']}" if run["failed"] else ""), tail(run["output"], 15))
    r = run_py(f"import runpy, sys\nsys.argv = ['test_phase0_config_consistency.py']\n"
               f"runpy.run_path({str(ROOT / 'test_phase0_config_consistency.py')!r}, run_name='__main__')",
               timeout=900)
    summary = next((ln.strip() for ln in r.out.splitlines() if "SUMMARY:" in ln), last_line(r.err))
    record("C7", "test_phase0_config_consistency.py", "PASS" if r.rc == 0 else "FAIL",
           f"{summary}; {r.secs:.0f} s", tail(r.out, 15))


# ---------------------------------------------------------------------------
# C8  README check
# ---------------------------------------------------------------------------

README_NAMES = ["EA_Designer", "SphereProblem", "EvaluationFramework", "OptimizationProblem",
                "RastriginProblem", "RosenbrockProblem", "AckleyProblem", "ZDT1Problem", "ZDT2Problem",
                "LLMDesigner", "SurrogateModel", "EA_CONFIG_SPACE"]
README_METHODS = {"EA_Designer": ["evolve"], "EvaluationFramework": ["evaluate", "summarize"],
                  "LLMDesigner": ["propose_algorithm", "generate_description"],
                  "SurrogateModel": ["fit", "predict", "predict_with_uncertainty"]}
README_PARAMS = {"EA_Designer": ["population_size", "generations", "runs_per_eval", "use_surrogate"],
                 "EvaluationFramework": ["n_runs"], "LLMDesigner": ["api_key", "model"],
                 "SurrogateModel": ["model_type"], "SphereProblem": ["dim"]}
README_METRICS = ["best_fitness", "mean_fitness", "std_fitness", "convergence_gen", "stability", "mean_runtime"]

# Only the names a README example uses without defining them; nothing else is changed.
README_CONTEXT = {
    "custom problem": "from aad import OptimizationProblem\n",
    "EA_Designer": "from aad import SphereProblem\nproblem = SphereProblem(dim=10)\n",
    "LLM designer": "from aad import SphereProblem\nproblem = SphereProblem(dim=10)\n",
    "surrogate": ("import numpy as np\nrng = np.random.default_rng(0)\n"
                  "X_train = rng.random((30, 5))\ny_train = rng.random(30)\nX_test = rng.random((5, 5))\n"),
}


def _label_block(code):
    if "class MyProblem" in code:
        return "custom problem"
    if "EA_CONFIG_SPACE = {" in code:
        return "config space"
    if "LLMDesigner" in code:
        return "LLM designer"
    if "SurrogateModel" in code:
        return "surrogate"
    if "evolve(problem, verbose=True)" in code and "SphereProblem(dim=10)" in code:
        return "minimal example"
    if "EA_Designer(" in code:
        return "EA_Designer"
    return code.strip().splitlines()[0][:40]


def c8_readme():
    import numpy as np
    import aad
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # Names, methods and parameters the README uses
    not_in_readme = [n for n in README_NAMES if n not in readme]
    missing = [n for n in README_NAMES if not hasattr(aad, n)]
    record("C8", "README names exist in aad", "PASS" if not missing else "FAIL",
           f"{len(README_NAMES) - len(missing)} of {len(README_NAMES)}; missing: {missing or 'none'}"
           + (f"; (list check: not found in README: {not_in_readme})" if not_in_readme else ""))
    bad = []
    for cls, methods in README_METHODS.items():
        bad += [f"{cls}.{m}" for m in methods if not hasattr(getattr(aad, cls), m)]
    for cls, params in README_PARAMS.items():
        sig = inspect.signature(getattr(aad, cls))
        bad += [f"{cls}({p}=...)" for p in params if p not in sig.parameters]
    with captured():
        has_mc = hasattr(aad.EA_Designer(), "meta_convergence")
    if not has_mc:
        bad.append("EA_Designer().meta_convergence")
    record("C8", "README methods, parameters, meta_convergence", "PASS" if not bad else "FAIL",
           f"missing: {bad or 'none'}")

    # Code examples, run as written (decision D5), each in a fresh process in the temp folder
    blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.S)
    examples = []
    for i, code in enumerate(blocks, 1):
        label = _label_block(code)
        cwd = TEMP / f"readme_{i}"
        cwd.mkdir(exist_ok=True)
        prelude = NO_NET_PRELUDE if label == "LLM designer" else ""
        heavy = label in ("minimal example", "EA_Designer")
        r = run_py(prelude + code, cwd=cwd, timeout=2400 if heavy else 600)
        entry = {"block": i, "label": label, "as_written": "ok" if r.rc == 0 else
                 ("timeout" if r.timeout else last_line(r.err)), "secs": round(r.secs, 1)}
        status = "PASS" if r.rc == 0 else "FAIL"
        detail = f"as written: {entry['as_written']} ({r.secs:.0f} s)"
        evidence = tail(r.out, 12)
        if r.rc != 0 and label in README_CONTEXT and "NameError" in r.err:
            r2 = run_py(prelude + README_CONTEXT[label] + code, cwd=cwd, timeout=2400 if heavy else 600)
            entry["with_missing_names_supplied"] = "ok" if r2.rc == 0 else (
                "timeout" if r2.timeout else last_line(r2.err))
            entry["secs_supplied"] = round(r2.secs, 1)
            status = "WARN" if r2.rc == 0 else "FAIL"
            detail += (f"; with only the undefined names supplied: {entry['with_missing_names_supplied']} "
                       f"({r2.secs:.0f} s)")
            evidence = tail(r2.out, 12) or tail(r2.err, 12)
        if label == "LLM designer":
            detail += "; network switched off for this block"
        examples.append(entry)
        record("C8", f"README example {i}: {label}", status, detail, evidence)
    record("C8", "README Quick Start 'python experiment_runner.py'", "INFO",
           "not run as written: it runs the full default experiment; C5 runs it for real on one tiny problem (D2)")
    STATE["readme_examples"] = examples

    # A filled-in custom problem through the main components (README: 'Custom problem plug-in')
    code = ("from aad import OptimizationProblem, EvaluationFramework, LandscapeFeatureExtractor\n"
            "class MyProblem(OptimizationProblem):\n"
            "    def evaluate(self, solution):\n        return float(sum(x * x for x in solution))\n"
            "    def bounds(self):\n        return [-5.0] * 3, [5.0] * 3\n"
            "    def is_multiobjective(self):\n        return False\n"
            "p = MyProblem()\n"
            "m = EvaluationFramework(n_runs=1).evaluate({'algorithm': 'PSO', 'population_size': 10, "
            "'max_fe': 300, 'seed': 1}, p)\n"
            "f = LandscapeFeatureExtractor().extract_features(p)\n"
            "print(p.name, round(m['best_fitness'], 6), len(f))\n")
    r = run_py(code, cwd=TEMP, timeout=300)
    record("C8", "custom problem with only the 3 README methods", "PASS" if r.rc == 0 else "FAIL",
           (f"runs through EvaluationFramework and the feature extractor: {last_line(r.out)}"
            if r.rc == 0 else last_line(r.err)))

    # README claims against the code
    claims = []
    listed = set(re.findall(r"([a-z_]+\.py)\s+#", readme))
    actual = {p.name for p in (ROOT / "aad").glob("*.py")}
    claims.append(("repository tree lists every aad module",
                   not (actual - listed), f"not in the README tree: {sorted(actual - listed) or 'none'}"))
    m = re.search(r"EA_CONFIG_SPACE = (\{.*?\n\})", readme, flags=re.S)
    readme_space = ast.literal_eval(m.group(1)) if m else None
    claims.append(("EA_CONFIG_SPACE equals the README's", readme_space == aad.EA_CONFIG_SPACE,
                   "identical" if readme_space == aad.EA_CONFIG_SPACE else f"README {readme_space}"))
    with captured():
        default_space = aad.EA_Designer().config_space
    claims.append(("EA_Designer searches the README's EA_CONFIG_SPACE by default",
                   default_space == aad.EA_CONFIG_SPACE,
                   f"default space has {len(default_space.get('algorithm', ['ea']))} algorithms "
                   f"(keys {list(default_space)})"))
    keys = STATE.get("metrics_keys")
    if keys is None:
        with captured():
            keys = sorted(aad.EvaluationFramework(n_runs=1).evaluate(
                {"algorithm": "PSO", "population_size": 10, "max_fe": 200, "seed": 1}, f1_sphere()))
    claims.append(("result keys match the README's sample output", set(README_METRICS) <= set(keys),
                   f"README shows {README_METRICS}; code returns {keys}"))
    llm_src = (ROOT / "aad" / "llm_designer.py").read_text(encoding="utf-8")
    uses_openai = bool(re.search(r"^\s*(import openai|from openai)", llm_src, flags=re.M))
    uses_gemini = "google.genai" in llm_src or "from google import genai" in llm_src
    claims.append(("LLM mode uses OpenAI, as the README says", uses_openai and not uses_gemini,
                   f"imports openai: {uses_openai}; uses Google Gemini (google.genai): {uses_gemini}"))
    runner = (ROOT / "experiment_runner.py").read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, ln in enumerate(runner) if re.search(r"if problems is None", ln))
        block = []
        for ln in runner[start + 1:]:
            block.append(ln)
            if re.match(r"\s*\]\s*$", ln):
                break
        active = [ln for ln in block if re.search(r"Problem\(|BBOB_F\d+", ln) and not ln.strip().startswith("#")]
        claims.append(("Quick Start runs 5 benchmark problems", len(active) == 5,
                       f"the default list in run_experiment has {len(active)} active entries "
                       f"(read from the source, lines {start + 2} on)"))
    except StopIteration:
        claims.append(("Quick Start runs 5 benchmark problems", False, "default problem list not found"))
    files = STATE.get("runner_files")
    if files is not None:
        want = ["results/experiment_results.json", "results/fitness_SphereProblem.png",
                "results/meta_SphereProblem.png"]
        claims.append(("output files named as in the README", all(w in files for w in want),
                       f"expected {want}; the tiny run wrote {files}"))
    claims.append(("pareto_ZDT1Problem.png shows the EA's result", not _grep(
        ROOT / "experiment_runner.py", r"np\.random\.random\(\(200"),
        "the runner plots 200 random points (see C6 #13)"))
    claims.append(("ZDT1/ZDT2 are optimised as multi-objective problems", False,
                   "both algorithm families optimise f1 + f2 (see C6 #12)"))
    with captured():
        d = aad.EA_Designer()
    claims.append(("surrogate screening on by default", d.use_surrogate is True, f"use_surrogate={d.use_surrogate}"))
    sig = inspect.signature(aad.SurrogateModel)
    claims.append(("parallel surrogate, n_jobs=-1", sig.parameters["n_jobs"].default == -1,
                   f"default n_jobs={sig.parameters['n_jobs'].default} (repeatability: C6 #5)"))
    abstract = sorted(getattr(aad.OptimizationProblem, "__abstractmethods__", set()))
    claims.append(("custom problems need only three methods",
                   abstract == ["bounds", "evaluate", "is_multiobjective"], f"abstract methods: {abstract}"))
    readme_deps = {"numpy", "scikit-learn", "matplotlib", "openai"}
    third = sorted(r["item"].split("'")[1] for r in RESULTS if r["check"] == "C2" and r["item"].startswith("import '"))
    if third:
        claims.append(("README dependency list covers the imports",
                       not (set(third) - {"numpy", "sklearn", "matplotlib", "openai"}),
                       f"README lists {sorted(readme_deps)}; the code imports {third}"))
    for title, ok, evidence in claims:
        record("C8", f"README claim: {title}", "MATCH" if ok else "DIFFERS", evidence)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(selected, started, total_secs):
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": started.isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "executable": rel(sys.executable),
        "checks_run": selected, "seconds_total": round(total_secs, 1),
        "seconds_per_check": {k: round(v, 1) for k, v in TIMES.items()},
        "results": RESULTS,
        "tiny_runs": STATE.get("c5_rows"),
        "fingerprint": STATE.get("fingerprint"),
        "readme_examples": STATE.get("readme_examples"),
    }
    (OUT / "health_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = ["Phase 0 health report", "=" * 72,
             f"Generated {payload['generated']} with Python {payload['python']} ({payload['executable']})",
             f"Checks run: {', '.join(selected)}; total {total_secs / 60:.1f} min",
             "Plan: docs/planning/phase0/phase0_plan.md. This script changed no code.", ""]
    by_check = collections.defaultdict(list)
    for r in RESULTS:
        by_check[r["check"]].append(r)
    lines.append("Summary")
    for chk in selected:
        counts = collections.Counter(r["status"] for r in by_check[chk])
        lines.append(f"  {chk}  {TIMES.get(chk, 0):6.0f} s  " +
                     ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"  -- {CHECKS[chk]}")
    attention = [r for r in RESULTS if r["status"] in ("FAIL", "ERROR")]
    lines += ["", f"Needs attention (FAIL or ERROR): {len(attention)}"]
    lines += [f"  {r['check']} {r['item']}: {r['detail'][:200]}" for r in attention]
    for chk in selected:
        lines += ["", f"{chk}  {CHECKS[chk]}", "-" * 72]
        for r in by_check[chk]:
            lines.append(f"[{r['status']}] {r['item']}")
            if r["detail"]:
                lines.append(f"    {r['detail']}")
            if r["evidence"] and r["status"] not in ("PASS", "INFO"):
                lines += ["    | " + ln for ln in r["evidence"][:1500].splitlines()]
    (OUT / "health_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global TEMP
    ap = argparse.ArgumentParser(description="Phase 0 health check (reports only; changes no code).")
    ap.add_argument("--only", help="comma-separated checks, e.g. C1,C4")
    ap.add_argument("--keep-temp", action="store_true", help="keep the temporary folder")
    ap.add_argument("--list", action="store_true", help="describe the checks and exit")
    args = ap.parse_args()
    if args.list:
        for k, v in CHECKS.items():
            print(f"{k}  {v}")
        return 0
    selected = [c.strip().upper() for c in args.only.split(",")] if args.only else list(CHECKS)
    unknown = [c for c in selected if c not in CHECKS]
    if unknown:
        ap.error(f"unknown checks: {unknown}")
    os.chdir(ROOT)
    TEMP = Path(tempfile.mkdtemp(prefix="phase0_"))
    started = datetime.datetime.now()
    t_all = time.perf_counter()
    print(f"Phase 0 health check; temporary folder {TEMP}")
    functions = {"C1": c1_imports, "C2": c2_dependencies, "C3": c3_call_map, "C4": c4_fit,
                 "C5": c5_tiny_runs, "C6": c6_defects, "C7": c7_tests, "C8": c8_readme}
    try:
        for chk in selected:
            print(f"\n{chk}  {CHECKS[chk]}", flush=True)
            t0 = time.perf_counter()
            try:
                functions[chk]()
            except Exception as exc:  # noqa: BLE001
                record(chk, "check crashed", "ERROR", f"{type(exc).__name__}: {exc}", traceback.format_exc())
            TIMES[chk] = time.perf_counter() - t0
    finally:
        total = time.perf_counter() - t_all
        write_report(selected, started, total)
        if args.keep_temp:
            print(f"\nTemporary folder kept: {TEMP}")
        else:
            shutil.rmtree(TEMP, ignore_errors=True)
    bad = sum(1 for r in RESULTS if r["status"] in ("FAIL", "ERROR"))
    print(f"\nDone in {total / 60:.1f} min. {bad} FAIL/ERROR. Report: {rel(OUT / 'health_report.txt')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
