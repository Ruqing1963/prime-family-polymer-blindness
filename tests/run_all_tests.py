#!/usr/bin/env python3
"""Run every test_* function in tests/ without requiring pytest; exit code 1 on any failure.
(pytest also works: `pytest tests/`.)"""
import os, sys, time, importlib, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    failed, passed = [], 0
    for mod_name in ("test_sieve", "test_ideal_chain"):
        mod = importlib.import_module(mod_name)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            t0 = time.time()
            try:
                getattr(mod, name)(); passed += 1
                print(f"  PASS  {mod_name}.{name}  ({time.time()-t0:.1f}s)", flush=True)
            except Exception:
                failed.append(f"{mod_name}.{name}")
                print(f"  FAIL  {mod_name}.{name}\n{traceback.format_exc()}", flush=True)
    print(f"\n{passed} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
