import importlib.util
import sys
import traceback


def _load(file: str) -> None:
    spec = importlib.util.spec_from_file_location('x', file)
    if spec is None or spec.loader is None:
        msg = f'Cannot load {file}'
        raise ImportError(msg)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


if __name__ == '__main__':
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            _load(file)
        except Exception:
            has_failure = True
            print(file)
            traceback.print_exc()
            print()

    sys.exit(1 if has_failure else 0)
