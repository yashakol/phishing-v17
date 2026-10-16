"""Make the macOS LightGBM wheel use sklearn's existing OpenMP runtime.

Only a library inside the active Python environment is modified. Linux needs no
patch. The original library is retained for rollback.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    check = subprocess.run([sys.executable, '-c', 'import lightgbm'], capture_output=True, text=True)
    if check.returncode == 0:
        print('LightGBM imports successfully; no patch required.')
        return
    if sys.platform != 'darwin' or '@rpath/libomp.dylib' not in check.stderr:
        raise RuntimeError(check.stderr)
    lightgbm = importlib.util.find_spec('lightgbm')
    sklearn = importlib.util.find_spec('sklearn')
    if lightgbm is None or sklearn is None:
        raise RuntimeError('Install the pinned requirements first.')
    library = Path(lightgbm.origin).parent / 'lib/lib_lightgbm.dylib'
    runtime = Path(sklearn.origin).parent / '.dylibs/libomp.dylib'
    if not runtime.is_file():
        raise RuntimeError('The installed sklearn wheel does not contain libomp.')
    if library.parent.parent.parent != runtime.parent.parent.parent:
        raise RuntimeError('LightGBM and sklearn must belong to the same Python environment.')
    backup = library.with_suffix('.dylib.original')
    if not backup.exists():
        shutil.copy2(library, backup)
    try:
        subprocess.run(['/usr/bin/install_name_tool', '-change', '@rpath/libomp.dylib',
                        '@loader_path/../../sklearn/.dylibs/libomp.dylib', str(library)], check=True)
        subprocess.run(['/usr/bin/codesign', '--force', '--sign', '-', str(library)], check=True)
        subprocess.run([sys.executable, '-c', 'import lightgbm; print(lightgbm.__version__)'], check=True)
    except BaseException:
        shutil.copy2(backup, library)
        raise
    print('Verified LightGBM with the existing sklearn OpenMP runtime.')


if __name__ == '__main__':
    main()
