import platform
import subprocess

import numpy as np
import numpy._core._multiarray_umath as m

print('machine', platform.machine(), platform.processor())
print('cpu_baseline', m.__cpu_baseline__)
print('cpu_dispatch', m.__cpu_dispatch__)
print('cpu_features_on', sorted(k for k, v in m.__cpu_features__.items() if v))
np.show_runtime()
np.show_config()
try:
    from threadpoolctl import threadpool_info

    for d in threadpool_info():
        print('threadpool', d)
except Exception as exc:
    print('threadpoolctl failed', exc)
print(subprocess.run(['lscpu'], capture_output=True, text=True).stdout)
