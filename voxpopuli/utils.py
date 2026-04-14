# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional

from tqdm.contrib.concurrent import process_map


def multiprocess_run(
        a_list: list, func: callable, n_workers: Optional[int] = None
):
    if os.name == "nt":
        # On Windows, ProcessPoolExecutor uses "spawn" — each worker starts a
        # fresh Python interpreter and imports all modules from disk.  For
        # VoxPopuli's cut_session (load full session file, cut segments, save)
        # this causes OOM because workers can't GC fast enough between items.
        # Run sequentially in the main process instead.
        from tqdm import tqdm
        for item in tqdm(a_list):
            func(item)
    else:
        process_map(func, a_list, max_workers=n_workers, chunksize=1)
