# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Private subprocess entry point; visibility is set before interpreter startup."""
import os
import sys
import traceback
from multiprocessing import shared_memory, resource_tracker
from multiprocessing.connection import Connection

import numpy as np
import torch

from .decoder import RelayBPDecoder
from .process import validate_payload


def main():
    conn = Connection(int(sys.argv[1]))
    try:
        torch.set_num_threads(max(1, int(os.environ.get("PELAGOS_WORKER_THREADS", "2"))))
        ctor = conn.recv()
        decoder = RelayBPDecoder(**ctor, device=[0])
        conn.send(("ready", {"wj": decoder.wj.detach().cpu().numpy()}))
        while True:
            syndromes, spec = conn.recv()
            output = decoder.decode(torch.from_numpy(syndromes))
            torch.cuda.synchronize()
            payload = {}
            for key, value in output.items():
                if key == "errors" and spec is not None:
                    name, shape, start, end = spec
                    shared = shared_memory.SharedMemory(name=name)
                    # A subprocess has its own resource tracker. The parent alone
                    # owns unlink; otherwise an exiting worker unlinks live output.
                    resource_tracker.unregister(shared._name, "shared_memory")
                    try:
                        target = np.ndarray(shape, dtype=np.uint8, buffer=shared.buf)[start:end]
                        if tuple(value.shape) != target.shape or value.dtype != torch.uint8:
                            raise ValueError("shared error output shape/dtype mismatch")
                        torch.from_numpy(target).copy_(value.detach())
                    finally:
                        shared.close()
                else:
                    payload[key] = value.detach().cpu().contiguous().numpy()
            validate_payload(payload)
            conn.send(("ok", payload))
    except EOFError:
        pass
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
