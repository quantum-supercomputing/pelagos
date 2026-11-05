# (C) Copyright IBM 2025, 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Triton GPU backend for relay bp library.

This module provides GPU-accelerated implementations of the relay BP algorithm
using Triton kernels, with interfaces compatible with the Rust implementation.
"""

# Import the core Triton decoder
from .decoder import RelayBPDecoder

# Import adapter classes that provide Rust-compatible interfaces
from .adapter import RelayDecoder, ObservableDecoderRunner

# Re-export with Rust-compatible names for seamless integration
# These provide the same interface as the Rust RelayDecoderF32/F64 classes
RelayDecoderF32 = RelayDecoder
RelayDecoderF64 = RelayDecoder  # Same implementation, dtype controlled by constructor

# For compatibility with existing code that expects these names
__all__ = [
    "RelayBPDecoder",  # Core Triton implementation
    "RelayDecoder",  # Adapter with Rust-compatible interface
    "ObservableDecoderRunner",  # Observable decoder adapter
    "RelayDecoderF32",  # Rust-compatible name for fp32
    "RelayDecoderF64",  # Rust-compatible name for fp64
]
