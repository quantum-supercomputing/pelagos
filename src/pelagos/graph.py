# (C) Copyright IBM 2025, 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Graph utilities for converting SciPy CSR matrices to device-optimized edge-centric representation.

This module provides utilities for converting quantum error correction check matrices
from SciPy sparse format to GPU-optimized representations for belief propagation.
The CSRGraph class builds edge-centric data structures that enable efficient
message passing on GPU hardware.

The implementation follows the sparse graph representation used in the Rust
implementation in crates/relay_bp/src/bipartite_graph.rs.
"""

import torch
import numpy as np
import math
from scipy.sparse import csr_matrix
from typing import Tuple, Optional

from scipy.sparse import issparse


class CSRGraph:
    """GPU-optimized representation of quantum error correction check matrix.

    This class converts a SciPy CSR matrix representing quantum stabilizer generators
    into GPU-optimized edge-centric data structures for belief propagation.

    The representation includes:
    - Check-centric view: chk_ptr, chk_edges for check-to-variable message passing
    - Variable-centric view: var_ptr, var_edges for variable-to-check message passing
    - Edge mappings: edge_var, edge_chk for edge-to-node lookups

    This follows the sparse bipartite graph representation in Rust bipartite_graph.rs.
    """

    def __init__(
        self,
        H_csr: csr_matrix,
        device: str = "cuda",
        check_permutation: Optional[np.ndarray] = None,
    ):
        """Initialize CSRGraph from quantum check matrix.

        Args:
            H_csr: Check matrix (C x V) in CSR format representing stabilizer generators
            device: Target GPU device ("cuda" or "rocm")
            check_permutation: Optional permutation of check rows [C] int32 array. If provided,
                reorders CSR rows before building graph structures (used for graph-aware
                block definitions in layered scheduling).
        """
        if device == "cpu":
            raise RuntimeError(
                "CPU not supported - Triton kernels require GPU. Use device='cuda' or 'rocm'"
            )

        # Accept any scipy sparse, ensure CSR matrix
        if not issparse(H_csr):
            raise ValueError("Input must be a scipy.sparse matrix")
        if not isinstance(H_csr, csr_matrix):
            H_csr = H_csr.tocsr()

        self.device = device
        # Store shape before permutation (for reference)
        self.C, self.V = H_csr.shape
        self.E = H_csr.nnz

        # Validate input
        if not isinstance(H_csr, csr_matrix):
            raise ValueError("Input must be a scipy.sparse.csr_matrix")

        if H_csr.dtype != np.uint8 and H_csr.dtype != bool:
            # Convert to binary if needed
            H_csr = (H_csr != 0).astype(np.uint8)

        # Apply check permutation if provided (reorder CSR rows)
        if check_permutation is not None:
            check_permutation = np.asarray(check_permutation, dtype=np.int32)
            if check_permutation.shape[0] != H_csr.shape[0]:
                raise ValueError(
                    f"check_permutation length ({check_permutation.shape[0]}) must match number of checks ({H_csr.shape[0]})"
                )
            if not (check_permutation.min() >= 0 and check_permutation.max() < H_csr.shape[0]):
                raise ValueError(
                    f"check_permutation indices must be in range [0, {H_csr.shape[0]})"
                )
            # Reorder CSR matrix rows
            H_csr = H_csr[check_permutation, :]

        # Convert to device tensors
        self._build_edge_representation(H_csr)

    def _build_edge_representation(self, H_csr: csr_matrix):
        """Build edge-centric representation from CSR matrix.

        This method constructs the GPU-optimized data structures needed for
        belief propagation message passing, following the sparse graph format
        used in the Rust implementation.

        Args:
            H_csr: Check matrix in CSR format
        """
        # Get CSR data
        chk_ptr = H_csr.indptr.astype(np.int32)
        chk_edges = H_csr.indices.astype(np.int32)  # These are variable indices

        # Build variable-centric representation
        var_ptr = np.zeros(self.V + 1, dtype=np.int32)
        var_edges = np.zeros(self.E, dtype=np.int32)
        edge_var = np.zeros(self.E, dtype=np.int32)
        edge_chk = np.zeros(self.E, dtype=np.int32)

        # Build edge list and variable-centric pointers
        edge_idx = 0
        for chk_idx in range(self.C):
            start = chk_ptr[chk_idx]
            end = chk_ptr[chk_idx + 1]

            for local_idx in range(start, end):
                var_idx = chk_edges[local_idx]

                # Add to variable's edge list
                var_ptr[var_idx + 1] += 1

                # Store edge information
                edge_var[edge_idx] = var_idx
                edge_chk[edge_idx] = chk_idx
                var_edges[edge_idx] = edge_idx  # Will be updated after sorting

                edge_idx += 1

        # Convert to cumulative pointers for variables
        for i in range(1, self.V + 1):
            var_ptr[i] += var_ptr[i - 1]

        # Sort edges by variable index and update var_edges
        edge_order = np.argsort(edge_var)
        for i, edge_idx in enumerate(edge_order):
            var_idx = edge_var[edge_idx]
            var_edges[var_ptr[var_idx]] = edge_idx
            var_ptr[var_idx] += 1

        # Reset var_ptr to start positions
        for i in range(self.V - 1, 0, -1):
            var_ptr[i] = var_ptr[i - 1]
        var_ptr[0] = 0

        # Now build chk_edges as edge IDs (not variable indices)
        chk_edges_edge_ids = np.zeros(self.E, dtype=np.int32)
        edge_idx = 0
        for chk_idx in range(self.C):
            start = chk_ptr[chk_idx]
            end = chk_ptr[chk_idx + 1]

            for local_idx in range(start, end):
                chk_edges_edge_ids[local_idx] = edge_idx
                edge_idx += 1

        # Move to device
        self.chk_ptr = torch.from_numpy(chk_ptr).to(self.device)
        self.chk_edges = torch.from_numpy(chk_edges_edge_ids).to(self.device)  # Edge IDs
        self.var_ptr = torch.from_numpy(var_ptr).to(self.device)
        self.var_edges = torch.from_numpy(var_edges).to(self.device)
        self.edge_var = torch.from_numpy(edge_var).to(self.device)
        self.edge_chk = torch.from_numpy(edge_chk).to(self.device)

        # Compute degree statistics for autotuning
        self.max_deg_chk = int((chk_ptr[1:] - chk_ptr[:-1]).max())
        self.max_deg_var = int((var_ptr[1:] - var_ptr[:-1]).max())

        # Compute sparsity pattern hash for autotuning (captures connectivity, not just dimensions)
        # This ensures different H matrices with same (C, V, E) get different autotuned configs
        # The hash captures memory access patterns that affect kernel performance (degree distribution,
        # irregularity, locality) which influence optimal BLOCK_SIZE, BTILE, and ROWS_PER_CHK values
        import hashlib

        # Hash the actual edge structure: degree distribution + sorted edge list per check
        chk_degrees = (chk_ptr[1:] - chk_ptr[:-1]).astype(np.int32)
        var_degrees = (var_ptr[1:] - var_ptr[:-1]).astype(np.int32)
        # Build sorted edge list per check (for consistent hashing regardless of input CSR column order)
        # Sort variable indices within each check to get canonical representation
        sorted_edges_per_check = []
        for chk_idx in range(self.C):
            start = chk_ptr[chk_idx]
            end = chk_ptr[chk_idx + 1]
            if end > start:
                # Get variable indices for this check from original CSR and sort them
                var_indices = chk_edges[start:end]  # These are variable indices from CSR
                sorted_edges_per_check.append(np.sort(var_indices))
            else:
                sorted_edges_per_check.append(np.array([], dtype=np.int32))
        # Combine degree distributions and sorted edge lists into hash
        # Sort degree lists for consistency (order doesn't matter for degree distribution)
        pattern_blob = np.concatenate(
            [
                np.sort(chk_degrees).astype(np.int32),  # Check degree distribution (sorted)
                np.sort(var_degrees).astype(np.int32),  # Variable degree distribution (sorted)
                np.concatenate([np.sort(e) for e in sorted_edges_per_check]).astype(
                    np.int32
                ),  # All edges (sorted per check)
            ]
        ).tobytes()
        self.sparsity_hash = hashlib.blake2b(pattern_blob, digest_size=16).hexdigest()

    def get_device_tensors(self) -> Tuple[torch.Tensor, ...]:
        """Get all device tensors for kernel calls.

        Returns:
            Tuple of (chk_ptr, chk_edges, var_ptr, var_edges, edge_var, edge_chk)
        """
        return (
            self.chk_ptr,
            self.chk_edges,
            self.var_ptr,
            self.var_edges,
            self.edge_var,
            self.edge_chk,
        )

    def validate(self) -> bool:
        """Validate graph structure."""
        # Check that all edge indices are valid
        assert self.edge_var.max() < self.V, "Invalid variable index in edges"
        assert self.edge_chk.max() < self.C, "Invalid check index in edges"

        # Correctness checks: ensure CSR pointers have correct length
        assert self.chk_ptr.shape[0] == self.C + 1, (
            f"chk_ptr must have length C+1 ({self.C + 1}), got {self.chk_ptr.shape[0]}"
        )
        assert self.var_ptr.shape[0] == self.V + 1, (
            f"var_ptr must have length V+1 ({self.V + 1}), got {self.var_ptr.shape[0]}"
        )

        # Check that pointers are monotonic
        assert torch.all(self.chk_ptr[1:] >= self.chk_ptr[:-1]), "Invalid check pointers"
        assert torch.all(self.var_ptr[1:] >= self.var_ptr[:-1]), "Invalid variable pointers"

        # Check that edge counts match
        assert self.chk_ptr[-1] == self.E, "Check pointer mismatch"
        assert self.var_ptr[-1] == self.E, "Variable pointer mismatch"

        return True

    def get_stats(self) -> dict:
        """Get graph statistics for debugging."""
        return {
            "checks": self.C,
            "variables": self.V,
            "edges": self.E,
            "max_degree_check": self.max_deg_chk,
            "max_degree_variable": self.max_deg_var,
            "avg_degree_check": self.E / self.C,
            "avg_degree_variable": self.E / self.V,
        }

    def build_block_metadata(
        self,
        num_blocks: Optional[int] = None,
        target_edges_per_block: Optional[int] = None,
    ):
        """Build block metadata for layered scheduling using edge-based heuristics.

        This method computes:
        - block_chk_start[nb], block_chk_end[nb]: Check ranges for each block
        - block_var_ptr[nb+1], block_vars[total]: Variable lists touched by each block

        Block count is determined by edge-based heuristics (more stable than check-based)
        to balance kernel launch overhead vs block size.

        Args:
            num_blocks: Explicit number of blocks (if None, computed from target_edges_per_block)
            target_edges_per_block: Target edges per block for automatic sizing (default: 200000)

        Returns:
            Tuple of (block_chk_start, block_chk_end, block_var_ptr, block_vars, num_blocks)
            All as torch tensors on device (except num_blocks which is int).
        """
        # Determine num_blocks from edge-based heuristic if not provided
        if num_blocks is None:
            if target_edges_per_block is None:
                target_edges_per_block = 1000  # Default: 200k edges per block
            # Compute num_blocks based on total edges (use ceil, not floor)
            # No max cap - if user wants many blocks, let them (they can set num_blocks explicitly)
            num_blocks = max(1, math.ceil(self.E / target_edges_per_block))

        num_blocks = int(num_blocks)
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if num_blocks > self.C:
            raise ValueError(f"num_blocks ({num_blocks}) cannot exceed number of checks ({self.C})")

        # Edge-balanced block splitting: split by cumulative edge count, not check count
        # This ensures each block processes roughly the same number of edges
        chk_ptr_host = self.chk_ptr.cpu().numpy()

        # Compute edge count per check (degree)
        edges_per_check = chk_ptr_host[1:] - chk_ptr_host[:-1]

        # Compute prefix sum of edge counts (cumulative edges up to each check)
        cumulative_edges = np.zeros(self.C + 1, dtype=np.int64)
        cumulative_edges[1:] = np.cumsum(edges_per_check)
        # cumulative_edges[i] = total edges in checks [0, i)

        # Target edges per block
        target_edges = self.E / num_blocks

        # Find block boundaries by splitting on cumulative edge count
        block_chk_start = np.zeros(num_blocks, dtype=np.int32)
        block_chk_end = np.zeros(num_blocks, dtype=np.int32)

        block_idx = 0
        target_cumulative = target_edges
        block_chk_start[0] = 0

        for chk_idx in range(1, self.C + 1):
            # Check if we've accumulated enough edges for current block
            # Or if we're on the last block, we must end it at C
            if cumulative_edges[chk_idx] >= target_cumulative:
                # End current block at this check
                block_chk_end[block_idx] = chk_idx
                block_idx += 1
                if block_idx < num_blocks:
                    block_chk_start[block_idx] = chk_idx
                    target_cumulative = (block_idx + 1) * target_edges
                else:
                    # We've filled all blocks, break early
                    break

        # Ensure all blocks are properly terminated
        # If we didn't fill all blocks, the last one ends at C
        if block_idx < num_blocks:
            # Some blocks weren't filled - end the current block at C
            block_chk_end[block_idx] = self.C
            block_idx += 1
            # Fill any remaining empty blocks (shouldn't happen, but handle gracefully)
            for b in range(block_idx, num_blocks):
                block_chk_start[b] = self.C
                block_chk_end[b] = self.C
        else:
            # All blocks filled - ensure last block ends at C
            block_chk_end[num_blocks - 1] = self.C

        # Post-process: ensure no empty blocks (merge empty blocks into previous)
        # This can happen with highly irregular degree distributions
        non_empty_blocks = []
        for b in range(num_blocks):
            if block_chk_end[b] > block_chk_start[b]:
                non_empty_blocks.append(b)

        if len(non_empty_blocks) < num_blocks:
            # Some blocks are empty - compact them
            num_non_empty = len(non_empty_blocks)
            if num_non_empty == 0:
                # All blocks empty (shouldn't happen, but handle gracefully)
                num_non_empty = 1
                non_empty_blocks = [0]
                block_chk_start[0] = 0
                block_chk_end[0] = self.C

            # Rebuild block arrays with only non-empty blocks
            block_chk_start_new = np.zeros(num_non_empty, dtype=np.int32)
            block_chk_end_new = np.zeros(num_non_empty, dtype=np.int32)
            for i, b in enumerate(non_empty_blocks):
                block_chk_start_new[i] = block_chk_start[b]
                block_chk_end_new[i] = block_chk_end[b]

            block_chk_start = block_chk_start_new
            block_chk_end = block_chk_end_new
            num_blocks = num_non_empty

        # Correctness assertions: ensure blocks form a contiguous, non-empty partition
        # These catch any edge cases where compaction or splitting logic might produce invalid blocks
        assert block_chk_start[0] == 0, f"First block must start at 0, got {block_chk_start[0]}"
        assert block_chk_end[-1] == self.C, (
            f"Last block must end at C={self.C}, got {block_chk_end[-1]}"
        )
        assert all(block_chk_end[i] == block_chk_start[i + 1] for i in range(num_blocks - 1)), (
            "Blocks must be contiguous (end[i] == start[i+1] for all adjacent pairs)"
        )
        assert all(block_chk_end[i] > block_chk_start[i] for i in range(num_blocks)), (
            "All blocks must be non-empty (end[i] > start[i] for all blocks)"
        )

        # For each block, find unique variables touched by checks in that block
        # Use CSR structure: checks in [chk_start, chk_end) have edges in [chk_ptr[chk_start], chk_ptr[chk_end])
        # This is contiguous in check-major ordering, so we can use a single slice
        chk_ptr_host = self.chk_ptr.cpu().numpy()
        chk_edges_host = self.chk_edges.cpu().numpy()
        edge_var_host = self.edge_var.cpu().numpy()

        # Correctness check: ensure CSR pointers have correct length
        assert chk_ptr_host.shape[0] == self.C + 1, (
            f"chk_ptr must have length C+1 ({self.C + 1}), got {chk_ptr_host.shape[0]}"
        )

        block_vars_list = []  # List of variable lists per block
        block_var_ptr = [0]  # Prefix sum pointer

        # Note: num_blocks may have been reduced if empty blocks were removed
        for b in range(num_blocks):
            chk_start = int(block_chk_start[b])
            chk_end = int(block_chk_end[b])

            # Fast numpy path: use contiguous edge range (much faster than list comprehension)
            # Checks in [chk_start, chk_end) have edges in [chk_ptr[chk_start], chk_ptr[chk_end])
            # This works because edges are stored in check-major order
            e0 = chk_ptr_host[chk_start]
            e1 = chk_ptr_host[chk_end]

            if e1 > e0:
                # Get all edge IDs for this block (contiguous slice)
                all_edge_ids = chk_edges_host[e0:e1]
                # Map edge IDs → var IDs (vectorized indexing)
                all_var_ids = edge_var_host[all_edge_ids]
                # Get unique var IDs (sorted automatically by np.unique)
                block_vars_sorted = np.unique(all_var_ids).astype(np.int32)
            else:
                block_vars_sorted = np.array([], dtype=np.int32)

            block_vars_list.append(block_vars_sorted.tolist())
            block_var_ptr.append(block_var_ptr[-1] + len(block_vars_sorted))

        # Flatten into single array
        total_vars = block_var_ptr[-1]
        block_vars_flat = np.zeros(total_vars, dtype=np.int32)
        for b, var_list in enumerate(block_vars_list):
            start = block_var_ptr[b]
            end = block_var_ptr[b + 1]
            block_vars_flat[start:end] = var_list

        # Diagnostic output: block statistics for validation/optimization
        import logging

        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Block metadata: {num_blocks} blocks, {total_vars} total variables")
            for b in range(num_blocks):
                chk_start = int(block_chk_start[b])
                chk_end = int(block_chk_end[b])
                var_start = int(block_var_ptr[b])
                var_end = int(block_var_ptr[b + 1])
                # Compute edges in block
                e0 = chk_ptr_host[chk_start]
                e1 = chk_ptr_host[chk_end]
                edges_in_block = e1 - e0
                vars_in_block = var_end - var_start
                # Compute overlap with previous block
                overlap_with_prev = 0
                if b > 0:
                    prev_var_start = int(block_var_ptr[b - 1])
                    prev_var_end = int(block_var_ptr[b])
                    prev_vars = set(block_vars_flat[prev_var_start:prev_var_end])
                    curr_vars = set(block_vars_flat[var_start:var_end])
                    overlap_with_prev = len(prev_vars & curr_vars)
                    overlap_ratio = overlap_with_prev / vars_in_block if vars_in_block > 0 else 0.0
                else:
                    overlap_ratio = 0.0
                logger.debug(
                    f"  Block {b}: checks [{chk_start}, {chk_end}), "
                    f"edges={edges_in_block}, vars={vars_in_block}, "
                    f"overlap_with_prev={overlap_with_prev} (ratio={overlap_ratio:.3f})"
                )

        # Convert to tensors on device (use existing tensor's device for safety)
        dev = self.chk_ptr.device
        block_chk_start_t = torch.from_numpy(block_chk_start).to(dev)
        block_chk_end_t = torch.from_numpy(block_chk_end).to(dev)
        block_var_ptr_t = torch.from_numpy(np.array(block_var_ptr, dtype=np.int32)).to(dev)
        block_vars_t = torch.from_numpy(block_vars_flat).to(dev)

        return block_chk_start_t, block_chk_end_t, block_var_ptr_t, block_vars_t, num_blocks
