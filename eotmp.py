#!/usr/bin/env python3
"""Reference implementation of the EoTMP protocol at the vector level.

This module follows the data flow in Li et al., *EoTMP: Efficient
Over-Threshold Multi-Party Private Set Intersection* (TIFS 2025):

1. encode each private set as a Bloom filter;
2. batch the filters into SIMD-sized vectors;
3. add the vectors to obtain encrypted-position frequencies;
4. evaluate the over-threshold polynomial and decode at the receiver.

The homomorphic ciphertext layer is represented by integer vectors so the
experiment is deterministic and easy to inspect.  This is useful for
correctness, false-positive, and scaling experiments, but it is not a
cryptographic implementation and must not be used to protect real data.
The operation counters and communication estimates mirror the BFV/MHE
operations described in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


def murmur3_32(data: bytes, seed: int = 0) -> int:
    """Return the unsigned MurmurHash3 x86 32-bit hash."""

    mask = 0xFFFFFFFF
    h = seed & mask
    nblocks = len(data) // 4
    for i in range(nblocks):
        k = int.from_bytes(data[4 * i : 4 * i + 4], "little")
        k = (k * 0xCC9E2D51) & mask
        k = ((k << 15) | (k >> 17)) & mask
        k = (k * 0x1B873593) & mask
        h ^= k
        h = ((h << 13) | (h >> 19)) & mask
        h = (h * 5 + 0xE6546B64) & mask

    tail = data[4 * nblocks :]
    k = 0
    if len(tail) == 3:
        k ^= tail[2] << 16
    if len(tail) >= 2:
        k ^= tail[1] << 8
    if len(tail) >= 1:
        k ^= tail[0]
        k = (k * 0xCC9E2D51) & mask
        k = ((k << 15) | (k >> 17)) & mask
        k = (k * 0x1B873593) & mask
        h ^= k

    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & mask
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & mask
    h ^= h >> 16
    return h & mask


def hash_positions(item: str | bytes, bits: int, hashes: int, seed: int) -> tuple[int, ...]:
    """Use double hashing to derive ``hashes`` positions in ``[0, bits)``."""

    raw = item.encode("utf-8") if isinstance(item, str) else item
    h1 = murmur3_32(raw, seed)
    h2 = murmur3_32(raw, seed ^ 0x9E3779B9) | 1
    return tuple((h1 + i * h2) % bits for i in range(hashes))


class BloomFilter:
    """Small integer Bloom filter used by the protocol simulation."""

    def __init__(self, bits: int, hashes: int, seed: int = 0) -> None:
        if bits <= 0 or hashes <= 0:
            raise ValueError("bits and hashes must be positive")
        self.bits = bits
        self.hashes = hashes
        self.seed = seed
        self.values = [0] * bits

    def add(self, item: str | bytes) -> None:
        for pos in hash_positions(item, self.bits, self.hashes, self.seed):
            self.values[pos] = 1

    def contains(self, item: str | bytes) -> bool:
        return all(self.values[p] for p in hash_positions(item, self.bits, self.hashes, self.seed))

    @classmethod
    def from_set(cls, items: Iterable[str], bits: int, hashes: int, seed: int = 0) -> "BloomFilter":
        result = cls(bits, hashes, seed)
        for item in items:
            result.add(item)
        return result


def bloom_fpp_estimate(bits: int, hashes: int, set_size: int) -> float:
    """The standard Bloom filter false-positive estimate from the paper."""

    if set_size <= 0:
        return 0.0
    return (1.0 - (1.0 - 1.0 / bits) ** (hashes * set_size)) ** hashes


def choose_bloom_parameters(set_size: int, target_fpp: float, *, max_hashes: int = 32) -> tuple[int, int, float]:
    """Choose ``(d, k, estimated_fpp)`` by a small exhaustive search.

    The paper recommends doing this numerical search because the OT-MPSI
    false-positive probability also depends on ``N`` and ``T``.  This helper
    optimizes the single-filter estimate and is a useful starting point.
    """

    if set_size <= 0 or not 0 < target_fpp < 1:
        raise ValueError("set_size must be positive and target_fpp must be in (0, 1)")
    # d ≈ -m ln(eps)/(ln 2)^2 is a good upper bound for the search.
    upper = max(8, int(math.ceil(-set_size * math.log(target_fpp) / math.log(2) ** 2 * 1.3)))
    best: tuple[int, int, float] | None = None
    for d in range(max(8, set_size), upper + 1):
        for k in range(1, max_hashes + 1):
            fpp = bloom_fpp_estimate(d, k, set_size)
            candidate = (d, k, fpp)
            if fpp <= target_fpp and (best is None or d < best[0] or (d == best[0] and k < best[1])):
                best = candidate
    if best is not None:
        return best
    # The loop should normally find a solution; return the best estimate if a
    # very strict target was requested with a small search budget.
    return min(
        ((d, k, bloom_fpp_estimate(d, k, set_size)) for d in range(max(8, set_size), upper + 1) for k in range(1, max_hashes + 1)),
        key=lambda x: x[2],
    )


def polynomial_coefficients(roots: Sequence[int]) -> list[int]:
    """Return coefficients in ascending order for ``prod(x - root)``."""

    coefficients = [1]
    for root in roots:
        out = [0] * (len(coefficients) + 1)
        for power, coefficient in enumerate(coefficients):
            out[power] -= root * coefficient
            out[power + 1] += coefficient
        coefficients = out
    return coefficients


def evaluate_polynomial(coefficients: Sequence[int], value: int) -> int:
    """Evaluate ascending-order coefficients with Horner's rule."""

    result = 0
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def paterson_stockmeyer_shape(degree: int) -> tuple[int, int, int]:
    """Return ``(L, H, estimated ciphertext-ciphertext multiplications)``.

    This is an operation-count model of the optimization in Section IV-C;
    it does not perform ciphertext arithmetic itself.
    """

    if degree < 1:
        return 1, 1, 0
    # The paper uses L ≈ sqrt(2(degree+1)); H = ceil((degree+1)/L).
    l = max(2, int(math.sqrt(2 * (degree + 1))))
    h = math.ceil((degree + 1) / l)
    # Build x^2...x^(L-1), x^L...x^((H-1)L), plus block products.
    multiplications = max(0, l - 2) + max(0, h - 2) + max(0, h - 1)
    return l, h, multiplications


@dataclass(frozen=True)
class ProtocolStats:
    participants: int
    online_participants: int
    threshold: int
    receiver: int
    set_size: int
    bloom_bits: int
    bloom_hashes: int
    slots: int
    batches: int
    polynomial_degree: int
    polynomial_low_degree: int
    polynomial_high_blocks: int
    ciphertext_additions: int
    ciphertext_multiplications: int
    plaintext_ciphertext_multiplications: int
    rounds: int
    estimated_ciphertext_bytes: int
    estimated_upload_bytes: int
    estimated_server_download_bytes: int
    estimated_receiver_upload_bytes: int
    estimated_total_communication_bytes: int
    elapsed_ms: float
    exact_count: int
    predicted_count: int
    false_positives: int
    false_negatives: int


@dataclass
class ProtocolResult:
    result: set[str]
    exact: set[str]
    false_positives: set[str]
    false_negatives: set[str]
    threshold_vector: list[int]
    frequency_vector: list[int]
    stats: ProtocolStats


def _batch(values: Sequence[int], slots: int) -> list[list[int]]:
    return [list(values[i : i + slots]) for i in range(0, len(values), slots)]


def run_eotmp(
    sets: Sequence[set[str]],
    threshold: int,
    *,
    receiver: int = 0,
    bloom_bits: int = 1024,
    bloom_hashes: int = 10,
    hash_seed: int = 0,
    slots: int = 8192,
    online: Sequence[int] | None = None,
    ciphertext_bytes: int = 0,
    rng_seed: int = 1,
) -> ProtocolResult:
    """Run the EoTMP data path over plaintext vectors.

    ``online`` selects the participants whose ciphertexts arrive.  The
    threshold is evaluated over those online participants, matching the
    t-out-of-N availability experiment.  ``ciphertext_bytes`` is optional;
    passing the paper's BFV estimate (about 0.63 MiB for log(n)=13) makes the
    communication estimate concrete.
    """

    started = time.perf_counter()
    n_total = len(sets)
    if n_total == 0:
        raise ValueError("at least one participant is required")
    if online is None:
        online_indices = list(range(n_total))
    else:
        online_indices = list(online)
    if len(set(online_indices)) != len(online_indices) or any(i < 0 or i >= n_total for i in online_indices):
        raise ValueError("online contains invalid or duplicate participant indices")
    if receiver not in online_indices:
        raise ValueError("the receiver must be online")
    n = len(online_indices)
    if not 1 <= threshold <= n:
        raise ValueError("threshold must be between 1 and the number of online participants")
    if not 0 <= receiver < n_total:
        raise ValueError("invalid receiver")
    if slots <= 0:
        raise ValueError("slots must be positive")

    filters = [BloomFilter.from_set(sets[i], bloom_bits, bloom_hashes, hash_seed) for i in online_indices]
    frequency = [sum(f.values[pos] for f in filters) for pos in range(bloom_bits)]

    # The paper uses the complementary polynomial when T < N/2: it has roots
    # at 0,...,T-1 and the receiver accepts non-zero positions.  Otherwise,
    # use roots T,...,N and accept zero positions.  This is the advertised
    # O(min(T, N-T+1)) threshold functionality.
    complement = threshold < n / 2
    roots = range(0, threshold) if complement else range(threshold, n + 1)
    coefficients = polynomial_coefficients(list(roots))
    rng = random.Random(rng_seed)
    threshold_vector = []
    for count in frequency:
        raw = evaluate_polynomial(coefficients, count)
        threshold_vector.append(0 if raw == 0 else raw * rng.randrange(1, 2**31))
    predicted = {
        item
        for item in sets[receiver]
        if all(
            (threshold_vector[pos] != 0 if complement else threshold_vector[pos] == 0)
            for pos in hash_positions(item, bloom_bits, bloom_hashes, hash_seed)
        )
    }
    counts = {item: sum(item in sets[i] for i in online_indices) for item in sets[receiver]}
    exact = {item for item, count in counts.items() if count >= threshold}
    false_positives = predicted - exact
    false_negatives = exact - predicted

    batches = math.ceil(bloom_bits / slots)
    degree = len(coefficients) - 1
    low, high, ps_muls = paterson_stockmeyer_shape(degree)
    # One addition per other participant and batch; the server then evaluates
    # one polynomial per batch.  Counts are useful even though arithmetic is
    # represented by vectors here.
    ciphertext_additions = max(0, n - 1) * batches
    ciphertext_multiplications = ps_muls * batches
    plaintext_ciphertext_multiplications = (degree + 1) * batches
    upload = max(0, n - 1) * batches * ciphertext_bytes
    # Round 2 sends the masked result to the other parties; round 3 sends
    # partial key-switch results to the receiver.  Each has one ciphertext per
    # batch and non-server participant in this estimate.
    download = max(0, n - 1) * batches * ciphertext_bytes
    receiver_upload = max(0, n - 1) * batches * ciphertext_bytes
    elapsed_ms = (time.perf_counter() - started) * 1000
    stats = ProtocolStats(
        participants=n_total,
        online_participants=n,
        threshold=threshold,
        receiver=receiver,
        set_size=len(sets[receiver]),
        bloom_bits=bloom_bits,
        bloom_hashes=bloom_hashes,
        slots=slots,
        batches=batches,
        polynomial_degree=degree,
        polynomial_low_degree=low,
        polynomial_high_blocks=high,
        ciphertext_additions=ciphertext_additions,
        ciphertext_multiplications=ciphertext_multiplications,
        plaintext_ciphertext_multiplications=plaintext_ciphertext_multiplications,
        rounds=3,
        estimated_ciphertext_bytes=ciphertext_bytes,
        estimated_upload_bytes=upload,
        estimated_server_download_bytes=download,
        estimated_receiver_upload_bytes=receiver_upload,
        estimated_total_communication_bytes=upload + download + receiver_upload,
        elapsed_ms=elapsed_ms,
        exact_count=len(exact),
        predicted_count=len(predicted),
        false_positives=len(false_positives),
        false_negatives=len(false_negatives),
    )
    return ProtocolResult(predicted, exact, false_positives, false_negatives, threshold_vector, frequency, stats)


def make_synthetic_sets(
    participants: int,
    set_size: int,
    threshold: int,
    *,
    universe_size: int | None = None,
    seed: int = 7,
) -> list[set[str]]:
    """Generate sets with both guaranteed threshold hits and distractors."""

    if participants < 1 or set_size < 1 or not 1 <= threshold <= participants:
        raise ValueError("invalid participants, set_size, or threshold")
    universe_size = universe_size or max(participants * set_size * 4, set_size + 10)
    if universe_size < set_size:
        raise ValueError("universe_size must be at least set_size")
    rng = random.Random(seed)
    universe = [f"item-{i}" for i in range(universe_size)]
    # Reserve a small set of common items at exactly threshold and one item at
    # every higher frequency, so correctness checks exercise all branches.
    common = set(universe[: min(4, set_size)])
    sets = [set(common) for _ in range(participants)]
    for item in list(common):
        for participant in rng.sample(range(participants), participants - threshold):
            sets[participant].remove(item)
    for participant in range(participants):
        choices = [item for item in universe if item not in sets[participant]]
        sets[participant].update(rng.sample(choices, set_size - len(sets[participant])))
    return sets


def benchmark(
    participant_values: Sequence[int],
    set_sizes: Sequence[int],
    *,
    threshold_gap: int = 3,
    bits: int = 4096,
    hashes: int = 10,
    slots: int = 8192,
    ciphertext_bytes: int = 0,
    repeats: int = 3,
    seed: int = 7,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for participants in participant_values:
        for set_size in set_sizes:
            threshold = max(1, participants - threshold_gap + 1)
            samples = []
            for repeat in range(repeats):
                sets = make_synthetic_sets(participants, set_size, threshold, seed=seed + repeat)
                samples.append(
                    run_eotmp(
                        sets,
                        threshold,
                        bloom_bits=bits,
                        bloom_hashes=hashes,
                        slots=slots,
                        ciphertext_bytes=ciphertext_bytes,
                        rng_seed=seed + repeat,
                    )
                )
            stats = [sample.stats for sample in samples]
            row = asdict(stats[0])
            row["elapsed_ms_mean"] = statistics.mean(s.elapsed_ms for s in stats)
            row["elapsed_ms_stdev"] = statistics.pstdev(s.elapsed_ms for s in stats)
            rows.append(row)
    return rows


def _jsonable_result(result: ProtocolResult) -> dict[str, object]:
    return {
        "result": sorted(result.result),
        "exact": sorted(result.exact),
        "false_positives": sorted(result.false_positives),
        "false_negatives": sorted(result.false_negatives),
        "stats": asdict(result.stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-level EoTMP experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a correctness experiment")
    demo.add_argument("--participants", type=int, default=8)
    demo.add_argument("--set-size", type=int, default=64)
    demo.add_argument("--threshold", type=int, default=6)
    demo.add_argument("--bits", type=int, default=4096)
    demo.add_argument("--hashes", type=int, default=10)
    demo.add_argument("--slots", type=int, default=8192)
    demo.add_argument("--ciphertext-bytes", type=int, default=0,
                      help="optional BFV ciphertext size for communication estimates")
    demo.add_argument("--json", action="store_true")

    fpp = sub.add_parser("fpp", help="print the standard Bloom filter FPP estimate")
    fpp.add_argument("--set-size", type=int, required=True)
    fpp.add_argument("--bits", type=int, required=True)
    fpp.add_argument("--hashes", type=int, required=True)

    bench = sub.add_parser("benchmark", help="run a scaling benchmark")
    bench.add_argument("--participants", type=int, nargs="+", default=[8, 16, 32])
    bench.add_argument("--set-sizes", type=int, nargs="+", default=[64, 256])
    bench.add_argument("--threshold-gap", type=int, default=3)
    bench.add_argument("--bits", type=int, default=4096)
    bench.add_argument("--hashes", type=int, default=10)
    bench.add_argument("--slots", type=int, default=8192)
    bench.add_argument("--ciphertext-bytes", type=int, default=0,
                       help="optional BFV ciphertext size for communication estimates")
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "demo":
        sets = make_synthetic_sets(args.participants, args.set_size, args.threshold)
        result = run_eotmp(sets, args.threshold, bloom_bits=args.bits, bloom_hashes=args.hashes,
                           slots=args.slots, ciphertext_bytes=args.ciphertext_bytes)
        print(json.dumps(_jsonable_result(result), indent=2, sort_keys=True) if args.json else _human_demo(result))
    elif args.command == "fpp":
        print(f"estimated_fpp={bloom_fpp_estimate(args.bits, args.hashes, args.set_size):.8g}")
    else:
        rows = benchmark(args.participants, args.set_sizes, threshold_gap=args.threshold_gap, bits=args.bits,
                         hashes=args.hashes, slots=args.slots, ciphertext_bytes=args.ciphertext_bytes,
                         repeats=args.repeats)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(rows, indent=2))


def _human_demo(result: ProtocolResult) -> str:
    s = result.stats
    return (
        f"predicted={len(result.result)} exact={len(result.exact)} "
        f"false_positives={len(result.false_positives)} false_negatives={len(result.false_negatives)}\n"
        f"N={s.online_participants} T={s.threshold} d={s.bloom_bits} k={s.bloom_hashes} "
        f"batches={s.batches} degree={s.polynomial_degree} rounds={s.rounds}\n"
        f"vector_runtime={s.elapsed_ms:.3f} ms (plaintext reference; MHE is not performed)"
    )


if __name__ == "__main__":
    main()
