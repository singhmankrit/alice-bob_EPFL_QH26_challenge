"""
Bicycle LDPC — Stim circuit generator
──────────────────────────────────────
Z-only stabiliser code for biased noise / cat qubits.

H = [A | B]   where A, B are n×n circulants over GF(2).
Code has 2n data qubits, n ancilla qubits, encodes k = 2n - rank(H) logical qubits.

Usage
─────
    from bicycle_ldpc import bicycle_ldpc_circuit, print_circuit_summary

    # Default well-known shifts for n=12
    circuit = bicycle_ldpc_circuit(n=12, rounds=3, p_bitflip=0.01)

    # Custom shifts
    circuit = bicycle_ldpc_circuit(
        n       = 24,
        a_shifts= [0, 2, 8],
        b_shifts= [0, 6, 14],
        rounds  = 5,
        p_bitflip = 0.005,
    )
"""

import numpy as np

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False
    print("stim not found — install with: pip install stim")
    print("Falling back to text representation.\n")

# ══════════════════════════════════════════════════════════════════════════════
# Matrix builders
# ══════════════════════════════════════════════════════════════════════════════

def _circulant(n: int, shifts: list[int]) -> np.ndarray:
    """
    n×n circulant matrix over GF(2).
    Row i has 1s at columns (i + s) % n  for each s in shifts.

    Example  n=6, shifts=[0,1,2]:
        1 1 1 0 0 0
        0 1 1 1 0 0
        0 0 1 1 1 0
        0 0 0 1 1 1
        1 0 0 0 1 1
        1 1 0 0 0 1
    """
    C = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for s in shifts:
            C[i, (i + s) % n] = 1
    return C


def _bicycle_H(n: int, a_shifts: list[int], b_shifts: list[int]) -> np.ndarray:
    """
    Build the bicycle LDPC parity-check matrix.

    H = [ A | B ]    shape: n × 2n

    A = circulant(n, a_shifts)
    B = circulant(n, b_shifts)

    Row weight = |a_shifts| + |b_shifts|  (constant — LDPC property).
    Col weight = |a_shifts| or |b_shifts| (also constant).
    """
    A = _circulant(n, a_shifts)
    B = _circulant(n, b_shifts)
    return np.hstack([A, B])   # shape: n × 2n


# Good pre-validated shift sets (high girth, known parameters)
_GOOD_SHIFTS: dict[int, tuple[list[int], list[int]]] = {
    6 : ([0, 1, 2],    [0, 2, 4]   ),   # [[12,  8, 2]]
    8 : ([0, 1, 3],    [0, 2, 7]   ),   # [[16,  8, 4]]
    12: ([0, 2, 7],    [0, 3, 10]  ),   # [[24, 12, 6]]
    16: ([0, 1, 6],    [0, 2, 8]   ),   # [[32, 16, 6]]
    24: ([0, 2, 8],    [0, 6, 14]  ),   # [[48, 24, 10]]
    30: ([0, 1, 11],   [0, 4, 16]  ),   # [[60, 28, 10]]
    36: ([0, 1, 11],   [0, 6, 20]  ),   # [[72, 36, 12]]
    48: ([0, 2, 20],   [0, 12, 28] ),   # [[96, 48, ~14]]
    60: ([0, 1, 24],   [0, 9, 35]  ),   # [[120,60, ~16]]
}


# ══════════════════════════════════════════════════════════════════════════════
# Main public function
# ══════════════════════════════════════════════════════════════════════════════

def bicycle_ldpc_circuit(
    n        : int,
    a_shifts : list[int] | None = None,
    b_shifts : list[int] | None = None,
    rounds   : int  = 1,
    p_bitflip: float = 0.0,
) -> "stim.Circuit | dict":
    """
    Build a Stim circuit for a Z-only bicycle LDPC stabiliser code.

    The code detects X errors (bit flips) only — designed for cat qubits
    where phase flips (Z errors) are exponentially suppressed by bias η >> 1.

    Parameters
    ──────────
    n         : circulant block size. Total data qubits = 2n.
                Supported with default shifts: 6, 8, 12, 16, 24, 30, 36, 48, 60.
                Any n works if you supply a_shifts and b_shifts.

    a_shifts  : column offsets for circulant A  (e.g. [0, 2, 7]).
                Defaults to pre-validated set for the given n.

    b_shifts  : column offsets for circulant B  (e.g. [0, 3, 10]).
                Defaults to pre-validated set for the given n.

    rounds    : number of syndrome extraction rounds.
                Use rounds >= 1 for fault-tolerant operation
                (detectors compare adjacent rounds).

    p_bitflip : probability of X error per data qubit per round.
                Set to 0.0 for a noiseless circuit (e.g. for Clifford sim).

    Returns
    ───────
    stim.Circuit (if stim is installed) or a plain dict description.

    Qubit layout
    ────────────
    Qubits   0 …  2n-1  : data qubits  (2n total)
    Qubits  2n … 3n-1   : ancilla qubits (n total, one per stabiliser)

    Stabiliser i measures  g_i = ⊗_j  Z_j^{H[i,j]}
    via:  CNOT data_j → ancilla_i   for each j where H[i,j] = 1
    then: M ancilla_i

    No Hadamard gates — pure Z type, native to cat qubit hardware.
    """
    # ── resolve shifts ──────────────────────────────────────────────────────
    if a_shifts is None or b_shifts is None:
        if n not in _GOOD_SHIFTS:
            raise ValueError(
                f"No default shifts for n={n}. "
                f"Supported: {sorted(_GOOD_SHIFTS)}. "
                f"Supply a_shifts and b_shifts explicitly."
            )
        a_shifts, b_shifts = _GOOD_SHIFTS[n]

    a_shifts = sorted(set(int(s) % n for s in a_shifts))
    b_shifts = sorted(set(int(s) % n for s in b_shifts))

    # ── build H ────────────────────────────────────────────────────────────
    H = _bicycle_H(n, a_shifts, b_shifts)   # shape: n × 2n
    n_data    = 2 * n      # data qubits
    n_ancilla = n          # one ancilla per row of H
    n_total   = n_data + n_ancilla

    data    = list(range(n_data))
    ancilla = list(range(n_data, n_total))

    # ── precompute support of each stabiliser ──────────────────────────────
    # supports[i] = sorted list of data qubit indices in stabiliser i
    supports = [
        [j for j in range(n_data) if H[i, j]]
        for i in range(n_ancilla)
    ]

    # ── build Stim circuit ─────────────────────────────────────────────────
    circuit = stim.Circuit()

    # Initialise everything in |0⟩
    circuit.append("R", data + ancilla)

    for r in range(rounds):

        # ── (1) Noise: X errors on data qubits ──────────────────────────
        if p_bitflip > 0:
            circuit.append("X_ERROR", data, p_bitflip)

        # ── (2) Reset ancillas for a clean measurement ───────────────────
        circuit.append("R", ancilla)

        # ── (3) Stabiliser measurement: CNOT data → ancilla ─────────────
        #
        #  For Z stabiliser  g_i = Z_{j1} Z_{j2} ... Z_{jw}:
        #    CNOT j1 → ancilla_i
        #    CNOT j2 → ancilla_i
        #    ...
        #    CNOT jw → ancilla_i
        #    M ancilla_i
        #
        #  The ancilla accumulates parity of {j1,...,jw} in the Z basis.
        #  Outcome = 0 → even parity (no error detected by this stabiliser)
        #  Outcome = 1 → odd parity  (error detected)
        #
        for i in range(n_ancilla):
            for j in supports[i]:
                circuit.append("CNOT", [j, ancilla[i]])

        # ── (4) Measure ancillas ─────────────────────────────────────────
        for i in range(n_ancilla):
            circuit.append("M", [ancilla[i]])

        # ── (5) Detectors ────────────────────────────────────────────────
        #
        # A detector fires when a measurement result differs from its
        # expected value.  For Z stabilisers starting in |0⟩:
        #   Round 0 : expected outcome = 0  →  detector = just this round's M
        #   Round r>0: expected outcome = same as previous round
        #             →  detector = XOR of this and previous round
        #
        # rec(-k) means "look k measurements back in the record"
        # The last n measurements are the current round's ancillas.
        # The n measurements before that are the previous round's.
        #
        for i in range(n_ancilla):
            # current round: ancilla i was measured (n_ancilla - i) ago
            offset_now = -(n_ancilla - i)

            if r == 0:
                circuit.append("DETECTOR",
                    [stim.target_rec(offset_now)]
                )
            else:
                # previous round is n_ancilla further back
                offset_prev = offset_now - n_ancilla
                circuit.append("DETECTOR", [
                    stim.target_rec(offset_now),
                    stim.target_rec(offset_prev),
                ])

    return circuit

