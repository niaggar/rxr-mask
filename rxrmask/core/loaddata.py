from rxrmask.core import ReflectivityScan, EnergyScan

from typing import List, Literal, Optional
import numpy as np


def _pol(tok: str) -> Literal["s", "p"]:
    return "s" if tok.upper().startswith("S") else "p"


def load_reflectivity_scans(path: str, initial_name: str) -> List[ReflectivityScan]:
    scans: List[ReflectivityScan] = []
    cur_scan: Optional[int] = None
    cur_pol: Optional[str] = None
    cur_E: Optional[float] = None
    qz_vals: List[float] = []
    R_vals: List[float] = []

    def flush():
        nonlocal cur_scan, cur_pol, cur_E, qz_vals, R_vals
        if cur_scan is None or cur_pol is None or cur_E is None or not qz_vals:
            # nothing to flush
            cur_scan = cur_pol = cur_E = None
            qz_vals, R_vals = [], []
            return

        qz = np.asarray(qz_vals, float)
        R = np.asarray(R_vals, float)
        scans.append(
            ReflectivityScan(
                name=f"{initial_name}_{cur_scan}_{cur_E:g}",
                energy_eV=cur_E,
                pol=_pol(cur_pol),
                qz=qz,
                R=R,
                bounds=[(float(qz.min()), float(qz.max()))],
                weights=[1.0],
            )
        )
        cur_scan = cur_pol = cur_E = None
        qz_vals, R_vals = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("="):
                flush()
                continue

            parts = s.split()
            # Header: "<scan> <S|P> A <energy> ..."
            if len(parts) >= 4:
                try:
                    maybe_scan = int(parts[0])
                    if parts[1].upper() in ("S", "P"):
                        flush()
                        cur_scan = maybe_scan
                        cur_pol = parts[1]
                        cur_E = float(parts[3])
                        continue
                except ValueError:
                    pass

            # Data: energy angle qz R  (we only use qz, R)
            if len(parts) >= 4:
                try:
                    # row_E = float(parts[0])   # could be checked against cur_E if desired
                    # angle = float(parts[1])   # unused here
                    qz_vals.append(float(parts[2]))
                    R_vals.append(float(parts[3]))
                except ValueError:
                    pass

    flush()
    return scans


def load_energy_scans(path: str, initial_name: str) -> List[EnergyScan]:
    scans: List[EnergyScan] = []
    cur_scan: Optional[int] = None
    cur_pol: Optional[str] = None
    cur_theta: Optional[float] = None
    E_vals: List[float] = []
    R_vals: List[float] = []

    def flush():
        nonlocal cur_scan, cur_pol, cur_theta, E_vals, R_vals
        if cur_scan is None or cur_pol is None or cur_theta is None or not E_vals:
            cur_scan = cur_pol = cur_theta = None
            E_vals, R_vals = [], []
            return
        E = np.asarray(E_vals, float)
        R = np.asarray(R_vals, float)
        scans.append(
            EnergyScan(
                name=f"{initial_name}_{cur_scan}_{cur_theta:g}deg",
                theta_deg=cur_theta,
                pol=_pol(cur_pol),
                E_eV=E,
                R=R,
                bounds=[(float(E.min()), float(E.max()))],
                weights=[1.0],
            )
        )
        cur_scan = cur_pol = cur_theta = None
        E_vals, R_vals = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("="):
                flush()
                continue

            parts = s.split()
            # Header: "<scan> <S|P> E <E0> <theta>"
            if len(parts) >= 5:
                try:
                    maybe_scan = int(parts[0])
                    if parts[1].upper() in ("S", "P") and parts[2].upper() == "E":
                        flush()
                        cur_scan = maybe_scan
                        cur_pol = parts[1]
                        # parts[3] is initial energy (not strictly needed)
                        cur_theta = float(parts[4])
                        continue
                except ValueError:
                    pass

            # Data: energy angle qz R  (we use energy, R; angle is constant)
            if len(parts) >= 4:
                try:
                    E_vals.append(float(parts[0]))
                    # angle = float(parts[1])  # could be checked against cur_theta
                    # qz = float(parts[2])     # available if needed
                    R_vals.append(float(parts[3]))
                except ValueError:
                    pass

    flush()
    return scans
