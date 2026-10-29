#!/usr/bin/env python3
"""
cg_structure_factor.py

Reciprocal-space static structure-factor analysis for coarse-grained (CG)
lyotropic liquid-crystalline and self-assembled systems.

Default equal-weight definition:

    S(q) = |sum_j exp(i q . r_j)|^2 / N

Optional scalar-weight definition:

    S(q) = |sum_j w_j exp(i q . r_j)|^2 / sum_j w_j^2

Scientific scope
----------------
The default equal-weight calculation is a coarse-grained geometric/static
structure factor intended for reciprocal-lattice analysis, Bragg-peak
positions, characteristic periodicities, and mesophase characterization.
It is not, by itself, an absolute experimental SAXS-intensity calculation,
because atomistic X-ray form factors and electron-density contrast are not
included.

Input can be a standard PDB or a GROMACS-style multi-model PDB trajectory.
Each PDB file is treated as one physical system; MODEL/ENDMDL records inside
that PDB are treated as time frames of the same system.

Coordinates are assumed to be in Angstrom, as in the PDB standard; q is
therefore reported in Angstrom^-1.

The optional phase labels produced by the program are reciprocal-space
diagnostic hints only. Final assignments should be based on the averaged
S(q), Bragg vectors, and representative real-space configurations.
"""
import json
import math
import re
import sys
import platform
import argparse
import itertools
from pathlib import Path

import numpy as np
import taichi as ti
from tqdm import tqdm


__version__ = "3.0.0"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def comma_list(text):
    if text is None or str(text).strip() == "":
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Reciprocal-space static structure factor S(q) for coarse-grained "
            "single- or multi-model PDB trajectories"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}"
    )
    p.add_argument(
        "-d", "--data-path", required=True,
        help="Directory containing PDB file(s), or one multi-model PDB file"
    )
    p.add_argument("-q", "--qmax", required=True, type=float,
                   help="Maximum |q| in A^-1")
    p.add_argument("--qmin", default=0.0, type=float,
                   help="Minimum |q| in A^-1 (default: 0)")
    p.add_argument("-b", "--bin-size", default=0.005, type=float,
                   help="Radial q-bin width in A^-1 (default: 0.005)")
    p.add_argument("-o", "--outdir", default=None,
                   help="Output directory (default: DATA_PATH/DATA)")

    p.add_argument("--arch", choices=["cuda", "cpu"], default="cuda",
                   help="Taichi backend (default: cuda)")
    p.add_argument("--q-chunk", type=int, default=4096,
                   help="q vectors per Taichi kernel launch (default: 4096)")

    p.add_argument("--water-names", default="W",
                   help="Comma-separated water bead names (default: W)")
    p.add_argument("--include-water", action="store_true",
                   help="Include water beads; by default water beads are excluded")
    p.add_argument("--include-beads", default=None,
                   help="Only include these comma-separated bead names")
    p.add_argument("--exclude-beads", default=None,
                   help="Exclude these comma-separated bead names")

    p.add_argument("--weights-json", default=None,
                   help="Optional JSON mapping bead_name -> scalar scattering weight")
    p.add_argument("--unknown-weight", type=float, default=None,
                   help="Fallback weight for bead names absent from weights JSON")

    p.add_argument("--peak-qmax", type=float, default=0.60,
                   help="Upper q limit for Bragg/diagnostic analysis")
    p.add_argument("--top-peaks", type=int, default=200,
                   help="Number of Friedel-unique peaks saved per frame")
    p.add_argument("--first-shell-rel-width", type=float, default=0.08,
                   help="Relative width for split first shell (default: 0.08)")
    p.add_argument("--first-shell-intensity-frac", type=float, default=0.20,
                   help="Strong-direction cutoff relative to shell maximum")
    p.add_argument("--significant-frac", type=float, default=0.10,
                   help="Fraction of strongest low-q peak used to locate first shell")

    p.add_argument("--frame-stride", type=int, default=1,
                   help="Analyze every Nth MODEL frame (default: 1)")
    p.add_argument("--start-frame", type=int, default=1,
                   help="First MODEL index to analyze, 1-based (default: 1)")
    p.add_argument("--stop-frame", type=int, default=None,
                   help="Last MODEL index to analyze, inclusive")

    p.add_argument("--write-raw-3d", action="store_true",
                   help="Write complete h,k,l,qx,qy,qz,S(q) table for every frame")
    p.add_argument("--no-frame-radial", action="store_true",
                   help="Do not save a radial S(q) file for every individual frame")

    p.add_argument("--hex-angle-min", type=float, default=50.0,
                   help="Minimum Friedel-equivalent H1 first-shell angle")
    p.add_argument("--hex-angle-max", type=float, default=70.0,
                   help="Maximum Friedel-equivalent H1 first-shell angle")
    p.add_argument("--hex-q-spread", type=float, default=0.12,
                   help="Maximum relative length spread of H1 first-shell axes")
    p.add_argument("--lam-angle-tol", type=float, default=8.0,
                   help="Maximum angular deviation for lamellar harmonics")
    p.add_argument("--lam-ratio-tol", type=float, default=0.10,
                   help="Relative tolerance around integer lamellar q ratios")

    return p


# ---------------------------------------------------------------------------
# Cell / PDB
# ---------------------------------------------------------------------------

def cell_matrix_from_cryst1(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    sg = math.sin(gamma)
    if abs(sg) < 1e-8:
        raise ValueError("Invalid CRYST1 cell: sin(gamma) is zero")

    avec = np.array([a, 0.0, 0.0], dtype=np.float64)
    bvec = np.array([b * math.cos(gamma), b * sg, 0.0], dtype=np.float64)

    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sg
    cz2 = c*c - cx*cx - cy*cy
    if cz2 < -1e-6:
        raise ValueError("Invalid CRYST1 cell: negative c_z^2")
    cz = math.sqrt(max(0.0, cz2))
    cvec = np.array([cx, cy, cz], dtype=np.float64)
    return np.column_stack([avec, bvec, cvec])


def parse_cryst1(line):
    try:
        a = float(line[6:15])
        b = float(line[15:24])
        c = float(line[24:33])
        alpha = float(line[33:40])
        beta = float(line[40:47])
        gamma = float(line[47:54])
    except Exception:
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"Cannot parse CRYST1 line: {line.rstrip()}")
        a, b, c, alpha, beta, gamma = map(float, parts[1:7])
    params = (a, b, c, alpha, beta, gamma)
    return cell_matrix_from_cryst1(*params), params


def parse_title_metadata(title):
    time_ps = float("nan")
    step = -1
    if title:
        mt = re.search(r"\bt\s*=\s*([0-9eE+\-.]+)", title)
        ms = re.search(r"\bstep\s*=\s*([0-9]+)", title)
        if mt:
            try:
                time_ps = float(mt.group(1))
            except ValueError:
                pass
        if ms:
            try:
                step = int(ms.group(1))
            except ValueError:
                pass
    return time_ps, step


def _selected_atom(line, include_water, water_names, include_beads, exclude_beads):
    bead = line[12:16].strip()
    if not bead:
        parts = line.split()
        if len(parts) < 3:
            return None
        bead = parts[2]

    if (not include_water) and bead in water_names:
        return None
    if include_beads and bead not in include_beads:
        return None
    if bead in exclude_beads:
        return None

    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except Exception as exc:
        raise ValueError(f"Cannot parse coordinates: {line.rstrip()}") from exc
    return bead, (x, y, z)


def iter_pdb_frames(path, include_water, water_names, include_beads, exclude_beads):
    """
    Yield frames from either:
      * standard single-model PDB
      * GROMACS multi-model PDB containing MODEL/ENDMDL

    CRYST1 and TITLE immediately preceding MODEL are attached to that frame.
    """
    path = Path(path)
    pending_cell = None
    pending_cell_params = None
    pending_title = ""
    current = None
    saw_model = False
    implicit_index = 0

    def start_frame(model_index):
        return {
            "model_index": model_index,
            "cell": pending_cell,
            "cell_params": pending_cell_params,
            "title": pending_title,
            "coords": [],
            "beads": [],
        }

    def finalize(frame):
        if frame is None:
            return None
        if frame["cell"] is None:
            raise ValueError(f"No CRYST1 available for frame {frame['model_index']} in {path}")
        if not frame["coords"]:
            raise ValueError(f"No selected beads in frame {frame['model_index']} of {path}")
        time_ps, step = parse_title_metadata(frame["title"])
        return {
            "model_index": frame["model_index"],
            "cell": frame["cell"],
            "cell_params": frame["cell_params"],
            "title": frame["title"],
            "time_ps": time_ps,
            "step": step,
            "coords": np.asarray(frame["coords"], dtype=np.float32),
            "bead_names": np.asarray(frame["beads"], dtype=object),
        }

    with open(path, "r", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            rec = line[0:6].strip()

            if rec == "TITLE":
                pending_title = line.rstrip("\n")
                if current is not None and not current["coords"]:
                    current["title"] = pending_title
                continue

            if rec == "CRYST1":
                pending_cell, pending_cell_params = parse_cryst1(line)
                if current is not None and not current["coords"]:
                    current["cell"] = pending_cell
                    current["cell_params"] = pending_cell_params
                continue

            if rec == "MODEL":
                saw_model = True
                if current is not None:
                    obj = finalize(current)
                    if obj is not None:
                        yield obj
                txt = line[10:14].strip()
                model_index = int(txt) if txt else implicit_index + 1
                current = start_frame(model_index)
                continue

            if rec == "ENDMDL":
                obj = finalize(current)
                if obj is not None:
                    yield obj
                current = None
                continue

            if rec not in ("ATOM", "HETATM"):
                continue

            if current is None:
                if saw_model:
                    continue
                implicit_index += 1 if implicit_index == 0 else 0
                current = start_frame(1)

            try:
                parsed = _selected_atom(
                    line, include_water, water_names, include_beads, exclude_beads
                )
            except Exception as exc:
                raise ValueError(
                    f"{path}, line {line_no}: {line.rstrip()}"
                ) from exc
            if parsed is None:
                continue
            bead, xyz = parsed
            current["beads"].append(bead)
            current["coords"].append(xyz)

    if current is not None:
        obj = finalize(current)
        if obj is not None:
            yield obj


# ---------------------------------------------------------------------------
# Reciprocal lattice
# ---------------------------------------------------------------------------

def reciprocal_matrix(cell):
    return 2.0 * np.pi * np.linalg.inv(cell).T


def generate_q_grid(cell, qmin, qmax):
    B = reciprocal_matrix(cell)
    bnorm = np.linalg.norm(B, axis=0)
    nmax = int(math.ceil(qmax / max(1e-12, float(np.min(bnorm))))) + 2

    r = np.arange(-nmax, nmax + 1, dtype=np.int32)
    h, k, l = np.meshgrid(r, r, r, indexing="ij")
    hkl = np.column_stack([h.ravel(), k.ravel(), l.ravel()]).astype(np.int32)
    q_vec = hkl.astype(np.float64) @ B.T
    q_abs = np.linalg.norm(q_vec, axis=1)
    mask = (q_abs > max(0.0, qmin)) & (q_abs <= qmax)

    return (
        np.ascontiguousarray(hkl[mask]),
        np.ascontiguousarray(q_vec[mask].astype(np.float32)),
        np.ascontiguousarray(q_abs[mask].astype(np.float32)),
        B,
    )


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def load_weight_map(path):
    if path is None:
        return None
    with open(path, "r") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError("weights JSON must be {bead_name: weight, ...}")
    clean = {}
    for k, v in obj.items():
        if str(k).startswith("_"):
            continue
        clean[str(k)] = float(v)
    return clean


def make_weights(bead_names, weight_map, unknown_weight):
    if weight_map is None:
        return np.ones(len(bead_names), dtype=np.float32)

    out = np.empty(len(bead_names), dtype=np.float32)
    missing = set()
    for i, name in enumerate(bead_names):
        name = str(name)
        aliases = [name]
        if name == "CO2":
            aliases.append("COO")
        elif name == "COO":
            aliases.append("CO2")

        found = False
        for alias in aliases:
            if alias in weight_map:
                out[i] = weight_map[alias]
                found = True
                break
        if found:
            continue

        if unknown_weight is not None:
            out[i] = unknown_weight
        else:
            missing.add(name)

    if missing:
        raise ValueError(
            "Missing bead weights: " + ", ".join(sorted(missing))
        )
    return out


# ---------------------------------------------------------------------------
# Taichi S(q)
# ---------------------------------------------------------------------------

@ti.kernel
def ft_weighted_kernel(
    q_vec: ti.types.ndarray(dtype=ti.f32, ndim=2),
    coords: ti.types.ndarray(dtype=ti.f32, ndim=2),
    weights: ti.types.ndarray(dtype=ti.f32, ndim=1),
    cos_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    sin_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in range(q_vec.shape[0]):
        c = 0.0
        s = 0.0
        for j in range(coords.shape[0]):
            phase = (
                q_vec[i, 0] * coords[j, 0]
                + q_vec[i, 1] * coords[j, 1]
                + q_vec[i, 2] * coords[j, 2]
            )
            w = weights[j]
            c += w * ti.cos(phase)
            s += w * ti.sin(phase)
        cos_out[i] = c
        sin_out[i] = s


def calculate_sq(q_vec, coords, weights, chunk_size):
    nq = q_vec.shape[0]
    sq = np.empty(nq, dtype=np.float32)
    norm = float(np.sum(weights.astype(np.float64) ** 2))
    if norm <= 0:
        raise ValueError("sum(weights^2) is zero")

    for start in tqdm(range(0, nq, chunk_size), desc="q chunks", leave=False):
        stop = min(start + chunk_size, nq)
        qch = np.ascontiguousarray(q_vec[start:stop], dtype=np.float32)
        c = np.zeros(stop-start, dtype=np.float32)
        s = np.zeros(stop-start, dtype=np.float32)
        ft_weighted_kernel(qch, coords, weights, c, s)
        sq[start:stop] = (c*c + s*s) / norm
    return sq


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def radial_average(q_abs, sq, qmax, bin_size):
    nb = int(math.ceil(qmax / bin_size))
    idx = np.floor(q_abs / bin_size).astype(np.int64)
    valid = (idx >= 0) & (idx < nb)

    sums = np.zeros(nb, dtype=np.float64)
    sums2 = np.zeros(nb, dtype=np.float64)
    counts = np.zeros(nb, dtype=np.int64)

    np.add.at(sums, idx[valid], sq[valid])
    np.add.at(sums2, idx[valid], sq[valid].astype(np.float64)**2)
    np.add.at(counts, idx[valid], 1)

    mean = np.full(nb, np.nan)
    std = np.full(nb, np.nan)
    nz = counts > 0
    mean[nz] = sums[nz] / counts[nz]
    var = np.maximum(0.0, sums2[nz]/counts[nz] - mean[nz]**2)
    std[nz] = np.sqrt(var)
    centers = (np.arange(nb, dtype=np.float64)+0.5)*bin_size
    return centers, mean, std, counts


def canonical_friedel(hkl):
    x = np.asarray(hkl, dtype=np.int32)
    for v in x:
        if v > 0:
            return tuple(int(z) for z in x)
        if v < 0:
            return tuple(int(-z) for z in x)
    return (0, 0, 0)


def unique_friedel_peaks(hkl, q_vec, q_abs, sq, qmax, top_n):
    best = {}
    for i in range(len(sq)):
        if q_abs[i] > qmax:
            continue
        key = canonical_friedel(hkl[i])
        old = best.get(key)
        if old is None or sq[i] > sq[old]:
            best[key] = i
    inds = sorted(best.values(), key=lambda i: float(sq[i]), reverse=True)
    return inds[:top_n]


def angle_deg_mod_friedel(v1, v2):
    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    c = abs(float(np.dot(a, b)/(na*nb)))
    c = min(1.0, max(-1.0, c))
    return math.degrees(math.acos(c))


def first_shell_report(hkl, q_vec, q_abs, sq, peak_qmax,
                       significant_frac, rel_width, intensity_frac):
    inds = unique_friedel_peaks(hkl, q_vec, q_abs, sq, peak_qmax, top_n=1000)
    if not inds:
        return None

    smax = max(float(sq[i]) for i in inds)
    significant = [i for i in inds if float(sq[i]) >= significant_frac*smax]
    if not significant:
        return None

    q0 = min(float(q_abs[i]) for i in significant)
    lo = q0*(1.0-rel_width)
    hi = q0*(1.0+rel_width)

    shell = [i for i in inds if lo <= float(q_abs[i]) <= hi]
    if not shell:
        return None

    shell_max = max(float(sq[i]) for i in shell)
    strong = [i for i in shell if float(sq[i]) >= intensity_frac*shell_max]
    strong = sorted(strong, key=lambda i: float(sq[i]), reverse=True)

    angles = []
    for a in range(len(strong)):
        for b in range(a+1, len(strong)):
            angles.append((a, b, angle_deg_mod_friedel(
                q_vec[strong[a]], q_vec[strong[b]]
            )))

    return {
        "q0": q0,
        "shell_lo": lo,
        "shell_hi": hi,
        "indices": strong,
        "angles": angles,
    }


def min_axis_closure(hkls):
    """
    For three Friedel-equivalent reciprocal axes, test whether signs can be
    chosen such that +/-h1 +/-h2 +/-h3 = 0. Returns the minimum Euclidean
    norm in integer hkl space and the corresponding signs.
    """
    best = None
    vecs = [np.asarray(x, dtype=np.int32) for x in hkls]
    for signs in itertools.product((-1, 1), repeat=3):
        s = signs[0]*vecs[0] + signs[1]*vecs[1] + signs[2]*vecs[2]
        n = float(np.linalg.norm(s))
        if best is None or n < best[0]:
            best = (n, signs, s)
    return best


def find_hexagonal_triplet(report, hkl, q_vec, q_abs,
                            angle_min, angle_max, q_spread_tol):
    if report is None or len(report["indices"]) < 3:
        return None

    cand = report["indices"][:8]
    best = None
    for triple in itertools.combinations(cand, 3):
        qs = np.asarray([float(q_abs[i]) for i in triple])
        rel_spread = (float(np.max(qs))-float(np.min(qs))) / float(np.mean(qs))
        if rel_spread > q_spread_tol:
            continue

        angs = [
            angle_deg_mod_friedel(q_vec[triple[0]], q_vec[triple[1]]),
            angle_deg_mod_friedel(q_vec[triple[0]], q_vec[triple[2]]),
            angle_deg_mod_friedel(q_vec[triple[1]], q_vec[triple[2]]),
        ]
        if not all(angle_min <= a <= angle_max for a in angs):
            continue

        closure_norm, signs, residual = min_axis_closure([hkl[i] for i in triple])
        if closure_norm > 1e-9:
            continue

        score = (
            rel_spread
            + sum(abs(a-60.0) for a in angs)/180.0
        )
        item = {
            "indices": triple,
            "angles": angs,
            "rel_spread": rel_spread,
            "closure_norm": closure_norm,
            "closure_signs": signs,
            "q_mean": float(np.mean(qs)),
        }
        if best is None or score < best[0]:
            best = (score, item)

    return None if best is None else best[1]


def find_lamellar_harmonics(primary_i, peak_inds, q_vec, q_abs, sq,
                             angle_tol, ratio_tol):
    q1 = float(q_abs[primary_i])
    s1 = float(sq[primary_i])
    found = []

    for n in (2, 3, 4, 5):
        target = n*q1
        for i in peak_inds:
            if i == primary_i:
                continue
            ratio_err = abs(float(q_abs[i])-target) / max(target, 1e-12)
            if ratio_err > ratio_tol:
                continue
            ang = angle_deg_mod_friedel(q_vec[primary_i], q_vec[i])
            if ang > angle_tol:
                continue
            if float(sq[i]) < 0.02*s1:
                continue
            found.append({
                "n": n,
                "index": i,
                "q": float(q_abs[i]),
                "Sq": float(sq[i]),
                "angle": ang,
                "ratio_error": ratio_err,
            })
            break
    return found


def classify_frame(report, hkl, q_vec, q_abs, sq, peak_qmax, top_peaks,
                   hex_angle_min, hex_angle_max, hex_q_spread,
                   lam_angle_tol, lam_ratio_tol):
    peak_inds = unique_friedel_peaks(
        hkl, q_vec, q_abs, sq, peak_qmax, top_n=max(top_peaks, 500)
    )
    if not peak_inds or report is None or not report["indices"]:
        return {
            "phase": "disordered/micellar-like",
            "confidence": "low",
            "q_primary": float("nan"),
            "n_first_shell_axes": 0,
            "hex": None,
            "lam_harmonics": [],
        }

    primary = report["indices"][0]

    hexfit = find_hexagonal_triplet(
        report, hkl, q_vec, q_abs,
        hex_angle_min, hex_angle_max, hex_q_spread
    )

    lam_h = find_lamellar_harmonics(
        primary, peak_inds, q_vec, q_abs, sq,
        lam_angle_tol, lam_ratio_tol
    )

    n_axes = len(report["indices"])
    q_primary = float(np.mean([q_abs[i] for i in report["indices"][:max(1, min(3, n_axes))]]))

    if hexfit is not None:
        return {
            "phase": "H1-hexagonal",
            "confidence": "high",
            "q_primary": hexfit["q_mean"],
            "n_first_shell_axes": n_axes,
            "hex": hexfit,
            "lam_harmonics": lam_h,
        }

    if lam_h and n_axes <= 2:
        conf = "high" if len(lam_h) >= 1 else "medium"
        return {
            "phase": "lamellar",
            "confidence": conf,
            "q_primary": float(q_abs[primary]),
            "n_first_shell_axes": n_axes,
            "hex": None,
            "lam_harmonics": lam_h,
        }

    # If neither reciprocal-lattice signature is stable, call it micellar-like
    # but retain a conservative confidence label.
    return {
        "phase": "disordered/micellar-like",
        "confidence": "medium" if n_axes >= 3 else "low",
        "q_primary": q_primary,
        "n_first_shell_axes": n_axes,
        "hex": None,
        "lam_harmonics": lam_h,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_raw_3d(path, hkl, q_vec, q_abs, sq):
    with open(path, "w") as out:
        out.write("# h k l qx[A^-1] qy[A^-1] qz[A^-1] q[A^-1] Sq\n")
        for n, qv, q, s in zip(hkl, q_vec, q_abs, sq):
            out.write(
                f"{int(n[0]):5d} {int(n[1]):5d} {int(n[2]):5d} "
                f"{qv[0]:12.7f} {qv[1]:12.7f} {qv[2]:12.7f} "
                f"{q:12.7f} {s:16.8g}\n"
            )


def write_radial(path, centers, mean, std, counts):
    with open(path, "w") as out:
        out.write("# q[A^-1] Sq_mean Sq_std n_qvectors\n")
        for q, m, sd, n in zip(centers, mean, std, counts):
            if n <= 0:
                continue
            out.write(f"{q:10.5f} {m:16.8g} {sd:16.8g} {int(n):8d}\n")


def write_bragg(path, inds, hkl, q_vec, q_abs, sq):
    with open(path, "w") as out:
        out.write("# rank h k l q[A^-1] Sq qx qy qz\n")
        for rank, i in enumerate(inds, 1):
            n = hkl[i]
            qv = q_vec[i]
            out.write(
                f"{rank:5d} {int(n[0]):5d} {int(n[1]):5d} {int(n[2]):5d} "
                f"{q_abs[i]:12.7f} {sq[i]:16.8g} "
                f"{qv[0]:12.7f} {qv[1]:12.7f} {qv[2]:12.7f}\n"
            )


def write_first_shell(path, report, classification, hkl, q_vec, q_abs, sq):
    with open(path, "w") as out:
        if report is None:
            out.write("No first-shell report could be generated.\n")
            out.write(f"Phase hint: {classification['phase']}\n")
            return

        out.write(f"First significant shell q0 ~ {report['q0']:.7f} A^-1\n")
        out.write(
            f"Grouping window: {report['shell_lo']:.7f} - "
            f"{report['shell_hi']:.7f} A^-1\n"
        )
        out.write(
            f"Phase hint: {classification['phase']} "
            f"(confidence={classification['confidence']})\n\n"
        )

        out.write("Strong Friedel-unique directions in first shell:\n")
        out.write("idx   h   k   l      q[A^-1]            Sq       qx        qy        qz\n")
        for local_idx, i in enumerate(report["indices"], 1):
            n = hkl[i]
            qv = q_vec[i]
            out.write(
                f"{local_idx:3d} {int(n[0]):4d} {int(n[1]):4d} {int(n[2]):4d} "
                f"{q_abs[i]:12.7f} {sq[i]:14.6g} "
                f"{qv[0]:10.6f} {qv[1]:10.6f} {qv[2]:10.6f}\n"
            )

        if report["angles"]:
            out.write("\nPair angles (Friedel-equivalent, 0-90 deg):\n")
            for a, b, ang in report["angles"]:
                out.write(f"direction {a+1} - {b+1}: {ang:.3f} deg\n")

        if classification["hex"] is not None:
            hfit = classification["hex"]
            out.write("\nH1 reciprocal-lattice triplet:\n")
            out.write(
                f"mean q10 = {hfit['q_mean']:.7f} A^-1\n"
                f"relative q spread = {100*hfit['rel_spread']:.3f}%\n"
                f"angles = {', '.join(f'{a:.3f}' for a in hfit['angles'])} deg\n"
                f"closure signs = {hfit['closure_signs']}\n"
                f"closure residual = {hfit['closure_norm']:.6g}\n"
            )

        if classification["lam_harmonics"]:
            out.write("\nLamellar-like collinear harmonics:\n")
            for x in classification["lam_harmonics"]:
                out.write(
                    f"n={x['n']} q={x['q']:.7f} Sq={x['Sq']:.6g} "
                    f"angle={x['angle']:.3f} deg "
                    f"ratio_error={100*x['ratio_error']:.3f}%\n"
                )


def write_system_radial_average(path, centers, frame_means):
    arr = np.asarray(frame_means, dtype=np.float64)
    nframe = np.sum(np.isfinite(arr), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=0)
        sem = np.full_like(mean, np.nan)
        ok = nframe > 0
        sem[ok] = sd[ok]/np.sqrt(nframe[ok])

    with open(path, "w") as out:
        out.write("# q[A^-1] Sq_frame_mean Sq_frame_SD Sq_frame_SEM n_frames\n")
        for q, m, s, se, n in zip(centers, mean, sd, sem, nframe):
            if n <= 0:
                continue
            out.write(
                f"{q:10.5f} {m:16.8g} {s:16.8g} "
                f"{se:16.8g} {int(n):8d}\n"
            )



def write_run_metadata(path, args, pdb_files):
    """Write settings and software versions used for reproducibility."""
    metadata = {
        "program": "cg_structure_factor.py",
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "taichi": getattr(ti, "__version__", "unknown"),
        "input_pdbs": [str(p) for p in pdb_files],
        "parameters": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "scientific_note": (
            "Default equal-weight mode is a coarse-grained geometric/static "
            "structure factor, not an absolute experimental SAXS intensity."
        ),
    }
    with open(path, "w") as out:
        json.dump(metadata, out, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_pdbs(data_path):
    p = Path(data_path)
    if p.is_file():
        if p.suffix.lower() != ".pdb":
            raise ValueError("Input file must be .pdb")
        return [p], p.parent
    if not p.is_dir():
        raise ValueError(f"Not found: {p}")
    files = sorted(p.glob("*.pdb"))
    if not files:
        raise ValueError(f"No *.pdb files found in {p}")
    return files, p


def selected_frame_number(model_index, start, stop, stride):
    if model_index < start:
        return False
    if stop is not None and model_index > stop:
        return False
    return ((model_index-start) % stride) == 0


def main():
    args = build_parser().parse_args()

    print(f"cg_structure_factor.py v{__version__}")
    print("Mode: coarse-grained geometric/static structure factor")

    if args.qmax <= 0 or args.bin_size <= 0 or args.q_chunk <= 0:
        raise ValueError("qmax, bin-size, and q-chunk must be positive")
    if args.qmin < 0 or args.qmin >= args.qmax:
        raise ValueError("Require 0 <= qmin < qmax")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be >= 1")

    pdb_files, base_dir = discover_pdbs(args.data_path)
    outroot = Path(args.outdir) if args.outdir else base_dir / "DATA"
    outroot.mkdir(parents=True, exist_ok=True)

    water_names = set(comma_list(args.water_names))
    include_beads = set(comma_list(args.include_beads))
    exclude_beads = set(comma_list(args.exclude_beads))
    weight_map = load_weight_map(args.weights_json)

    arch = ti.cuda if args.arch == "cuda" else ti.cpu
    ti.init(arch=arch)

    write_run_metadata(outroot / "run_parameters.json", args, pdb_files)

    global_summary = []

    for pdb_path in pdb_files:
        system_name = pdb_path.stem
        sysdir = outroot / system_name
        framedir = sysdir / "frames"
        sysdir.mkdir(parents=True, exist_ok=True)
        framedir.mkdir(parents=True, exist_ok=True)

        all_frames = list(iter_pdb_frames(
            pdb_path,
            include_water=args.include_water,
            water_names=water_names,
            include_beads=include_beads,
            exclude_beads=exclude_beads,
        ))
        frames = [
            f for f in all_frames
            if selected_frame_number(
                f["model_index"], args.start_frame, args.stop_frame, args.frame_stride
            )
        ]
        if not frames:
            print(f"WARNING: no selected frames in {pdb_path}")
            continue

        print(f"\n=== {system_name}: {len(frames)} selected frame(s) ===")

        frame_radials = []
        radial_centers = None
        phase_rows = []
        cell_params_list = []
        bead_counts = []

        for fr in tqdm(frames, desc=system_name):
            model = fr["model_index"]
            coords = fr["coords"]
            bead_names = fr["bead_names"]
            cell = fr["cell"]
            cell_params = fr["cell_params"]
            weights = make_weights(bead_names, weight_map, args.unknown_weight)

            hkl, q_vec, q_abs, _ = generate_q_grid(cell, args.qmin, args.qmax)
            sq = calculate_sq(q_vec, coords, weights, args.q_chunk)

            centers, rmean, rstd, rcount = radial_average(
                q_abs, sq, args.qmax, args.bin_size
            )
            radial_centers = centers
            frame_radials.append(rmean)
            cell_params_list.append(cell_params)
            bead_counts.append(len(bead_names))

            if not args.no_frame_radial:
                write_radial(
                    framedir / f"SqRadial-frame{model:04d}.dat",
                    centers, rmean, rstd, rcount
                )

            if args.write_raw_3d:
                write_raw_3d(
                    framedir / f"Sq3D-frame{model:04d}.dat",
                    hkl, q_vec, q_abs, sq
                )

            peak_inds = unique_friedel_peaks(
                hkl, q_vec, q_abs, sq,
                qmax=min(args.peak_qmax, args.qmax),
                top_n=args.top_peaks,
            )
            write_bragg(
                framedir / f"Bragg-frame{model:04d}.dat",
                peak_inds, hkl, q_vec, q_abs, sq
            )

            report = first_shell_report(
                hkl, q_vec, q_abs, sq,
                peak_qmax=min(args.peak_qmax, args.qmax),
                significant_frac=args.significant_frac,
                rel_width=args.first_shell_rel_width,
                intensity_frac=args.first_shell_intensity_frac,
            )

            cls = classify_frame(
                report, hkl, q_vec, q_abs, sq,
                peak_qmax=min(args.peak_qmax, args.qmax),
                top_peaks=args.top_peaks,
                hex_angle_min=args.hex_angle_min,
                hex_angle_max=args.hex_angle_max,
                hex_q_spread=args.hex_q_spread,
                lam_angle_tol=args.lam_angle_tol,
                lam_ratio_tol=args.lam_ratio_tol,
            )

            write_first_shell(
                framedir / f"FirstShell-frame{model:04d}.txt",
                report, cls, hkl, q_vec, q_abs, sq
            )

            q_primary = cls["q_primary"]
            lam_d_nm = float("nan")
            hex_a_nm = float("nan")
            if np.isfinite(q_primary) and q_primary > 0:
                if cls["phase"] == "lamellar":
                    lam_d_nm = (2.0*np.pi/q_primary)/10.0
                elif cls["phase"] == "H1-hexagonal":
                    hex_a_nm = (4.0*np.pi/(math.sqrt(3.0)*q_primary))/10.0

            phase_rows.append({
                "model": model,
                "time_ps": fr["time_ps"],
                "step": fr["step"],
                "phase": cls["phase"],
                "confidence": cls["confidence"],
                "q_primary": q_primary,
                "n_axes": cls["n_first_shell_axes"],
                "lam_d_nm": lam_d_nm,
                "hex_a_nm": hex_a_nm,
            })

        # Same-system radial average across MODEL frames
        write_system_radial_average(
            sysdir / "SqRadial-average.dat",
            radial_centers,
            frame_radials,
        )

        # Per-frame phase table
        with open(sysdir / "Phase-frame-summary.tsv", "w") as out:
            out.write(
                "model\ttime_ps\tstep\tphase\tconfidence\tq_primary_A^-1\t"
                "n_first_shell_axes\tlamellar_d_nm\thexagonal_a_nm\n"
            )
            for r in phase_rows:
                out.write(
                    f"{r['model']}\t{r['time_ps']:.6g}\t{r['step']}\t"
                    f"{r['phase']}\t{r['confidence']}\t"
                    f"{r['q_primary']:.7g}\t{r['n_axes']}\t"
                    f"{r['lam_d_nm']:.7g}\t{r['hex_a_nm']:.7g}\n"
                )

        # Aggregate classification
        phases = [r["phase"] for r in phase_rows]
        unique, counts = np.unique(phases, return_counts=True)
        phase_counts = dict(zip(unique.tolist(), counts.tolist()))
        winner = max(phase_counts, key=phase_counts.get)
        winner_fraction = phase_counts[winner] / len(phases)

        qvals = np.asarray(
            [r["q_primary"] for r in phase_rows if np.isfinite(r["q_primary"])],
            dtype=float
        )

        lam_d = np.asarray(
            [r["lam_d_nm"] for r in phase_rows if np.isfinite(r["lam_d_nm"])],
            dtype=float
        )
        hex_a = np.asarray(
            [r["hex_a_nm"] for r in phase_rows if np.isfinite(r["hex_a_nm"])],
            dtype=float
        )

        cells = np.asarray(cell_params_list, dtype=float)
        with open(sysdir / "System-summary.txt", "w") as out:
            out.write(f"Program: cg_structure_factor.py v{__version__}\n")
            out.write("Definition: coarse-grained geometric/static structure factor\n")
            out.write(f"System: {system_name}\n")
            out.write(f"Input: {pdb_path}\n")
            out.write(f"Frames analyzed: {len(frames)}\n")
            out.write(f"Water included: {args.include_water}\n")
            out.write(f"Water names: {sorted(water_names)}\n")
            out.write(f"Selected bead count/frame: {sorted(set(bead_counts))}\n")
            out.write(
                "Cell length ranges [A]: "
                f"a={cells[:,0].min():.3f}-{cells[:,0].max():.3f}, "
                f"b={cells[:,1].min():.3f}-{cells[:,1].max():.3f}, "
                f"c={cells[:,2].min():.3f}-{cells[:,2].max():.3f}\n\n"
            )

            out.write("Per-frame phase counts:\n")
            for k in sorted(phase_counts):
                out.write(f"  {k}: {phase_counts[k]}/{len(phases)}\n")
            out.write(
                f"\nConsensus phase hint: {winner} "
                f"({100*winner_fraction:.1f}% of analyzed frames)\n"
            )

            if len(qvals):
                out.write(
                    f"Primary q: mean={np.mean(qvals):.7f} A^-1, "
                    f"SD={np.std(qvals):.7f} A^-1\n"
                )
            if len(lam_d):
                out.write(
                    f"Lamellar d: mean={np.mean(lam_d):.5f} nm, "
                    f"SD={np.std(lam_d):.5f} nm\n"
                )
            if len(hex_a):
                out.write(
                    f"H1 lattice a: mean={np.mean(hex_a):.5f} nm, "
                    f"SD={np.std(hex_a):.5f} nm\n"
                )

            out.write(
                "\nNOTE: phase labels are reciprocal-space structural hints only. "
                "They are not a substitute for inspection of the averaged S(q), "
                "Bragg/FirstShell outputs, and representative configurations. "
                "In equal-weight mode the result is a CG geometric structure factor, "
                "not an absolute experimental SAXS intensity.\n"
            )

        global_summary.append(
            (system_name, len(frames), winner, winner_fraction, phase_counts)
        )

    with open(outroot / "All-systems-summary.tsv", "w") as out:
        out.write("system\tn_frames\tconsensus_phase\tfraction\tphase_counts\n")
        for system_name, nframes, winner, frac, counts in global_summary:
            out.write(
                f"{system_name}\t{nframes}\t{winner}\t{frac:.6f}\t"
                f"{json.dumps(counts, ensure_ascii=False)}\n"
            )

    print(f"\nDone. Results: {outroot}")
    print("Each multi-model PDB is treated as an independent physical system.")
    print("Its MODEL frames are averaged only within that system.")


if __name__ == "__main__":
    main()
