# CG Structure Factor

GPU-accelerated reciprocal-space static structure-factor analysis for coarse-grained (CG) self-assembled systems.

The code was developed for coarse-grained lyotropic liquid-crystalline and surfactant systems, including lamellar, hexagonal-columnar (H1), and disordered/micellar morphologies.

## Scientific scope

By default, the program calculates the equal-weight geometric structure factor

The equations below use GitHub-compatible math delimiters (`$...$` for inline math and `$$...$$` for display math).

$$
S(\mathbf q)=\frac{1}{N}\left|\sum_{j=1}^{N}
\exp(i\mathbf q\cdot\mathbf r_j)\right|^2 .
$$

This calculation is intended primarily for:

- reciprocal-lattice analysis;
- Bragg-peak positions;
- characteristic periodicities;
- comparison of ordered and disordered CG morphologies;
- lamellar repeat spacing;
- H1 lattice-parameter analysis.

**Important:** equal-weight CG `S(q)` is not an absolute experimental SAXS intensity. Atomistic X-ray form factors and electron-density contrast are not included. Peak positions and reciprocal-space organization are therefore the primary quantities to interpret. Relative peak intensities should not be directly equated with experimental SAXS intensities unless a physically justified bead-scattering model is supplied.

An optional scalar bead-weight mode is available through `--weights-json`.

## Main features

- Taichi-accelerated calculation on NVIDIA GPU (`--arch cuda`) or CPU (`--arch cpu`).
- Standard PDB and GROMACS-style multi-model PDB (`MODEL/ENDMDL`) input.
- Per-frame simulation cell read from `CRYST1`.
- Orthorhombic and triclinic reciprocal lattices.
- One PDB file = one independent physical system.
- Multiple `MODEL` records = time frames of the same physical system.
- Radial `S(q)` averaged only among frames of the same PDB/system.
- Friedel-equivalent Bragg-peak analysis.
- Auxiliary reciprocal-space diagnostics for lamellar and H1 structures.
- Water beads excluded by default.
- Optional bead selection and scalar weighting.
- Reproducibility metadata written to `run_parameters.json`.
- Full 3D reciprocal-space output available on request.

## Requirements

### Software

- Python 3.9 or newer
- NumPy
- Taichi 1.7.4
- tqdm

The reference development environment used Python 3.9 and Taichi 1.7.4.

### GPU mode

For `--arch cuda`, an NVIDIA GPU and a working NVIDIA driver are required.

If CUDA initialization fails, use

```bash
--arch cpu
```

instead. CPU calculations are considerably slower for large systems.

## Installation

Clone/download the repository and install the dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Check the installation:

```bash
python3 cg_structure_factor.py --version
python3 cg_structure_factor.py --help
```

## Recommended input format

A GROMACS-style multi-model PDB is recommended:

```text
TITLE ...
CRYST1 ...
MODEL        1
ATOM ...
ATOM ...
...
ENDMDL
TITLE ...
CRYST1 ...
MODEL        2
ATOM ...
...
ENDMDL
```

Each `MODEL` is analyzed as an independent time frame.

The `CRYST1` record is required because the allowed reciprocal-space vectors are derived from the instantaneous simulation cell.

PDB coordinates are assumed to be in Angstrom.

### Preparing a sampled trajectory with GROMACS

A trajectory can be sampled to a multi-model PDB with `gmx trjconv`:

```bash
gmx trjconv \
  -s md.tpr \
  -f md.xtc \
  -o sampled.pdb \
  -b START_PS \
  -e END_PS \
  -dt STRIDE_PS
```

Replace `START_PS`, `END_PS`, and `STRIDE_PS` with values appropriate for the equilibrated trajectory segment.

Because the calculation uses reciprocal-lattice vectors compatible with the periodic simulation cell, translation of a bead by an integer lattice vector does not change its phase factor. Special unwrapping is therefore not required for the reciprocal-space calculation itself, provided that each frame contains the correct `CRYST1` cell.

## Recommended directory layout

```text
project/
├── cg_structure_factor.py
├── README.md
├── METHODS.md
├── requirements.txt
├── run_example.sh
├── example_weights.json
└── pdb/
    ├── lamellar.pdb
    ├── hexagonal.pdb
    └── micellar.pdb
```

Different PDB files are treated as different physical systems and are never averaged together.

## Reference calculation

The settings used in the current workflow are:

```bash
python3 cg_structure_factor.py \
  --data-path ./pdb \
  --qmax 1.20 \
  --bin-size 0.005 \
  --peak-qmax 0.60 \
  --arch cuda
```

Equivalent short options:

```bash
python3 cg_structure_factor.py \
  -d ./pdb \
  -q 1.20 \
  -b 0.005 \
  --peak-qmax 0.60 \
  --arch cuda
```

By default, water bead `W` is excluded.

## Core command-line options

| Option | Meaning | Default |
|---|---|---|
| `-d`, `--data-path` | One PDB or a directory of PDB files | required |
| `-q`, `--qmax` | Maximum $|q|$, A$^{-1}$ | required |
| `--qmin` | Minimum $|q|$, A$^{-1}$ | `0.0` |
| `-b`, `--bin-size` | Radial q-bin width, A$^{-1}$ | `0.005` |
| `-o`, `--outdir` | Output directory | `DATA_PATH/SF_CG_v3` |
| `--arch` | `cuda` or `cpu` | `cuda` |
| `--q-chunk` | q vectors per Taichi kernel launch | `4096` |
| `--water-names` | Water bead names | `W` |
| `--include-water` | Include water beads | off |
| `--include-beads` | Analyze only selected bead names | all retained |
| `--exclude-beads` | Additional bead names to exclude | none |
| `--frame-stride` | Analyze every Nth MODEL frame | `1` |
| `--start-frame` | First MODEL index | `1` |
| `--stop-frame` | Last MODEL index | all |
| `--write-raw-3d` | Write all 3D q vectors and S(q) | off |
| `--no-frame-radial` | Do not save individual-frame radial files | off |
| `--peak-qmax` | Maximum q for Bragg diagnostics | `0.60` |
| `--weights-json` | Optional bead-to-weight JSON file | none |

Run

```bash
python3 cg_structure_factor.py --help
```

for all available diagnostic parameters.

## Water treatment

The reference workflow excludes bead `W` because the objective is to characterize the periodic organization of the surfactant framework.

To include water:

```bash
python3 cg_structure_factor.py \
  -d ./pdb \
  -q 1.20 \
  --include-water
```

For other water bead names:

```bash
--water-names W,WP
```

## Bead selection

Analyze only selected beads:

```bash
--include-beads SO3,NC4,C2
```

or exclude additional beads:

```bash
--exclude-beads CC2,C31
```

## Optional scalar bead weights

Without `--weights-json`, every selected bead has weight 1.

A JSON file can instead define scalar weights:

```json
{
  "CC2": 1.0,
  "COO": 1.0,
  "C31": 1.0,
  "C4": 1.0,
  "C32": 1.0,
  "NC4": 1.0,
  "C2": 1.0,
  "SO3": 1.0
}
```

Use it as:

```bash
python3 cg_structure_factor.py \
  -d ./pdb \
  -q 1.20 \
  --weights-json example_weights.json
```

The weighted definition is

$$
S(\mathbf q)=
\frac{\left|\sum_j w_j\exp(i\mathbf q\cdot\mathbf r_j)\right|^2}
{\sum_j w_j^2}.
$$

A physically meaningful X-ray comparison requires physically justified bead-scattering amplitudes. Arbitrary weights should not be used simply to reproduce experimental intensities.

`COO` and `CO2` are treated as aliases during weight lookup for compatibility with alternative structure/topology naming conventions.

## Reciprocal-space calculation

For every frame, the direct-cell matrix $\mathbf H$ is constructed from `CRYST1`. The reciprocal-cell matrix is

$$
\mathbf B=2\pi(\mathbf H^{-1})^{T}.
$$

Allowed scattering vectors are

$$
\mathbf q_{hkl}=\mathbf B
\begin{pmatrix}
h\\k\\l
\end{pmatrix},
$$

where $h,k,l$ are integers.

Only vectors satisfying the requested q range are retained, and $\mathbf q=0$ is excluded.

## Radial average

The one-dimensional curve is obtained by shell averaging:

$$
S(q)=
\left\langle S(\mathbf q)\right\rangle_{|\mathbf q|\in\mathrm{bin}} .
$$

Bins containing no reciprocal vector are omitted rather than assigned an artificial zero value.

For multi-model PDB files, `S(q)` is calculated independently for each frame and then averaged only among frames from the same PDB/system. The system-level output includes the mean, standard deviation, standard error, and number of contributing frames.

## Output structure

```text
SF_CG_v3/
├── run_parameters.json
├── All-systems-summary.tsv
├── lamellar/
│   ├── SqRadial-average.dat
│   ├── Phase-frame-summary.tsv
│   ├── System-summary.txt
│   └── frames/
│       ├── SqRadial-frame0001.dat
│       ├── Bragg-frame0001.dat
│       ├── FirstShell-frame0001.txt
│       └── ...
├── hexagonal/
└── micellar/
```

### `SqRadial-average.dat`

```text
q[A^-1]  Sq_frame_mean  Sq_frame_SD  Sq_frame_SEM  n_frames
```

This is the principal file for plotting the time-averaged radial structure factor.

### `SqRadial-frameXXXX.dat`

```text
q[A^-1]  Sq_mean  Sq_std  n_qvectors
```

`Sq_std` is the spread among reciprocal-vector directions falling in the same radial q bin; it is not a temporal standard deviation.

### `Bragg-frameXXXX.dat`

```text
rank h k l q[A^-1] Sq qx qy qz
```

This file contains the strongest Friedel-unique reciprocal-space peaks.

### `FirstShell-frameXXXX.txt`

Contains first-shell reciprocal-space diagnostics, including strong directions, pair angles, possible lamellar harmonics, and possible H1 reciprocal-lattice triplets.

### `Phase-frame-summary.tsv` and `All-systems-summary.tsv`

These files contain **diagnostic phase hints** from the current heuristic rules.

They should not be treated as an independent automatic phase-identification method. Low-order reciprocal vectors of a finite periodic simulation box can occasionally satisfy simple geometric criteria even for disordered systems. Final structural assignment should therefore use the time-averaged radial `S(q)`, persistent Bragg features across frames, representative real-space configurations, and independent structural observables.

### `run_parameters.json`

Stores the program version, Python/NumPy/Taichi versions, input files, and command-line settings for reproducibility.

## Structural interpretation

### Lamellar phase

A lamellar structure typically shows one dominant reciprocal direction and approximately collinear harmonics:

$$
q_1:q_2:q_3:\cdots=1:2:3:\cdots
$$

The lamellar repeat spacing is

$$
d=\frac{2\pi}{q_1}.
$$

### Hexagonal-columnar H1 phase

For a two-dimensional hexagonal lattice, the radial peak-position sequence is commonly

$$
1:\sqrt{3}:2:\sqrt{7}:3:\cdots .
$$

Finite simulation cells, lattice distortion, and radial averaging can split or weaken individual radial peaks. The unaveraged reciprocal-space data are therefore particularly useful: persistent near-equal first-shell reciprocal vectors separated by approximately $60^\circ$ are characteristic of a hexagonal reciprocal lattice.

The H1 lattice parameter is obtained from the (10) reflection:

$$
a=\frac{4\pi}{\sqrt{3}q_{10}}.
$$

### Disordered/micellar structures

Disordered micellar systems generally lack persistent sharp reciprocal-lattice peaks and instead show broad correlation features. For such systems, the time-averaged `S(q)`, peak width, temporal persistence, and real-space morphology are more informative than a single-frame geometric phase hint.

## Recommended analysis practice

1. Use only the equilibrated trajectory segment.
2. Sample frames sufficiently far apart to reduce strong temporal correlation.
3. Analyze each physical system in a separate PDB file.
4. Use `SqRadial-average.dat` as the main radial structure-factor result.
5. Inspect `Bragg-*` and `FirstShell-*` for representative ordered systems.
6. Interpret peak positions and periodicities as the primary quantities from equal-weight CG `S(q)`.
7. Do not interpret relative CG peak intensities as experimental SAXS intensities without a physically justified scattering model.

## Reference settings used in the accompanying Methods

```text
water bead W excluded
equal weight for every retained CG bead
qmax = 1.20 A^-1
radial bin width = 0.005 A^-1
Bragg diagnostic qmax = 0.60 A^-1
Taichi CUDA backend
```

## Performance notes

The computational cost scales approximately with

$$
N_{\mathrm{beads}}\times N_{\mathbf q}
$$

per frame.

For large systems:

- use an NVIDIA GPU when available;
- do not enable `--write-raw-3d` unless needed;
- use `--frame-stride` for preliminary tests;
- reduce `--qmax` if only low-q mesophase peaks are required.

## Troubleshooting

### `AttributeError: module 'taichi' has no attribute 'any_arr'`

This code uses the Taichi 1.7 API (`ti.types.ndarray`) and does not require the obsolete `ti.any_arr` API. Install the versions specified in `requirements.txt`.

### CUDA initialization fails

Use CPU mode:

```bash
python3 cg_structure_factor.py \
  -d ./pdb \
  -q 1.20 \
  --arch cpu
```

You can also check the NVIDIA driver with:

```bash
nvidia-smi
```

### `No CRYST1 available`

Every analyzed frame requires a valid simulation-cell definition. Export the trajectory to a PDB containing `CRYST1` for each frame.

### Empty q bins

This is normal for a finite periodic simulation box. Allowed q values are discrete reciprocal-lattice vectors. Empty radial bins are intentionally omitted rather than assigned `S(q)=0`.

### Large output

Avoid `--write-raw-3d` unless the complete reciprocal-space table is required.

## Repository contents

Recommended files:

```text
cg_structure_factor.py
README.md
METHODS.md
requirements.txt
run_example.sh
example_weights.json
```

## Citation

If the code is used in a publication, cite the associated article or repository DOI when available. Describe the default calculation as a **coarse-grained geometric/static structure-factor analysis** unless physically justified X-ray scattering weights are supplied.

## License

Add the license appropriate for your repository before public release.
