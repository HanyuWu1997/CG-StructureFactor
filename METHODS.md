# Methods

## English manuscript-ready version

### Coarse-grained structure-factor analysis

The mesoscale organization of the simulated surfactant assemblies was characterized using a reciprocal-space static structure factor calculated from coarse-grained (CG) trajectories. Equilibrated trajectory segments were exported as multi-model PDB files, in which each `MODEL` record was treated as an independent configuration and the instantaneous simulation-cell parameters were read from the corresponding `CRYST1` record. Because the objective of this analysis was to quantify the periodic organization of the surfactant framework rather than to reproduce absolute experimental X-ray scattering intensities, CG water beads (`W`) were excluded and all retained surfactant beads were assigned an equal scattering weight.

For each configuration, the direct-cell matrix $\mathbf H$ was constructed from the instantaneous box vectors and the reciprocal-cell matrix was calculated as

$$
\mathbf B=2\pi(\mathbf H^{-1})^{T}.
$$

The scattering vectors were restricted to reciprocal-lattice vectors compatible with the periodic simulation cell,

$$
\mathbf q_{hkl}=\mathbf B(h,k,l)^{T},
$$

where $h$, $k$, and $l$ are integers. The zero vector was excluded. The equal-weight CG static structure factor was calculated as

$$
S(\mathbf q)=\frac{1}{N}
\left|
\sum_{j=1}^{N}
\exp(i\mathbf q\cdot\mathbf r_j)
\right|^{2},
$$

where $N$ is the number of selected surfactant beads and $\mathbf r_j$ is the position of bead $j$. Calculations were accelerated using Taichi with the CUDA backend. Reciprocal vectors with $0<|\mathbf q|\leq1.20\ \mathrm{\AA}^{-1}$ were included.

A one-dimensional structure-factor profile was obtained by radially averaging $S(\mathbf q)$ over reciprocal vectors within bins of width $\Delta q=0.005\ \mathrm{\AA}^{-1}$. Radial bins containing no reciprocal vector were omitted rather than assigned an artificial zero intensity. For each physical system, the radial structure factor was first calculated independently for every sampled configuration and subsequently averaged over configurations from the same trajectory. The resulting mean $S(q)$, standard deviation, and standard error were used to evaluate the temporal stability of the reciprocal-space features.

Because the simulated liquid-crystalline phases are anisotropic, the unaveraged three-dimensional reciprocal-space information was additionally retained for Bragg-peak analysis. Friedel-related vectors $\mathbf q$ and $-\mathbf q$ were treated as equivalent. Low-$q$ Bragg features up to $0.60\ \mathrm{\AA}^{-1}$ were inspected to identify persistent reciprocal-lattice directions and peak sequences. Lamellar ordering was characterized by a dominant reciprocal direction and approximately collinear integer harmonics, whereas hexagonal-columnar (H$_1$) ordering was identified from persistent near-equal first-shell reciprocal vectors separated by approximately $60^\circ$, together with the corresponding reciprocal-lattice relationships. The lamellar repeat distance was calculated as $d=2\pi/q_1$, and the H$_1$ lattice parameter was obtained from the (10) reflection as $a=4\pi/(\sqrt{3}q_{10})$.

The calculations were performed using a custom Python program (`cg_structure_factor.py`, version 3.0.0) with NumPy, Taichi 1.7.4, and tqdm. The analysis represents a geometric CG structure factor; therefore, peak positions and reciprocal-space periodicities, rather than absolute or relative X-ray scattering intensities, were used for structural interpretation.

## 中文参考版本

### 粗粒化体系结构因子分析

采用倒易空间静态结构因子对粗粒化分子动力学模拟所得表面活性剂聚集结构的介观有序性进行表征。选取达到平衡后的轨迹片段并导出为 multi-model PDB 文件，其中每一个 `MODEL` 作为一个独立构型进行计算，每帧模拟盒参数由对应的 `CRYST1` 记录读取。本分析的目的主要是表征表面活性剂骨架的周期性有序结构，而非直接重现实验绝对 X 射线散射强度，因此计算时排除粗粒化水 bead（`W`），并对所有保留的表面活性剂 beads 赋予相同散射权重。

对于每一帧构型，根据瞬时模拟盒矢量构建正空间晶胞矩阵 $\mathbf H$，并计算倒易晶胞矩阵

$$
\mathbf B=2\pi(\mathbf H^{-1})^{T}.
$$

仅使用与周期性模拟盒相容的离散倒易矢量

$$
\mathbf q_{hkl}=\mathbf B(h,k,l)^{T},
$$

其中 $h$、$k$ 和 $l$ 为整数，并排除 $\mathbf q=0$。等权重粗粒化静态结构因子定义为

$$
S(\mathbf q)=\frac{1}{N}
\left|
\sum_{j=1}^{N}
\exp(i\mathbf q\cdot\mathbf r_j)
\right|^{2},
$$

其中 $N$ 为参与计算的表面活性剂 bead 数目，$\mathbf r_j$ 为第 $j$ 个 bead 的空间坐标。计算采用 Taichi CUDA 后端进行 GPU 加速，倒易空间计算范围设置为 $0<|\mathbf q|\leq1.20\ \mathrm{\AA}^{-1}$。

随后按照 $|\mathbf q|$ 对三维结构因子进行径向平均，q-bin 宽度设置为 $\Delta q=0.005\ \mathrm{\AA}^{-1}$。对于没有任何允许倒易矢量落入的空 bin，不人为赋值为零，而是在输出中省略。对于同一物理体系的多帧轨迹，首先逐帧独立计算 $S(q)$，随后仅在同一体系内部进行时间平均，并输出平均值、标准差和标准误，以评价倒易空间结构特征的时间稳定性。

考虑到模拟得到的液晶相具有明显各向异性，在一维径向平均之外同时保留三维倒易空间 Bragg 峰信息，并将 $\mathbf q$ 与 $-\mathbf q$ 视为 Friedel 等价方向。对 $q\leq0.60\ \mathrm{\AA}^{-1}$ 的低 q Bragg 特征进行分析，以识别稳定的倒易晶格方向和峰序列。层状相主要表现为单一优势倒易方向及其近似共线的整数倍高阶谐波；六方柱状 H$_1$ 相则表现为第一倒易壳层中持续存在的近等长倒易矢量，其方向夹角接近 $60^\circ$，并满足相应的二维六方倒易晶格关系。层状相的重复周期按 $d=2\pi/q_1$ 计算，H$_1$ 相的晶格参数由 (10) 峰按照 $a=4\pi/(\sqrt{3}q_{10})$ 计算。

计算采用自编 Python 程序 `cg_structure_factor.py`（version 3.0.0）完成，并使用 NumPy、Taichi 1.7.4 和 tqdm。需要强调的是，本方法得到的是基于粗粒化 bead 空间分布的几何静态结构因子，因此主要利用峰位、周期性和倒易空间有序特征进行结构分析，而不将所得相对峰强度直接等同于实验 SAXS 强度。
