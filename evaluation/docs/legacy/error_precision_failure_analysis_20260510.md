# Error-level Precision 失败样例分析

基于 `error_metrics.json` 中的 `unmatched_pred_locatable_count` 排序，选取高影响样例。

| id | unmatched_pred | v4_diag | baseline_diag | 典型未匹配诊断（rule + quote） |
|----|----------------|---------|---------------|----------------------------------|
| cl_236_160651 | 4 | 5 | 1 | exp_a88f99b62e7df910 :: n_{sv,2} = \frac{p_{sv,2} V_{\text{air},2}}{R T_2}<br>exp_0418bf8c32e6e830 :: The number of moles of water vapor in the air pocket at $ \theta_2 $ can be foun |
| cl_97_17636 | 4 | 5 | 1 | exp_2cf32fed971445e2 :: v_{\text{gas}} = v_{\text{star}} - v_{0}<br>exp_047e92b4d83463bc :: The total mechanical energy of the gas (kinetic plus potential) is conserved. |
| cl_174_96977 | 4 | 4 | 0 | exp_b23ecfa32d1dd46c :: n' = \frac{n}{\gamma}<br>exp_340f181f3f0b4ffa :: n' = \frac{n}{\gamma} |
| cl_214_137492 | 4 | 4 | 1 | exp_795ca7fba32604c7 :: M g d \sin \theta = \frac{17}{24} M v^2<br>exp_1670c843ed9a4a3c :: Since the prism is rolling without slipping, the linear velocity $v$ and angular |
| cl_85_5838 | 3 | 5 | 0 | exp_a84bda60eb4c269f :: m \cdot \frac{d^2 \xi}{dt^2} = F<br>exp_e4b0abf1359e2acc :: T = \pi \sqrt{\frac{h}{g}} |
| cl_261_186252 | 3 | 5 | 1 | exp_377af6ffee7493e4 :: The height of the tank from the free water surface of the reservoir is $h$ (the <br>exp_e540317efbc5369d :: At the exit of the hole, the pressure is also atmospheric but adjusted for the m |
| cl_129_50393 | 3 | 5 | 1 | exp_e540317efbc5369d :: The upwind air enterin<br>exp_616ca6a8d59de610 :: \frac{d}{dt} \left( C(t) \times L \times W \times H \right) = M - C(t) \times u  |
| cl_93_13749 | 3 | 4 | 0 | exp_1e3aa61ef5176162 :: T \rho^\gamma = \text{constant}<br>exp_24352c64b9f2d1c9 :: \frac{dP}{dz} = -\rho g |
| cl_224_148454 | 3 | 5 | 2 | exp_047e92b4d83463bc :: The ISS orbital decay is caused by one or more mechanisms which absorb energy fr<br>exp_326d94237b20217a :: For simplicity, we can use the average cross-sectional area of the ISS, which is |
| cl_169_91657 | 3 | 4 | 1 | exp_1052bcb5114eeda7 :: The upwind air enterin<br>exp_e540317efbc5369d :: To calculate the concentration $ C(t) $ of CO at 8:00 a.m., we need to consider  |
| cl_132_54083 | 3 | 3 | 3 | exp_d7f1c3a3eb6a1dae :: p_d = e B r_d = (1.60 × 10^{-19} C) × (1.00 T) × (0.210 m) = 3.36 × 10^{-19} kg·<br>exp_03ca09674b9ba8f3 :: The change in momentum is: \[ \Delta p = p_{\perp, \text{final}} - (-p_{\perp, \ |
| cl_257_181380 | 2 | 3 | 0 | exp_377af6ffee7493e4 :: q = \frac{k h}{\mu L} \Delta P<br>exp_7ad380952c9d4b28 :: From Step 2 and Step 5:

\[ -\frac{q(t)}{\varphi L^{2} c_r h} = \frac{\mu L}{k h |
| cl_223_147130 | 2 | 4 | 1 | exp_dc5d0a684b650313 :: ions generated at different positions... arrive as simultaneously as possible<br>exp_0eea791617b75b55 :: the distance between the entry and exit points is $l$ (see figure) |
| cl_230_153846 | 2 | 3 | 0 | exp_e54c097bd7f4f135 :: a_t = \frac{5}{7} g \sin \theta<br>exp_ad6ce22bf45ef7ac :: For rolling without slipping, $ f = \mu N $, and the acceleration condition give |
| cl_214_137777 | 2 | 4 | 1 | exp_d2c587c182607b74 :: In a cyclotron, the ion is accelerated by an alternating electric field between <br>exp_1e3b7cf9720ca55d :: The radius of the n-th orbit is: r_n = \sqrt{\frac{2m n V}{q B^2}} |

## 典型根因归纳

- **规则匹配过宽**：高分规则命中的是背景句/中间公式，而不是真正错误断言。
- **同段多规则叠加**：同一段落发布多个同义诊断，易产生 locatable 但不对齐 GT 的误报。
- **符号核查对 `none` 诊断约束不足**：缺少 `supported` 时仍可凭中等分数发布。

## 样例上下文（前 5 条）

### cl_236_160651

- question: Coffee - Flavored Physics

The most popular coffee making device in Italy is the so-called moka pot (Fig. 1) and the purpose of this problem is to analyze some of the physical processes and phenomena taking place during the coffee brewing process.
The moka pot has three independent structural elemen
- prediction: To derive an expression for the mass of saturated water vapors (\( m_{sv, 2} \)) in the air pocket at the temperature \( \theta_2 = 72.0^\circ \mathrm{C} \) and calculate its numerical value, we need to follow these steps:

### Step 1: Determine the volume of the air pocket at \( \theta_1 = 17.3^\circ \mathrm{C} \)
When the water level in the kettle is at \( \theta_1 = 17.3^\circ \mathrm{C} \), the kettle is filled with 60.0 g of water, and there is no air pocket above the water. The total volum

### cl_97_17636

- question: Assume that $M \gg m_{0}$, so that the ordinary star is basically rotating around the neutron star in a circular orbit of radius $r_{0}$. Assume that the ordinary star starts emitting gas toward the neutron star with a speed $v_{0}$, relative to the ordinary star (see Fig. 2). Assuming that the neut
- prediction: To determine the gas's distance of closest approach \( r_{f} \) using conservation laws, we need to consider the conservation of energy and angular momentum. Here's the step-by-step reasoning:

### Step 1: Identify the initial conditions
- The ordinary star is in a circular orbit of radius \( r_{0} \) around the neutron star.
- The mass of the neutron star is \( M \), and the mass of the ordinary star is \( m_{0} \).
- The gas is emitted radially inward from the ordinary star with a speed \( v_{

### cl_174_96977

- question: Active galactic nuclei (AGN) are supermassive black holes which form the centres of galaxies, and emit large amounts of energy in radiation and particle flows. One feature of many AGN are jetted outflows, which can be observed through radio emission, and sometimes also in other parts of the electrom
- prediction: To find the number density of particles \( n'(s) \) in the frame of the AGN, we need to consider the transformation of the number density from the jet frame (proper frame) to the AGN frame using special relativity principles.

### Step 1: Understand the relationship between the frames
The jet frame is the local co-moving frame of the jet plasma, where the particles are at rest on average. The AGN frame is the stationary frame of the active galactic nucleus. The transformation between these frame

### cl_214_137492

- question: Consider a long, solid, rigid, regular hexagonal prism like a common type of pencil (Figure 1.1). The mass of the prism is $M$ and it is uniformly distributed. The length of each side of the cross-sectional hexagon is $a$. The moment of inertia $I$ of the hexagonal prism about its central axis is

$
- prediction: To determine the minimum initial kinetic energy coefficient \(\delta\) in terms of the slope angle \(\theta\) and the energy retention coefficient \(r\), we need to analyze the motion and energy conservation of the hexagonal prism as it rolls down the slope and impacts.

### Step 1: Analyze the motion of the prism on the slope
When the prism rolls down the slope, it has both translational and rotational kinetic energy. The potential energy lost as the prism descends is converted into these kinet

### cl_85_5838

- question: In certain lakes there is a strange phenomenon called "seiching" which is an oscillation of the water. Lakes in which you can see this phenomenon are normally long compared with the depth and also narrow. It is natural to see waves in a lake but not something like the seiching, where the entire
 wat
- prediction: To derive a formula for the oscillation period \( T \) of the seiching in a rectangular container, we will follow these steps:

### Step 1: Understand the Problem
We have a rectangular container of length \( L \) and water depth \( h \). The water surface is initially inclined at a small angle and then oscillates about the horizontal plane. We assume that the water surface remains plane and oscillates symmetrically about the center of the container.

### Step 2: Simplify the Problem
Since the os

