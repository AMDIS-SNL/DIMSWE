# Test 2A sparse-endpoint direct versus FIML

All branches start from the same completed H1 model. Intermediate truth was excluded from optimization and used only for labeled post-hoc diagnostics.

| network | sparse H2 | sparse H5 | dense H1 | dense H2 | dense H5 | autonomous final | autonomous max |
|---|---:|---:|---:|---:|---:|---:|---:|
| h1-baseline | 0.00049822197817 | 0.00047490110405 | 0.000451080629713 | 0.000733923463575 | 0.0015110999502 | 3.88359217131e-07 | 5.0229189893e-07 |
| direct-h2 | 0.000493936864493 | 0.000465767084531 | 0.000456754896451 | 0.000733300280334 | 0.0014937912933 | 3.83473147437e-07 | 5.01121857162e-07 |
| fiml-h2 | 0.000493400050439 | 0.000469234384967 | 0.000447961188948 | 0.000727563916735 | 0.0014949704175 | 3.85014271009e-07 | 4.98997867327e-07 |
| direct-h5 | 0.000500676271297 | 0.000459990567007 | 0.000478499226513 | 0.000751916641372 | 0.0015001040853 | 3.78618556091e-07 | 4.89514966234e-07 |
| fiml-h5 | 0.000473306453074 | 0.000436608391251 | 0.00044523880589 | 0.000706127361386 | 0.00141057467785 | 3.56078329498e-07 | 4.91925091943e-07 |

Raw Stage-1 endpoint fits and the Stage-2 NN endpoint fits are both retained so amortization/compression loss is explicit.
