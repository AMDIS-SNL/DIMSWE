# Test 2A H1-H2-H5 horizon-curriculum stage boundaries

Autonomous metrics are post-hoc training-support diagnostics and did not select parameters or stop training.

| artifact | J_H1 | J_H2 | J_H5 | J_M2-X | J_op | mixed final | mixed max | mixed accumulated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1-200k-initial | 0.000704871420123 | 0.0011205544013 | 0.00225063315416 | 0.000834686430905 | 0.000373006108793 | 3.76365749345e-07 | 6.2154959234e-07 | 4.69286254499e-07 |
| H1-final | 0.000451080629713 | 0.000733923463575 | 0.0015110999502 | 0.000562814281957 | 0.000596002945174 | 3.88359217131e-07 | 5.0229189893e-07 | 4.22944898721e-07 |
| H2-final | 0.000452412954349 | 0.000731476238843 | 0.00149824796621 | 0.000565631135918 | 0.000602162284487 | 3.86165230758e-07 | 5.02223233858e-07 | 4.21079580464e-07 |
| H5-final | 0.000460603402166 | 0.000735632544376 | 0.00148479902977 | 0.000579287916931 | 0.000614873378205 | 3.80007259537e-07 | 4.96260173763e-07 | 4.14744110627e-07 |

All runs use truth only through state 80. Each horizon used a new optimizer process with empty L-BFGS history.
