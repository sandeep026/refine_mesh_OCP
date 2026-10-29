## Tricks in OCP

### Continuous-Time Constraint Satisfaction

On a sparse mesh, path constraints are typically satisfied only pointwise at the collocation nodes rather than between the nodes. Such a situation can lead to missing active constraints and a loss of optimal performance. A denser grid is a natural fix, but it comes at a high computational cost. An alternative method is to pose the path constraint as an integral constraint on the constraint violation.

Convert the inequality constraint:

$$
h(t, x, u) \le 0
$$

to the integral constraint:


$$
\int_{t_0}^{t_f} \max\big(0, h(t, x, u)\big)^2 \, dt = 0
$$

The integral of a non-negative function yields a non-negative area, and the boundary conditions on the state ensure that this area must be zero. The integrand represents the constraint violation; enforcing this integral constraint guarantees that the violation remains zero throughout the domain.

If your solver does not natively support integral constraints, you can introduce a new state $x_{\text{ct}}$ with its time derivative defined as:


$$
\dot{x}_{\text{ct}} = 
\max\big(0, h(t, x, u)\big)^2
$$

For this new state, the boundary conditions are:


$$
x_{\text{ct}}(t_0) = x_{\text{ct}}(t_f) = 0
$$

> **Note:** This augmentation can affect the convergence of the NLP solver due to LICQ (Linear Independence Constraint Qualification) failure, so special care is required during implementation.


### Mesh Refinement

Approaches to refine a mesh for an Optimal Control Problem (OCP) can be classified into:

1. **Static Refinement**
2. **Dynamic Refinement**

#### 1. Static Refinement

In static refinement, you solve an OCP using a direct method and, from the NLP solution, obtain a metric for mesh refinement. These metrics can include regions in the mesh with high control rates, curvature of the state or control (discontinuities), constraint violations at non-discretized points, or dynamics errors.

The curvature may be computed directly using finite differences, interpolation, or wavelet analysis for the control or state. Other techniques include consistent interpolation for shooting methods to find dynamics residuals at non-shooting points, or denser evaluation of defect constraints using polynomial approximation in pseudospectral methods.

Once the regions of refinement are identified, more nodes or discretization points are added, and points are removed from areas that do not require them. Based on the new mesh, the OCP is discretized again and solved (warm-started). A convergence criterion—such as the relative error of the state, control, or objective—may be used for termination.

#### 2. Dynamic Refinement

Dynamic refinement, on the other hand, attempts to include time or segment widths as free variables. Additional constraints on these free variables (such as positivity and monotonicity) must be placed to obtain a meaningful solution. While more complex than the original OCP, it requires only a single OCP to be solved.

#### Time Dilation

**Time dilation** is a method where the OCP is reformulated to provide dynamic mesh refinement without altering the transcription itself, making it compatible with any transcription software.

Given the system dynamics:


$$
\dot{x} = f(t, x, u)
$$

Introduce a new state and control variable, $x_t = t$ and $u_t > 0$, such that:


$$
\hat{x}_t = u_t
$$

where $\hat{(\cdot)}$ denotes $\frac{d}{d\tau}$, with $\tau \in [\tau_0, \tau_f]$ representing a new scaled time that differs from physical time (typically $(\tau_0, \tau_f) = (0, 1)$). Using $dt = u_t \, d\tau$, the final augmented control problem becomes:


$$
\hat{\bar{x}} = \bar{f}(\bar{x}, \bar{u})
$$

where the augmented vectors and functions are defined as:


$$
\bar{x} = 
\begin{bmatrix} x 
\\ x_t 
\end{bmatrix}, 
\quad \bar{u} = 
\begin{bmatrix} 
u 
\\ 
u_t 
\end{bmatrix}, 
\quad \bar{f} = 
\begin{bmatrix} 
f(\bar{x}, u) u_t 
\\ 
u_t 
\end{bmatrix}
$$

Here, $u_t$ acts as the physical time-width rate and a control variable that must be strictly positive. Time is now treated as a new state variable, and a small value of $u_t$ implies a clustering of nodes.
