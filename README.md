# 1D_Poisson_FDM
Solution of 1D Poisson’s equation with Finite Difference Method.
The following boundary-value problem is considered:
\begin{equation}
    \begin{dcases} 
        -\frac{d^2u_i}{dx^2}=f_i, x \in (0, 3) \\
        u_i(0)=1 \\
        u_i(3)=1 \\
    \end{dcases}
\end{equation}
with the source functions:
\begin{equation}
    \begin{dcases} 
        f_1(x)=3x-2 \\
        f_2(x)=x^2+3x-2
    \end{dcases}
\end{equation}
