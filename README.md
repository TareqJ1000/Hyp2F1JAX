# Hyp2F1JAX
JAX implementation of the Gaussian Hypergeometric Function $_2F_1(a,b;c;z)$ for complex values z within and beyond the complex unit disc. Our use case involves computing the overlap between Laguerre-Gauss modes in quantum optics which, in the language of the Gauss Hypergeometric Function, involves integer values of a,b,c and generally complex values for z. The specific formula we are trying to compute is detailed in [1] as follows: 

\begin{align}
    (\ell,p|\ell',p') = A \frac{\Gamma(p+p'+ |\ell| + 1)}{p!p'!} \frac{(\sigma-\mu')^{p'}(\sigma - \mu)^{p}}{\sigma^{p+p'+|\ell|+1}} \nonumber \\ \times _2F_1\left(-p,-p';-p-p'-|\ell|; \frac{\sigma(\sigma - \mu' - \mu)}{(\sigma-\mu')(\sigma-\mu)}\right) 
\end{align}

Suggestions welcome! 

---
[1] Jaouni, Tareq, Gu, Xuemei, Krenn, Mario, D’Errico, Alessio and Karimi, Ebrahim. "Tutorial: Hong–Ou–Mandel interference with structured photons" Nanophotonics, 2025. https://doi.org/10.1515/nanoph-2025-0034






