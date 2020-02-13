- first derivative check fails for some values in ipopt, but think this is because of clipping in war constraints (and differences between how these are handled in finite differences versus autograd). Adjusting clipping threshold seems to help. This might be a problem because convergence relies on values in Jacobian, however.
- other way around: finite differences are popping out zero when autograd has nonzero numbers