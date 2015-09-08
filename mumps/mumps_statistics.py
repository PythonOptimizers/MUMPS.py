"""
Statistic helpers for MUMPS.
"""

ordering_name = [ 'amd', 'user-defined', 'amf',
                  'scotch', 'pord', 'metis', 'qamd']


class AnalysisStatistics(object):
    def __init__(self, inst, time=None):
        self.est_mem_incore = inst.infog[17]
        self.est_mem_ooc = inst.infog[27]
        self.est_nonzeros = (inst.infog[20] if inst.infog[20] > 0 else
                             -inst.infog[20] * 1000000)
        self.est_flops = inst.rinfog[1]
        self.ordering = ordering_name[inst.infog[7]]
        self.time = time

    def __str__(self):
        parts = ["Analysis statistics\n",
                 "-------------------\n",
                 "estimated memory for in-core factorization: ",
                 str(self.est_mem_incore), " mbytes\n",
                 "estimated memory for out-of-core factorization: ",
                 str(self.est_mem_ooc), " mbytes\n",
                 "estimated number of nonzeros in factors: ",
                 str(self.est_nonzeros), "\n",
                 "estimated number of flops: ", str(self.est_flops), "\n",
                 "ordering used: ", self.ordering]
        if hasattr(self, "time"):
            parts.extend(["\nanalysis time: ", str(self.time), " secs"])
        return "".join(parts)


class FactorizationStatistics(object):
    def __init__(self, inst, time=None, include_ordering=False):
        # information about pivoting
        self.offdiag_pivots = inst.infog[12] if inst.issym() else 0
        self.delayed_pivots = inst.infog[13]
        self.tiny_pivots = inst.infog[25]

        # possibly include ordering (used in schur_complement)
        if include_ordering:
            self.ordering = ordering_name[inst.infog[7]]

        # information about runtime effiency
        self.memory = inst.infog[22]
        self.nonzeros = (inst.infog[29] if inst.infog[29] > 0 else
                         -inst.infog[29] * 1000000)
        self.flops = inst.rinfog[3]
        if time:
            self.time = time

    def __str__(self):
        parts = ["Factorization statistics\n",
                 "------------------------\n",
                 "off-diagonal pivots: ", str(self.offdiag_pivots), "\n",
                 "delayed pivots: ", str(self.delayed_pivots), "\n",
                 "tiny pivots: ", str(self.tiny_pivots), "\n"]
        if hasattr(self, "ordering"):
            parts.extend(["ordering used: ", self.ordering, "\n"])
        parts.extend(["memory used during factorization: ", str(self.memory),
                      " mbytes\n",
                      "nonzeros in factored matrix: ", str(self.nonzeros), "\n",
                      "floating point operations: ", str(self.flops)])
        if hasattr(self, "time"):
            parts.extend(["\nfactorization time: ", str(self.time), " secs"])
        return "".join(parts)


class SolveStatistics(object):
    def __init__(self, inst, time=None, include_ordering=False):
        # information about pivoting
        self.transpose_solve = False if inst.icntl[9]==1 else True
        self.matrix_infty_norm = inst.rinfog[4]
        self.solution_infty_norm = inst.rinfog[5]
        self.scaled_residual = inst.rinfog[6]
        self.omega1 = inst.rinfog[7]
        self.omega2 = inst.rinfog[8]
        self.complete_stats = 1 if inst.icntl[11]==1 else 0

        # more stats if complete stats are required
        if self.complete_stats:
            self.upperbound_forwarderror = inst.rinfog[9] # an upper bound of the forward error of the computed solution
            self.cond1 = inst.rinfog[10]
            self.cond2 = inst.rinfog[11]

        if time:
            self.time = time

    def __str__(self):
        parts = ["Solve statistics\n",
                 "----------------\n",
                 "transpose solve: ", str(self.transpose_solve), "\n"]

        if self.transpose_solve:
            parts.extend(["infty norm of A transpose: ", str(self.matrix_infty_norm), "\n"])
        else:
            parts.extend(["infty norm of A: ", str(self.matrix_infty_norm), "\n"])

        parts.extend(["solution infty norm: ", str(self.solution_infty_norm), "\n",
                      "scaled residual: ", str(self.scaled_residual), "\n"])
        if self.complete_stats:
            parts.extend(["upper bound on forward error: ", str(self.upperbound_forwarderror), "\n"])

        if hasattr(self, "time"):
            parts.extend(["\nsolve time: ", str(self.time), " secs"])
        return "".join(parts)
