from pymc3.backends.report import SamplerWarning, WarningType
from pymc3.distributions import BART
from pymc3.math import logbern, logdiffexp_numpy
from pymc3.step_methods.arraystep import Competence
from pymc3.step_methods.hmc.base_hmc import BaseHMC, DivergenceInfo, HMCStepData
from pymc3.step_methods.hmc.integration import IntegrationError
from pymc3.theanof import floatX
from pymc3.vartypes import continuous_types
from pymc3.step_methods.hmc import NUTS
from pymc3.step_methods.arraystep import metrop_select


from pymc3.step_methods.hmc.nuts import _Tree
from pymc3.step_methods.hmc import integration

import time

from collections import namedtuple

__all__ = ["NUTS"]

def delta_logp(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type("inarray1")

    logp1 = pm.CallableTensor(logp0)(inarray1)
    f = theano.function([inarray1, inarray0], logp1 - logp0)
    f.trust_input = True
    return f

class NUTS_SwapVars(NUTS):

    name = "nuts_swapvars"

    default_blocked = True
    generates_stats = True
    stats_dtypes = [
        {            ###### custom step stats #######
            "swapped": np.bool,
            "swap": np.bool,
            "accept": np.bool,
            "pair": list,
            "pre_weights": list,
            "new_weights": list,
            "permutation": np.ndarray,
            "cov": np.ndarray,
            "logged_vars": list,
            ################ nuts and BaseHMC stats ################
            "depth": np.int64,
            "step_size": np.float64,
            "tune": np.bool,
            "mean_tree_accept": np.float64,
            "step_size_bar": np.float64,
            "tree_size": np.float64,
            "diverging": np.bool,
            "energy_error": np.float64,
            "energy": np.float64,
            "max_energy_error": np.float64,
            "model_logp": np.float64,
            "process_time_diff": np.float64,
            "perf_counter_diff": np.float64,
            "perf_counter_start": np.float64
        }
    ]

    def __init__(self, vars=None, max_treedepth=10, early_max_treedepth=8, p_swap=.5, swap_all=False, **kwargs):

        super().__init__(vars, **kwargs)
        
        self.max_treedepth = max_treedepth
        self.early_max_treedepth = early_max_treedepth
        self._reached_max_treedepth = 0
        
        vars = (self._model).vars # self._model inherited from superclass
        vars = pm.inputvars(vars)
    
        shared = pm.make_shared_replacements(vars, self._model)  
#         self.swap_vars = swap_vars
        self.delta_logp = delta_logp(self._model.logpt, vars, shared)

        self.ncomponents = self._model.ndim // len(self._model.vars) # self._model inherited from BaseHMC
        self.p_swap = p_swap
        self.swapped = 0
        self.logodds = pm.distributions.transforms.logodds
        self.invlogit = scipy.special.expit
        
#         self.delta_logp = delta_logp(model.logpt, swap_vars, shared)
#         super().__init__(vars, shared)
    
    def astep(self, q0):
        '''
        astep() is a single Hamiltonian step originally defined in BaseHMC class. This fxn will replace BaseHMCs astep but will be entirely the same except for the following additions:
            1.) Redefine self.integrator for each step that decides to swap proposals. (self.integrator is defined in BaseHMC's init fxn)
            2.) 
        '''
        
        perf_start = time.perf_counter()
        process_start = time.process_time()

                         ############ custom step ##############################
        p_swap = self.p_swap # acceptance prob
        swap = (np.random.random() < p_swap)
        
        
        # set default values if swap not proposed
        q = np.copy(q0) 
        swapped = swap # will give True if swap step is proposed, and will be overwritten by False if swap step is proposed but rejected by metropolis
        accept = -1e8
        pair = []
        pre_weights = []
        new_weights = []
        q_new = q
        permutation = np.zeros(1)
        logged_vars = []
        
        if swap:
            # calculate new betas from current weights and betas
            logodds = self.logodds
            invlogit = self.invlogit
            ncomponents = self.ncomponents
            
            # pick random component, then pick the one above or below it
            component_idx = np.arange(ncomponents)
            pair.append(np.random.choice(component_idx))
            
            # boundary conditions - maybe breaking detailed symmetry here
            if pair[0] == 0:
                pair.append(1)
            elif pair[0] == ncomponents-1:
                pair.append(ncomponents-2)
            else:
                pair.append(pair[0] + np.random.choice([-1, 1]))
            pair = np.sort(pair)

            # Need to backward transform betas back into bounded [0,1] space to calculate current weights,
            # then calculate new betas, then retransform back into log space
            pre_betas = invlogit(q[-ncomponents:])
            portion_remaining = np.concatenate([[1], np.cumprod(1-pre_betas)[:-1]])
            pre_weights =  pre_betas * portion_remaining

            beta_0 = pre_weights[pair[1]] / portion_remaining[pair[0]]
            beta_1 = pre_weights[pair[0]] / (portion_remaining[pair[0]] * (1 - beta_0))
            
            # for diagnostic purposes -- comment out later if everything looks good
            new_betas = np.copy(pre_betas)
            new_betas[pair[0]], new_betas[pair[1]] = beta_0, beta_1
            new_portion_remaining = np.concatenate([[1], np.cumprod(1-new_betas)[:-1]])
            new_weights = new_betas * new_portion_remaining
#             print('pre_weights: ', pre_weights)
#             print('new_weights: ', new_weights)
#             print('next')
            q[-ncomponents + pair[0]], q[-ncomponents + pair[1]] = logodds.forward_val([beta_0, beta_1])
            if swap_all:
                q[pair[1]:-ncomponents-1:ncomponents], q[pair[0]:-ncomponents-1:ncomponents] = q[pair[0]:-ncomponents-1:ncomponents], q[pair[1]:-ncomponents-1:ncomponents]
            q = np.array(q)
            logged_vars = q
            
            # acceptance criterion
            accept = self.delta_logp(q, q0)
            
            q_new, swapped = metrop_select(accept, q, q0)
                
            self.swapped += swapped
                
        custom_stats = {
            'swap': swap,
            'swapped': swapped,
            'accept': np.exp(accept),
            'pair': pair,
            'pre_weights': pre_weights,
            'new_weights': new_weights,
            'permutation': permutation,
            'cov': self.potential._cov,
            'logged_vars': logged_vars
            }
    
        ################################# end custom_step #########################################
        
        p0 = self.potential.random()
        start = self.integrator.compute_state(q_new, p0)

        if not np.isfinite(start.energy):
            model = self._model
            check_test_point = model.check_test_point()
            error_logp = check_test_point.loc[
                (np.abs(check_test_point) >= 1e20) | np.isnan(check_test_point)
            ]
            self.potential.raise_ok(self._logp_dlogp_func._ordering.vmap)
            message_energy = (
                "Bad initial energy, check any log probabilities that "
                "are inf or -inf, nan or very small:\n{}".format(error_logp.to_string())
            )
            warning = SamplerWarning(
                WarningType.BAD_ENERGY,
                message_energy,
                "critical",
                self.iter_count,
            )
            self._warnings.append(warning)
            raise SamplingError("Bad initial energy")

        adapt_step = self.tune and self.adapt_step_size
        step_size = self.step_adapt.current(adapt_step)
        self.step_size = step_size

        if self._step_rand is not None:
            step_size = self._step_rand(step_size)

        hmc_step = self._hamiltonian_step(start, p0, step_size)

        perf_end = time.perf_counter()
        process_end = time.process_time()

        self.step_adapt.update(hmc_step.accept_stat, adapt_step)
        self.potential.update(hmc_step.end.q, hmc_step.end.q_grad, self.tune)
        if hmc_step.divergence_info:
            info = hmc_step.divergence_info
            point = None
            point_dest = None
            info_store = None
            if self.tune:
                kind = WarningType.TUNING_DIVERGENCE
            else:
                kind = WarningType.DIVERGENCE
                self._num_divs_sample += 1
                # We don't want to fill up all memory with divergence info
                if self._num_divs_sample < 100 and info.state is not None:
                    point = self._logp_dlogp_func.array_to_dict(info.state.q)
                if self._num_divs_sample < 100 and info.state_div is not None:
                    point_dest = self._logp_dlogp_func.array_to_dict(info.state_div.q)
                if self._num_divs_sample < 100:
                    info_store = info
            warning = SamplerWarning(
                kind,
                info.message,
                "debug",
                self.iter_count,
                info.exec_info,
                divergence_point_source=point,
                divergence_point_dest=point_dest,
                divergence_info=info_store,
            )

            self._warnings.append(warning)

        self.iter_count += 1
        if not self.tune:
            self._samples_after_tune += 1

        stats = {
            "tune": self.tune,
            "diverging": bool(hmc_step.divergence_info),
            "perf_counter_diff": perf_end - perf_start,
            "process_time_diff": process_end - process_start,
            "perf_counter_start": perf_start,
        }
        
        stats = {**stats, **custom_stats}

        stats.update(hmc_step.stats)
        stats.update(self.step_adapt.stats())
        

        return hmc_step.end.q, [stats]
        
    @staticmethod
    def competence(var, has_grad):
        """Check how appropriate this class is for sampling a random variable."""
        if var.dtype in continuous_types and has_grad and not isinstance(var.distribution, BART):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE

    def warnings(self):
        warnings = super().warnings()
        n_samples = self._samples_after_tune
        n_treedepth = self._reached_max_treedepth

        if n_samples > 0 and n_treedepth / float(n_samples) > 0.05:
            msg = (
                "The chain reached the maximum tree depth. Increase "
                "max_treedepth, increase target_accept or reparameterize."
            )
            warn = SamplerWarning(WarningType.TREEDEPTH, msg, "warn")
            warnings.append(warn)
        return warnings

class NUTS_SwapAll(NUTS):

    name = "nuts"

    default_blocked = True
    generates_stats = True
    stats_dtypes = [
        {            ###### custom step stats #######
            "swapped": np.bool,
            "swap": np.bool,
            "accept": np.bool,
            "pair": list,
            "pre_weights": list,
            "new_weights": list,
            "permutation": np.ndarray,
            "cov": np.ndarray,
            ################ nuts and BaseHMC stats ################
            "depth": np.int64,
            "step_size": np.float64,
            "tune": np.bool,
            "mean_tree_accept": np.float64,
            "step_size_bar": np.float64,
            "tree_size": np.float64,
            "diverging": np.bool,
            "energy_error": np.float64,
            "energy": np.float64,
            "max_energy_error": np.float64,
            "model_logp": np.float64,
            "process_time_diff": np.float64,
            "perf_counter_diff": np.float64,
            "perf_counter_start": np.float64
        }
    ]

    def __init__(self, vars=None, max_treedepth=10, early_max_treedepth=8, p_swap=.5, **kwargs):

        super().__init__(vars, **kwargs)
        
        self.max_treedepth = max_treedepth
        self.early_max_treedepth = early_max_treedepth
        self._reached_max_treedepth = 0
        
        vars = (self._model).vars # self._model inherited from superclass
        vars = pm.inputvars(vars)
    
        shared = pm.make_shared_replacements(vars, self._model)  
#         self.swap_vars = swap_vars
        self.delta_logp = delta_logp(self._model.logpt, vars, shared)

        self.ncomponents = self._model.ndim // len(self._model.vars) # self._model inherited from BaseHMC
        self.p_swap = p_swap
        self.swapped = 0
        self.logodds = pm.distributions.transforms.logodds
        self.invlogit = scipy.special.expit
        
#         self.delta_logp = delta_logp(model.logpt, swap_vars, shared)
#         super().__init__(vars, shared)
    
    def astep(self, q0):
        '''
        astep() is a single Hamiltonian step originally defined in BaseHMC class. This fxn will replace BaseHMCs astep but will be entirely the same except for the following additions:
            1.) Redefine self.integrator for each step that decides to swap proposals. (self.integrator is defined in BaseHMC's init fxn)
            2.) 
        '''
        
        perf_start = time.perf_counter()
        process_start = time.process_time()

                         ############ custom step ##############################
        p_swap = self.p_swap # acceptance prob
        swap = (np.random.random() < p_swap)
        
        
        # set default values if swap not proposed
        q = np.copy(q0) 
        swapped = swap # will give True if swap step is proposed, and will be overwritten by False if swap step is proposed but rejected by metropolis
        accept = -1e8
        pair = []
        pre_weights = []
        new_weights = []
        q_new = q
        permutation = np.zeros(1)
        cov = self.potential._cov
        
        if swap:
            # calculate new betas from current weights and betas
            logodds = self.logodds
            invlogit = self.invlogit
            ncomponents = self.ncomponents
            
            # pick random component, then pick the one above or below it
            component_idx = np.arange(ncomponents)
            pair.append(np.random.choice(component_idx))
            
            # boundary conditions - maybe breaking detailed symmetry here
            if pair[0] == 0:
                pair.append(1)
            elif pair[0] == ncomponents-1:
                pair.append(ncomponents-2)
            else:
                pair.append(pair[0] + np.random.choice([-1, 1]))
            pair = np.sort(pair)
            self.pair = pair
            # Need to backward transform betas back into bounded [0,1] space to calculate current weights,
            # then calculate new betas, then retransform back into log space
            pre_betas = invlogit(q[-ncomponents:])
            portion_remaining = np.concatenate([[1], np.cumprod(1-pre_betas)[:-1]])
            pre_weights =  pre_betas * portion_remaining

            beta_0 = pre_weights[pair[1]] / portion_remaining[pair[0]]
            beta_1 = pre_weights[pair[0]] / (portion_remaining[pair[0]] * (1 - beta_0))
            
            # for diagnostic purposes -- comment out later if everything looks good
            new_betas = np.copy(pre_betas)
            new_betas[pair[0]], new_betas[pair[1]] = beta_0, beta_1
            new_portion_remaining = np.concatenate([[1], np.cumprod(1-new_betas)[:-1]])
            new_weights = new_betas * new_portion_remaining
#             print('pre_weights: ', pre_weights)
#             print('new_weights: ', new_weights)
#             print('next')
            q[-ncomponents + pair[0]], q[-ncomponents + pair[1]] = logodds.forward_val([beta_0, beta_1])
            q[pair[1]:-ncomponents-1:ncomponents], q[pair[0]:-ncomponents-1:ncomponents] = q[pair[0]:-ncomponents-1:ncomponents], q[pair[1]:-ncomponents-1:ncomponents]
            q = np.array(q)
            
            # acceptance criterion
            accept = self.delta_logp(q, q0)
            
            q_new, swapped = metrop_select(accept, q, q0)
                
            self.swapped += swapped
            
#             if we swap elements, swap covariance too
            if swapped:
                # original ordering of covariance
                original_ordering = np.arange(potential._n)
                
                # permutation of cov according to pair-swap proposal
                permutation = original_ordering.copy()
                permutation[[pair[0], pair[0]+ncomponents + 1, pair[0]+2*ncomponents + 1, pair[0] + 3*ncomponents + 1, pair[0] + 4*ncomponents + 1, pair[0] + 5*ncomponents + 1, pair[1], pair[1]+ncomponents + 1, pair[1] + 2*ncomponents + 1, pair[1] + 3*ncomponents + 1, pair[1] + 4*ncomponents + 1, pair[1] + 5*ncomponents + 1]] = permutation[[pair[1], pair[1] + ncomponents + 1, pair[1] + 2*ncomponents + 1, pair[1] + 3*ncomponents + 1, pair[1] + 4*ncomponents + 1, pair[1] + 5*ncomponents + 1, pair[0], pair[0]+ncomponents + 1, pair[0] + 2*ncomponents + 1, pair[0] + 3*ncomponents + 1, pair[0] + 4*ncomponents + 1, pair[0] + 5*ncomponents+1]]
                
#                 swap appropriate elements of cov and inform the associated cholesky decomposition
#                 swapped_ordering = original_ordering[permutation]
#                 p = np.argsort(swapped_ordering)
                
                self.potential._cov = self.potential._cov[permutation][:, permutation]
                self.potential._chol = scipy.linalg.cholesky(self.potential._cov, lower=True)

                
                # redefine integrator as it also is informed by the potential
                self.integrator = integration.CpuLeapfrogIntegrator(self.potential, self._logp_dlogp_func)
                
                
        custom_stats = {
            'swap': swap,
            'swapped': swapped,
            'accept': np.exp(accept),
            'pair': pair,
            'pre_weights': pre_weights,
            'new_weights': new_weights,
            'permutation': permutation,
            'cov': self.potential._cov
            }   
        ################################# end custom_step #########################################
        
        p0 = self.potential.random()
        start = self.integrator.compute_state(q_new, p0)

        if not np.isfinite(start.energy):
            model = self._model
            check_test_point = model.check_test_point()
            error_logp = check_test_point.loc[
                (np.abs(check_test_point) >= 1e20) | np.isnan(check_test_point)
            ]
            self.potential.raise_ok(self._logp_dlogp_func._ordering.vmap)
            message_energy = (
                "Bad initial energy, check any log probabilities that "
                "are inf or -inf, nan or very small:\n{}".format(error_logp.to_string())
            )
            warning = SamplerWarning(
                WarningType.BAD_ENERGY,
                message_energy,
                "critical",
                self.iter_count,
            )
            self._warnings.append(warning)
            raise SamplingError("Bad initial energy")

        adapt_step = self.tune and self.adapt_step_size
        step_size = self.step_adapt.current(adapt_step)
        self.step_size = step_size

        if self._step_rand is not None:
            step_size = self._step_rand(step_size)

        hmc_step = self._hamiltonian_step(start, p0, step_size)

        perf_end = time.perf_counter()
        process_end = time.process_time()

        self.step_adapt.update(hmc_step.accept_stat, adapt_step)
        self.potential.update(hmc_step.end.q, hmc_step.end.q_grad, self.tune)
        if hmc_step.divergence_info:
            info = hmc_step.divergence_info
            point = None
            point_dest = None
            info_store = None
            if self.tune:
                kind = WarningType.TUNING_DIVERGENCE
            else:
                kind = WarningType.DIVERGENCE
                self._num_divs_sample += 1
                # We don't want to fill up all memory with divergence info
                if self._num_divs_sample < 100 and info.state is not None:
                    point = self._logp_dlogp_func.array_to_dict(info.state.q)
                if self._num_divs_sample < 100 and info.state_div is not None:
                    point_dest = self._logp_dlogp_func.array_to_dict(info.state_div.q)
                if self._num_divs_sample < 100:
                    info_store = info
            warning = SamplerWarning(
                kind,
                info.message,
                "debug",
                self.iter_count,
                info.exec_info,
                divergence_point_source=point,
                divergence_point_dest=point_dest,
                divergence_info=info_store,
            )

            self._warnings.append(warning)

        self.iter_count += 1
        if not self.tune:
            self._samples_after_tune += 1

        stats = {
            "tune": self.tune,
            "diverging": bool(hmc_step.divergence_info),
            "perf_counter_diff": perf_end - perf_start,
            "process_time_diff": process_end - process_start,
            "perf_counter_start": perf_start,
        }
        
        stats = {**stats, **custom_stats}

        stats.update(hmc_step.stats)
        stats.update(self.step_adapt.stats())
        

        return hmc_step.end.q, [stats]
        
    @staticmethod
    def competence(var, has_grad):
        """Check how appropriate this class is for sampling a random variable."""
        if var.dtype in continuous_types and has_grad and not isinstance(var.distribution, BART):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE

    def warnings(self):
        warnings = super().warnings()
        n_samples = self._samples_after_tune
        n_treedepth = self._reached_max_treedepth

        if n_samples > 0 and n_treedepth / float(n_samples) > 0.05:
            msg = (
                "The chain reached the maximum tree depth. Increase "
                "max_treedepth, increase target_accept or reparameterize."
            )
            warn = SamplerWarning(WarningType.TREEDEPTH, msg, "warn")
            warnings.append(warn)
        return warnings
