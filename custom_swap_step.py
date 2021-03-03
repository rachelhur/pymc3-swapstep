from pymc3.step_methods.arraystep import BlockedStep
from pymc3.step_methods.arraystep import ArrayStepShared
from pymc3.step_methods.arraystep import ArrayStep
from pymc3.model import modelcontext
from pymc3.step_methods.arraystep import metrop_select
from pymc3.step_methods.hmc import quadpotential

## delta_logp is needed for __init__ function for custom_step, as well as astep function
def delta_logp(logp, vars, shared):
    [logp0], inarray0 = pm.join_nonshared_inputs([logp], vars, shared)

    tensor_type = inarray0.type
    inarray1 = tensor_type("inarray1")

    logp1 = pm.CallableTensor(logp0)(inarray1)

    f = theano.function([inarray1, inarray0], logp1 - logp0)
    f.trust_input = True
    return f

class custom_step(ArrayStepShared):
    """Metropolis-Hastings optimized for binary variables
    Parameters
    ----------
    vars: list
        List of variables for sampler
    p_swap: float64
        Probability that the swap will be proposed at each iteration. Defaults to .7.
    tune: bool
        Flag for tuning. Defaults to True.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """
    
    name = 'custom_step'

    generates_stats = True
    stats_dtypes = [{
        'accept': np.float64,
        'swapped': np.bool,
        'tune': np.bool,
        'pair': list,
#         "model_logp": np.float64,
        'pre_betas': list,
        'pre_weights': list
        }]
    
    
    def __init__(self, vars=None, model=None, tune=True, p_swap=.7, swap_all=False):    
        
        model = pm.modelcontext(model)
        self.ncomponents = model.ndim // len(model.vars)
        
        if vars is None:
            vars = model.vars
        vars = [var for var in vars if var.type.ndim != 0] # get rid of the scalar amplitude variable
        vars = pm.inputvars(vars) # inputvars takes tensor variables and puts them into a list, also reverses order
        
        self.tune = tune
        self.p_swap = p_swap
        self.swapped = 0
        self.logodds = pm.distributions.transforms.logodds
        self.invlogit = scipy.special.expit
        
        shared = pm.make_shared_replacements(vars, model)
        self.delta_logp = delta_logp(model.logpt, vars, shared)
        super().__init__(vars, shared)
    
    
    def astep(self, q0):
        """ Custom jump to switch params of each component. Will randomly pick a component, pick one above or below, then switch components' params
        based on metrop_hast algorithm.
        """
        
        # accept __% of proposals
        p_swap = self.p_swap # acceptance prob
        swap = (np.random.random() > p_swap)
        
        # need list to swap items
        q = list(np.copy(q0)) # q0 is an array of coordinates in param space ordered by ArrayOrdered (will be ordered as appears in model, beta-->Q)
        #slices = np.array(ordering.vmap)[:, 1] # slices of q0 that correspond to each variable, ie 0th slice gives us our 8 beta values at this point
        swapped = swap # for when swap isn't True, we need to input a value for the stat "swapped"
        accept = -1e8
        pair = []
        q_new = q0
        pre_betas = []
        pre_weights = []
        
        if swap:
            # calculate new betas from current weights and betas
            logodds = self.logodds
            invlogit = self.invlogit
            ncomponents = self.ncomponents
            
            # pick random component, then pick the one above or below it
            component_idx = np.arange(ncomponents)
            pair.append(np.random.choice(component_idx))
            
            # this is periodic boundary conditions
    #         pair.append((pair[0] + 1) % self.ncomponents)
            
            # boundary conditions - maybe breaking detailed symmetry here
            if pair[0] == 0:
                pair.append(1)
            elif pair[0] == ncomponents-1:
                pair.append(ncomponents-2)
            else:
                pair.append(pair[0] + np.random.choice([-1, 1]))
            pair = np.sort(pair)
            
            # switch pairs
            if swap_all:
                q[pair[1]::ncomponents], q[pair[0]::ncomponents] = q[pair[0]::ncomponents], q[pair[1]::ncomponents]

            # Need to backward transform betas back into bounded [0,1] space to calculate current weights,
            # then calculate new betas, then retransform back into log space
            pre_betas = invlogit(q[-ncomponents:])
            portion_remaining = np.concatenate([[1], np.cumprod(1-pre_betas)[:-1]])
            pre_weights =  pre_betas * portion_remaining
#             print('pre_betas:', pre_betas)
            beta_0 = pre_weights[pair[1]] / portion_remaining[pair[0]]
            beta_1 = pre_weights[pair[0]] / (portion_remaining[pair[0]] * (1 - beta_0))

            new_betas = np.copy(pre_betas)
            new_betas[pair[0]], new_betas[pair[1]] = beta_0, beta_1
            
            q[-ncomponents + pair[0]], q[-ncomponents + pair[1]] = logodds.forward_val([beta_0, beta_1])
            q = np.array(q)
            
            # acceptance criterion
            accept = self.delta_logp(q, q0)
            q_new, swapped = metrop_select(accept, q, q0)
            self.swapped += swapped

        stats = {
            'accept': np.exp(accept),
            'swapped': swapped,
            'tune': self.tune,
            'pair': pair,
#             'model_logp': q_new.model_logp,
            'pre_betas': pre_betas,
            'pre_weights': pre_weights
            }   
        return q_new, [stats]

    @staticmethod
    def competence(var, has_grad):
        return Competence.COMPATIBLE
