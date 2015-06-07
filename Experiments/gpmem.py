from venture.lite.psp import DeterministicPSP, TypedPSP
from venture.lite.sp import SP, VentureSPRecord, SPType
from venture.lite.env import VentureEnvironment
from venture.lite.request import Request,ESR
from venture.lite.address import emptyAddress
from venture.lite.builtin import no_request, typed_nr, deterministic_typed, typed_func_psp
import venture.lite.types as t
import collections
import copy
from gp_der import GPOutputPSP

# Prior mean is fixed to zero, because the current GP implementation assumes
# this
def gpmem_output(ftrue_sp, prior_covariance_sp):
    state = GPMemState(ftrue_sp, prior_covariance_sp)
    compute = state.get_computer()
    emu = state.get_emulator()
    getXseen = state.get_Xseen_getter()
    getYseen = state.get_Yseen_getter()

    return [VentureSPRecord(p) for p in [compute, emu, getXseen, getYseen]]

gpmemSP = deterministic_typed(gpmem_output, [t.AnyType(), t.AnyType()], t.HomogeneousListType(t.AnyType()))

zero_sp = deterministic_typed(lambda xs: len(xs)*[0], [t.HomogeneousArrayType(t.AnyType())], t.HomogeneousArrayType(t.NumberType()))

class GPMemState:
    def __init__(self, ftrue_sp, prior_covariance_sp):
        self.ftrue_sp = ftrue_sp
        self.gp = GPOutputPSP(zero_sp, prior_covariance_sp)
        # TODO I am redundantly storing samples here...
        self.samples = collections.OrderedDict()

    def get_computer(self):
        #def compute_at_point(x):
        #    return self.ftrue_sp(x)
        #def logDensity(x, y):
        #    return self.ftrue_sp.logDensity(x, y)
        #def incorporate(value, args):
        #    self[args.operandValues[0]] = value
        #    self.gp.incorporate(value, args)
        #def unincorporate(value, args):
        #    x = args.operandValues[0]
        #    if x in self.samples:
        #        del self.samples[x]
        #    self.gp.unincorporate(value, args)
        #def isRandom():
        #    return self.ftrue.isRandom()
        #computerPSP = typed_func_psp(compute_at_point, [t.NumberType()], t.NumberType())
        #computerPSP.logDensity = logDensity
        #computerPSP.incorporate = incorporate
        #computerPSP.unincorporate = unincorporate
        #computerPSP.isRandom = isRandom
        #return no_request(computerPSP)

        ### TAKE TWO
        ### Er uh er
        # computerSP = copy.copy(self.ftrue_sp)
        # computerPSP = computerSP.outputPSP
        # old_incorporate = computerPSP.incorporate
        # old_unincorporate = computerPSP.unincorporate
        # def incorporate(value, args):
        #     old_incorporate(value, args)

        #     self[args.operandValues[0]] = value
        #     self.gp.incorporate(value, args)
        # def unincorporate(value, args):
        #     x = args.operandValues[0]
        #     if x in self.samples:
        #         del self.samples[x]
        #     self.gp.unincorporate(value, args)

        #     old_unincorporate(value, args)
        # computerPSP.incorporate = incorporate
        # computerPSP.unincorporate = unincorporate
        # return no_request(computerPSP)

    def get_emulator(self):
        #def sample_at(x):
        #    return self.gp.sample(x)
        #def logDensity(x, y):
        #    return self.gp.logDensity([x], [y])
        ## TODO wrap sample_at into an SP (that reports its likelihood) and
        ## return that SP
        return no_request(self.gp)

    def get_Xseen_getter(self):
        def get_Xseen():
            return self.samples.keys()
        return deterministic_typed(get_Xseen, [], t.HomogeneousArrayType(t.AnyType()))

    def get_Yseen_getter(self):
        def get_Yseen():
            return self.samples.values()
        return deterministic_typed(get_Yseen, [], t.HomogeneousArrayType(t.AnyType()))
