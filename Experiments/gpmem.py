from venture.lite.psp import PSP, DeterministicPSP, TypedPSP, RandomPSP
from venture.lite.sp import SP, VentureSPRecord, SPType
from venture.lite.env import VentureEnvironment
from venture.lite.request import Request,ESR
from venture.lite.address import emptyAddress
from venture.lite.builtin import no_request, typed_nr, deterministic_typed, typed_func_psp
from venture.lite.function import VentureFunction
import venture.lite.types as t
from gp_der import GPOutputPSP, GPSP

class MakeGPMSPOutputPSP(DeterministicPSP):
    def simulate(self, args):
        f_node = args.operandNodes[0]
        prior_covariance_function = args.operandValues[1]
        f_compute = VentureSPRecord(
                SP(GPMComputerRequestPSP(f_node), GPMComputerOutputPSP()))
        # Prior mean is fixed to zero, because the current GP implementation
        # assumes this
        f_emu = VentureSPRecord(GPSP(zero_function, prior_covariance_function))
        f_compute.spAux = f_emu.spAux
        # TODO ways to get_Xseen and get_Yseen? maybe this belongs in the
        # inference side
        return t.pythonListToVentureList([f_compute, f_emu])

gpmemSP = no_request(MakeGPMSPOutputPSP())

zero_function = VentureFunction(lambda x: 0, [t.AnyType()], t.NumberType())

# TODO treat this as not necessarily random?
class GPMComputerRequestPSP(RandomPSP):
    def __init__(self, f_node):
        self.f_node = f_node

    def simulate(self, args):
        id = str(args.operandValues)
        exp = ["gpmemmedSP"] + [["quote",val] for val in args.operandValues]
        env = VentureEnvironment(None,["gpmemmedSP"],[self.f_node])
        return Request([ESR(id,exp,emptyAddress,env)])

# TODO Perhaps this could subclass ESRRefOutputPSP to correctly handle
# back-propagation?
class GPMComputerOutputPSP(DeterministicPSP):
    def simulate(self,args):
        assert len(args.esrNodes) ==  1
        return args.esrValues[0]

    def incorporate(self, value, args):
        # TODO maybe best to just call on someone else's incorporate method
        # instead? idk
        x = args.operandValues[0].getNumber()
        y = value.getNumber()
        args.spaux.samples[x] = y

    def unincorporate(self, value, args):
        x = args.operandValues[0].getNumber()
        samples = args.spaux.samples
        if x in samples:
            del samples[x]

