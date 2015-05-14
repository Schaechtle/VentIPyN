from venture.lite.psp import DeterministicPSP, NullRequestPSP
from venture.lite.sp import SP


class ApplyFunctionGradientOutputPSP(DeterministicPSP):
    def simulate(self, args):
        function = args.operandValues[0]
        arguments = args.operandValues[1:]
        sp_type = function.sp_type
        unwrapped_args = sp_type.unwrap_arg_list(arguments)
        # print sp_type.name(), unwrapped_args
        returned = function.f(*unwrapped_args)
        wrapped_return = sp_type.wrap_return(returned)

        return wrapped_return

    def gradientOfSimulate(self, args, _value, _direction):
        f = args.operandValues[0]
        arguments = args.operandValues[1:]
        gradient = []
        for function in f.stuff['derivatives']:
            sp_type = function.sp_type
            unwrapped_args = sp_type.unwrap_arg_list(arguments)
            # print sp_type.name(), unwrapped_args
            returned = function.f(*unwrapped_args)
            wrapped_return = sp_type.wrap_return(returned)
            gradient.append(wrapped_return)
        return gradient


    def description(self, _name=None):
        return "Apply a VentureFunction to arguments."

# TODO Add type signature. Look at signature of apply?
applyFunctionGradientSP = SP(NullRequestPSP(), ApplyFunctionGradientOutputPSP())



class TestApplyFunctionGradientOutputPSP(DeterministicPSP):
    def simulate(self, args):
        function = args.operandValues[0]
        arguments = args.operandValues[1:]
        gradient = []
        derivatives = function.stuff['derivatives']
        for i in len(derivatives):
            sp_type = derivatives[i].sp_type
            unwrapped_args = sp_type.unwrap_arg_list(arguments)
            # print sp_type.name(), unwrapped_args
            returned = derivatives[i].f(*unwrapped_args)
            wrapped_return = sp_type.wrap_return(returned)
            gradient.append(wrapped_return)
        return gradient[0]




testApplyFunctionGradientSP = SP(NullRequestPSP(), ApplyFunctionGradientOutputPSP())