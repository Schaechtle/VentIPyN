
from venture.lite.psp import DeterministicPSP 

class Order(DeterministicPSP):
    '''
    def simulate(self,args):
        ordering = args.operandValues[0]
        i = args.operandValues[1]
        j = args.operandValues[2]

        for index in range(len(ordering)-1):
            if (ordering[index]+1)==i:
                if (ordering[index+1]+1)==j:
                    return True
                else:
                    return False
        return False
    '''        
                     
    
    def simulate(self,args):
        ordering = args.operandValues[0]
        i = args.operandValues[1].getNumber()
        j = args.operandValues[2].getNumber()
        seen_i = False
        for item in ordering:
            if seen_i:
                if (item+1)==j:
                    return True
            else:
                if (item+1)==j:
                    return False
                elif (item+1)==i:
                    seen_i=True
        return False
        
   

        
#### Collapsed order sampling with a symmetric dirichlet multinomial




'''

class CategoricalOutputPSP(DiscretePSP):
  # (categorical ps outputs)
  def simulate(self,args):
    if len(args.operandValues) == 1: # Default values to choose from
      return simulateCategorical(args.operandValues[0], [VentureAtom(i) for i in range(len(args.operandValues[0]))])
    else:
      if not len(args.operandValues[0]) == len(args.operandValues[1]):
        raise VentureValueError("Categorical passed different length arguments.")
      return simulateCategorical(*args.operandValues)

  def logDensity(self,val,args):
    if len(args.operandValues) == 1: # Default values to choose from
      return logDensityCategorical(val, args.operandValues[0], [VentureAtom(i) for i in range(len(args.operandValues[0]))])
    else:
      return logDensityCategorical(val,*args.operandValues)

  def enumerateValues(self,args):
    indexes = [i for i, p in enumerate(args.operandValues[0]) if p > 0]
    if len(args.operandValues) == 1: return indexes
    else: return [args.operandValues[1][i] for i in indexes]

  def description(self,name):
    return "  (%s weights objects) samples a categorical with the given weights.  In the one argument case, returns the index of the chosen option as an atom; in the two argument case returns the item at that index in the second argument.  It is an error if the two arguments have different length." % name



class MakerOrderSymDirMultOutputPSP(DeterministicPSP):
  def simulate(self,args):
    (alpha,d) = (float(args.operandValues[0]),int(args.operandValues[1]))
    
    os = args.operandValues[2] if len(args.operandValues) > 2 else [VentureAtom(i) for i in range(n)]
    if not len(os) == n:
      raise VentureValueError("Set of objects to choose from is the wrong length")
    output = TypedPSP(OrderSymDirMultOutputPSP(alpha,n,os), SPType([], AnyType()))
    return VentureSPRecord(DirMultSP(NullRequestPSP(),output,alpha,n))

  def childrenCanAAA(self): return True

  def madeSpLogDensityOfCountsBound(self, aux):
    """Upper bound the log density the made SP may report for its
    counts, up to arbitrary additions to the aux (but not removals
    from it), and up to arbitrary changes to the args wherewith the
    maker is simulated."""
    # TODO Communicate the maker's fixed parameters here for a more
    # precise bound
    # TODO In the case where alpha is required to be an integer, I
    # think the log density of the counts is maximized for all
    # values being as small as possible.
    # TODO Can the aux ever be null?
    # TODO Do the math properly, esp. for alpha < 1
    N = aux.counts.total
    A = len(aux.counts) * 1.0
    gamma_one = scipy.special.gammaln(1.0)
    term1 = scipy.special.gammaln(A) - scipy.special.gammaln(N+A)
    return term1 + sum([scipy.special.gammaln(1+count) - gamma_one for count in aux.counts])

  def description(self,name):
    return "  %s is a symmetric variant of make_dir_mult." % name

class OrderSymDirMultOutputPSP(CDirMultOutputPSP):
  def __init__(self,alpha,n,os):
    super(OrderSymDirMultOutputPSP, self).__init__([alpha] * n, os)
    
'''
