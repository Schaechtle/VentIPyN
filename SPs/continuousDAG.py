'''
class NormalDAGOutputPSP(RandomPSP):
  # TODO don't need to be class methods
  def simulateNumeric(self,params): return scipy.stats.norm.rvs(*params)
  
  def logDensityNumeric(self,x,params): return scipy.stats.norm.logpdf(x,*params)
  def logDensityBoundNumeric(self, x, mu, sigma):
    if sigma is not None:
      return -(math.log(sigma) + 0.5 * math.log(2 * math.pi))
    elif x is not None and mu is not None:
      # Per the derivative of the log density noted in the
      # gradientOfLogDensity method, the maximum occurs when
      # (x-mu)^2 = sigma^2
      return scipy.stats.norm.logpdf(x, mu, abs(x-mu))
    else:
      raise Exception("Cannot rejection sample psp with unbounded likelihood")

  def simulate(self,args): 
      out=np.zeros(1,len(args.operandValues[1]))
      for mean_variance in args.operandValues[1]:
      return self.simulateNumeric(args.operandValues)
  def gradientOfSimulate(self, args, value, direction):
    # Reverse engineering the behavior of scipy.stats.norm.rvs
    # suggests this gradient is correct.
    (mu, sigma) = args.operandValues
    deviation = (value - mu) / sigma
    return [direction*1, direction*deviation]
  def logDensity(self,x,args): return self.logDensityNumeric(x,args.operandValues)
  def logDensityBound(self, x, args): return self.logDensityBoundNumeric(x, *args.operandValues)

  def hasDeltaKernel(self): return False # have each gkernel control whether it is delta or not
  def getDeltaKernel(self,args): return NormalDriftKernel(args)

  def hasVariationalLKernel(self): return True
  def getParameterScopes(self): return ["REAL","POSITIVE_REAL"]

  def gradientOfLogDensity(self,x,args):
    mu = args.operandValues[0]
    sigma = args.operandValues[1]

    gradX = -(x - mu) / (math.pow(sigma,2))
    gradMu = (x - mu) / (math.pow(sigma,2))
    gradSigma = (math.pow(x - mu,2) - math.pow(sigma,2)) / math.pow(sigma,3)
    # for positive sigma, d(log density)/d(sigma) is <=> zero
    # when math.pow(x - mu,2) <=> math.pow(sigma,2) respectively
    return (gradX,[gradMu,gradSigma])

  def description(self,name):
    return "  (%s mu sigma) samples a normal distribution with mean mu and standard deviation sigma." % name
'''
