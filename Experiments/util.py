import ConfigParser
from venture.lite.function import VentureFunction
from venture.lite.sp import SPType
import venture.lite.types as t


Config = ConfigParser.ConfigParser()
def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


def makeObservations(x,y,ripl):
    print(y)
    xString = genSamples(x)
    ripl.observe(xString, array(y.tolist()))

constantType = SPType([t.AnyType()], t.NumberType())
def makeConstFunc(c):
  return VentureFunction(lambda _: c, sp_type=constantType)
def array(xs):
  return t.VentureArrayUnboxed(np.array(xs),  t.NumberType())
covfunctionType = SPType([t.NumberType(), t.NumberType()], t.NumberType())
