from venture.lite.psp import DeterministicPSP
from venture.lite.value import VentureSymbol

kernelList=['LIN','PER','SE','WN']



class GrammarInterpreter(DeterministicPSP):

    def simulate(self,args):
        
        K = args.operandValues[0]
        unsorted_string = K.stuff['name']
        unsorted_string = unsorted_string.replace("x","")
        for i in range(len(kernelList)):
            unsorted_string=unsorted_string.replace(kernelList[i],str(i))
        products = unsorted_string.split('+')
        products.sort()
        products.sort(key = lambda s: len(s))
        outstring=""
        for j in range(len(products)):
            currentProduct= ''.join(sorted(products[j]))
            currentProduct= "x".join(currentProduct)
            outstring+=currentProduct
            if j<(len(products)-1):
                outstring+="+"
        for i in range(len(kernelList)):
            outstring= outstring.replace(str(i),kernelList[i])
        return   VentureSymbol(outstring)  
    