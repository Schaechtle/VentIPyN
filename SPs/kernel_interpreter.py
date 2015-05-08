from venture.lite.psp import DeterministicPSP
from venture.lite.value import VentureSymbol

kernelList=['LIN','PER','SE','WN']
#             0     1     2    3


class GrammarInterpreter(DeterministicPSP):

    def simulate(self,args):
        K = args.operandValues[0]
        unsorted_string = K.stuff['name']
        unsorted_string = args.operar.replace("x","")
        for i in range(len(kernelList)):
            unsorted_string=unsorted_string.replace(kernelList[i],str(i))
        products = unsorted_string.split('+')
        products.sort()
        products.sort(key = lambda s: len(s))
        outstring=""
        for j in range(len(products)):
            products[j]= ''.join(sorted(products[j]))
            print(products[j])
            while ("22" in products[j]):
                products[j]=products[j].replace("22","2")
            while (("13" in products[j]) or ("23" in products[j]) ):
                products[j]=products[j].replace("13","3")
                products[j]=products[j].replace("23","3")
            products[j]= "x".join(products[j])

        products.sort()
        products.sort(key = lambda s: len(s))
        for j in range(len(products)):
            outstring+=products[j]
            if j<(len(products)-1):
                outstring+="+"

        while ("0+0+" in outstring):
            outstring=outstring.replace("0+0+","0+")
        if outstring == "0+0":
            outstring = "0"
        for i in range(len(kernelList)):
            outstring= outstring.replace(str(i),kernelList[i])


        return   VentureSymbol(outstring)  
    