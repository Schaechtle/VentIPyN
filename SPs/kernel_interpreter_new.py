from venture.lite.psp import DeterministicPSP
from venture.lite.value import VentureSymbol

kernelList=['LIN','PER','SE','WN']
#             0     1     2    3
kernelParameterLength={'0':1,'1':3,'2':2,'3':1}

class GrammarInterpreter(DeterministicPSP):

    def simulate(self,args):
        K = args.operandValues[0]
        unsorted_string = K.stuff['name']
        all_parameter=K.stuff['parameter']
        unsorted_string = unsorted_string.replace("x","")
        for i in range(len(kernelList)):
            unsorted_string=unsorted_string.replace(kernelList[i],str(i))
        parameter_dict={}
        PERs = []
        SEs=[]
        last= 0
        index = 0
        #print("unsorted_string")
        #print(unsorted_string)
        for kernel_code in unsorted_string:
            if kernel_code!='+':
                from_value = last
                to = last+kernelParameterLength[kernel_code]
                #print("from_value")
                #print(from_value)
                #print("to")
                #print(to)
                parameter_dict[index]=all_parameter[from_value:to]
                last+=kernelParameterLength[kernel_code]
                if kernel_code=='1':
                    PERs.append(index)
                elif kernel_code=='2':
                    SEs.append(index)
                index+=1
        products = unsorted_string.split('+')
        products.sort()
        products.sort(key = lambda s: len(s))
        out_string=""
        for j in range(len(products)):
            products[j]= ''.join(sorted(products[j]))
            while ("22" in products[j]):
                products[j]=products[j].replace("22","2")
            while (("13" in products[j]) or ("23" in products[j]) ):
                products[j]=products[j].replace("13","3")
                products[j]=products[j].replace("23","3")
            products[j]= "x".join(products[j])
        products.sort()
        products.sort(key = lambda s: len(s))
        for j in range(len(products)):
            out_string+=products[j]
            if j<(len(products)-1):
                out_string+="+"
        while ("0+0+" in out_string):
            out_string=out_string.replace("0+0+","0+")
        if out_string == "0+0":
            out_string = "0"
        if "0" in out_string:
            print(SEs)
        for i in range(len(kernelList)):
            out_string= out_string.replace(str(i),kernelList[i])

        return   VentureSymbol(out_string)  
    