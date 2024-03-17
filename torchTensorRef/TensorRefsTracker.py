from .common import VERBOSE_TENSORS_TRACKER
import sys
import copy
import gc
import torch

from .common import properties

TensorRef = None

def SetTensorRefType(tr):
    global TensorRef
    TensorRef = tr

def clearCuda():
    if torch.cuda.is_available():
        # Clear the cache
        torch.cuda.empty_cache()

        # Optionally, you can reset all CUDA devices to further ensure all memory is freed
        torch.cuda.reset_peak_memory_stats()

class TensorRefsTracker:
    def __init__(self):
        self.numOnCPU = 0
        self.numOnGPU = 0

        self.sizeOnCPU = 0
        self.sizeOnGPU = 0

        self.tensorRefs = {}
        self.tensors = {}
        self.refByTensor = {}
        #self.tensorByRef = {} # a checker dict, useless, remove it. damn.

    def calculateSizes(self):
        self.numOnCPU = 0
        self.numOnGPU = 0
        self.sizeOnCPU = 0
        self.sizeOnGPU = 0

        for key, tensor in self.tensors.items():
            size = tensor.numel() * tensor.element_size()
            if tensor.is_cpu:
                self.numOnCPU += 1
                self.sizeOnCPU += size
            else:
                self.numOnGPU += 1
                self.sizeOnGPU += size

    def countTensor(self, tensorRef):
        tensor = tensorRef

        if isinstance(tensorRef, TensorRef):
            tensorRefId = id(tensorRef)
            '''
            if tensorRefId in self.tensorByRef:
                prevTensor = self.tensorByRef[tensorRefId]
                self.uncountTensor(prevTensor)
            '''
            tensor = tensorRef.target
            self.tensorRefs[tensorRefId] = tensorRef
            self.refByTensor[id(tensor)] = tensorRef
            #self.tensorByRef[tensorRefId] = tensor

        self.tensors[id(tensor)] = tensor
        size = tensor.numel() * tensor.element_size() # in bytes
        if tensor.is_cpu:
            self.numOnCPU += 1
            self.sizeOnCPU += size 
        else:
            self.numOnGPU += 1
            self.sizeOnGPU += size
    
    def uncountTensor(self, tensorRef):
        tensor = tensorRef
        if isinstance(tensorRef, TensorRef):
            tensor = tensorRef.target

        idTensor = id(tensor)

        try:
            del self.tensors[idTensor]
        except Exception as err:
            pass

        if idTensor in self.refByTensor:
            size = tensor.numel() * tensor.element_size() # in bytes
            if tensor.is_cpu:
                self.numOnCPU -= 1
                self.sizeOnCPU -= size
            else:
                self.numOnGPU -= 1
                self.sizeOnGPU -= size
                #tensor.detach()

        try:
            #ref = self.refByTensor[idTensor]
            #del self.tensorByRef[id(ref)]
            del self.refByTensor[idTensor]
        except:
            pass                

    def printStatus(self):
        # Memory limiter
        if self.sizeOnGPU > ((1024 ** 3)*1):
            orderedRefs = sorted(self.tensorRefs.values(), key=lambda x: x.proxyInfo.usageNs)
            if len(orderedRefs) > 0:
                avgNs = sum(x.proxyInfo.usageNs for x in orderedRefs) / len(orderedRefs)
                for ref in orderedRefs:
                    if ref.proxyInfo.usageNs < avgNs:
                        if not ref.proxyInfo.locked:
                            ref.toCPU()
                    else:
                        break
            self.gcCollect()

        if VERBOSE_TENSORS_TRACKER:
            print('Tensors:\t CPU: '+str(self.numOnCPU)+' \t GPU: '+str(self.numOnGPU))

            cpuGB = self.sizeOnCPU / (1024) ** 3
            gpuGB = self.sizeOnGPU / (1024) ** 3
            print('CPU Size:\t '+str(cpuGB)+'GB \t GPU Size:\t '+str(gpuGB)+'GB')

    def remTensorRef(self, tensorRef):
        try:
            del self.tensorRefs[id(tensorRef)]
            del self.refByTensor[id(tensorRef.target)]
        except Exception as err:
            pass

    def gcCollect(self):
        gc.collect()
        clearCuda()

    def countTensorRefReferences(self, tensorRef):
        # To returns effectively used references
        #tensorRefs = copy.copy(self.tensorRefs)
        #tensors = copy.copy(self.tensors)
        return [sys.getrefcount(tensorRef), sys.getrefcount(tensorRef.target)]

    def checkTensors(self):
        removes = False

        tensorRefs = copy.copy(self.tensorRefs)
        for key, tensorRef in tensorRefs.items():
            countRefs = sys.getrefcount(tensorRef)
            if countRefs <= properties['minRefsTensorRef']: # self.tensorRefs + tensorRef + getrefcount(tensorRef) + tensorRefs + self.refByTensor
                if not tensorRef.proxyInfo.locked:
                    if VERBOSE_TENSORS_TRACKER:
                        print("Removing unused tensorRef...")

                    removes = True
                    #self.uncountTensor(tensorRef)
                    self.remTensorRef(tensorRef)
                else:
                    print("Impossible to remove locked tensorRef")

        tensors = copy.copy(self.tensors)
        for key, tensor in tensors.items():
            countRefs = sys.getrefcount(tensor)
            if countRefs <= properties['minRefsTensor'] + 1:  # self.tensors + tensor + getrefcount(tensor) + tensors
                if VERBOSE_TENSORS_TRACKER:
                    print("Removing unused tensor...")

                removes = True
                #self.uncountTensor(tensor)

        if removes:
            self.gcCollect()

            #self.calculateSizes() # calculate size from scratch
            self.printStatus()