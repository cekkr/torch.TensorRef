from .common import VERBOSE_TENSORS_TRACKER, VERBOSE_TENSORS_TRACKER_STATUS
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
        self.tensorByRef = {} # a checker dict, probably useless

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
            if tensorRefId in self.tensorByRef:
                prevTensor = self.tensorByRef[tensorRefId]
                self.uncountTensor(prevTensor)

            tensor = tensorRef.target
            self.tensorRefs[tensorRefId] = tensorRef
            self.refByTensor[id(tensor)] = tensorRef
            self.tensorByRef[tensorRefId] = tensor

        self.tensors[id(tensor)] = tensor
        size = tensor.numel() * tensor.element_size() # in bytes
        if tensor.is_cpu:
            self.numOnCPU += 1
            self.sizeOnCPU += size 
        else:
            self.numOnGPU += 1
            self.sizeOnGPU += size
    
    def uncountTensor(self, tensorRef, countStats = True):
        tensor = tensorRef
        if isinstance(tensorRef, TensorRef):
            tensor = tensorRef.target
            try:
                del self.tensorByRef[id(tensorRef)]
            except:
                pass

        idTensor = id(tensor)

        try:
            del self.tensors[idTensor]
        except Exception as err:
            pass

        if countStats:
            size = tensor.numel() * tensor.element_size() # in bytes
            if tensor.is_cpu:
                self.numOnCPU -= 1
                self.sizeOnCPU -= size
            else:
                self.numOnGPU -= 1
                self.sizeOnGPU -= size
                #tensor.detach()

        try:
            del self.refByTensor[idTensor]
            ref = self.refByTensor[idTensor]
            del self.tensorByRef[id(ref)]
        except:
            pass

        # debug purposes
        #count = sys.getrefcount(tensor)
        #print(count)

    def printStatus(self):
        # Memory limiter
        if self.sizeOnGPU > ((1024 ** 3)*4): #todo: move fixed size to dynamic size
            orderedRefs = sorted(self.tensorRefs.values(), key=lambda x: x.proxyInfo.usageNs)
            if len(orderedRefs) > 0:
                avgNs = sum(x.proxyInfo.usageNs for x in orderedRefs) / len(orderedRefs)
                for ref in orderedRefs:
                    if ref.proxyInfo.usageNs < avgNs:
                        if not ref.proxyInfo.locked:
                            ref.toCPU() #todo: mark as to move
                            pass
                    else:
                        break
            self.gcCollect()

        if VERBOSE_TENSORS_TRACKER_STATUS:
            print('Tensors:\t CPU: '+str(self.numOnCPU)+' \t GPU: '+str(self.numOnGPU))

            cpuGB = self.sizeOnCPU / (1024) ** 3
            gpuGB = self.sizeOnGPU / (1024) ** 3
            print('CPU Size:\t '+str(cpuGB)+'GB \t GPU Size:\t '+str(gpuGB)+'GB')

    def remTensorRef(self, tensorRef):
        try:
            del self.tensorRefs[id(tensorRef)]
            del self.refByTensor[id(tensorRef.target)]
            del self.tensorByRef[id(tensorRef)]
        except Exception as err:
            pass

    def gcCollect(self):
        return
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
            if not tensorRef.proxyInfo.locked:
                countRefs = sys.getrefcount(tensorRef)
                if countRefs <= properties['minRefsTensorRef']: # self.tensorRefs + tensorRef + getrefcount(tensorRef) + tensorRefs + self.refByTensor

                    if VERBOSE_TENSORS_TRACKER:
                        print("Removing unused tensorRef...")

                    countTensorRef = sys.getrefcount(tensorRef.target)
                    if countTensorRef <= properties['minRefsTensor']:
                        self.uncountTensor(tensorRef, True)
                        tensorRef.target = None
                        self.remTensorRef(tensorRef)
                        removes = True
                    else:
                        if VERBOSE_TENSORS_TRACKER:
                            print("TensorRef with still used tensor")
            else:
                if VERBOSE_TENSORS_TRACKER:
                    print("Impossible to remove locked tensorRef")

        # Pretty useless double checking
        '''
        tensors = copy.copy(self.tensors)
        for key, tensor in tensors.items():
            countRefs = sys.getrefcount(tensor)
            if id(tensor) not in self.refByTensor or countRefs <= properties['minRefsTensor']:
                if VERBOSE_TENSORS_TRACKER:
                    print("Removing unused tensor...")

                removes = True
                self.uncountTensor(tensor, False)
        '''

        if removes:
            self.gcCollect()
            #self.calculateSizes() # calculate size from scratch
            self.printStatus()