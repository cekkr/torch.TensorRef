from .common import VERBOSE_TENSORS_TRACKER
import sys
import copy
import gc
import torch

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

    def countTensor(self, tensorRef):
        tensor = tensorRef
        if isinstance(tensorRef, TensorRef):
            tensor = tensorRef.target
            self.tensorRefs[id(tensorRef)] = tensorRef

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

        try:
            del self.tensors[id(tensor)]
        except Exception as err:
            pass

        size = tensor.numel() * tensor.element_size() # in bytes
        if tensor.is_cpu:
            self.numOnCPU -= 1
            self.sizeOnCPU -= size 
        else:
            self.numOnGPU -= 1
            self.sizeOnGPU -= size

    def printStatus(self):
        if not VERBOSE_TENSORS_TRACKER:
            return

        print('Tensors:\t CPU: '+str(self.numOnCPU)+' \t GPU: '+str(self.numOnGPU))

        cpuGB = self.sizeOnCPU / (1024) ** 3
        gpuGB = self.sizeOnGPU / (1024) ** 3
        print('CPU Size:\t '+str(cpuGB)+'GB \t GPU Size:\t '+str(gpuGB)+'GB')

    def remTensorRef(self, tensor):
        try:
            del self.tensorRefs[id(tensor)]
        except Exception as err:
            pass

    def checkTensors(self):
        tensorRefs = copy.copy(self.tensorRefs)
        for key, tensor in tensorRefs.items():
            countRefs = sys.getrefcount(tensor)
            if countRefs <= 4: # self.tensors + tensor + getrefcount(tensor) + tensors
                if VERBOSE_TENSORS_TRACKER:
                    print("Removing unused tensorRef...")

                self.uncountTensor(tensor)
                self.remTensorRef(tensor)

        tensors = copy.copy(self.tensors)
        for key, tensor in tensors.items():
            countRefs = sys.getrefcount(tensor)
            if countRefs <= 5:  # self.tensors + tensor + getrefcount(tensor) + tensors
                if VERBOSE_TENSORS_TRACKER:
                    print("Removing unused tensor...")

                self.uncountTensor(tensor)

        gc.collect()
        clearCuda()
        self.printStatus()