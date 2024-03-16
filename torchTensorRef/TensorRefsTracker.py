from .common import VERBOSE_TENSORS_TRACKER
import sys

TensorRef = None

def SetTensorRefType(tr):
    global TensorRef
    TensorRef = tr

class TensorRefsTracker:
    def __init__(self):
        self.numOnCPU = 0
        self.numOnGPU = 0

        self.sizeOnCPU = 0
        self.sizeOnGPU = 0

        self.tensors = {}

    def countTensor(self, tensorRef):
        tensor = tensorRef
        if isinstance(tensorRef, TensorRef):
            tensor = tensorRef.target

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

    def checkTensors(self):
        for key, tensor in self.tensors.items():
            countRefs = sys.getrefcount(tensor)
            if countRefs <= 2: # 1 + self.tensors
                if VERBOSE_TENSORS_TRACKER:
                    print("Removing unused tensor...")

                self.uncountTensor(tensor)
                self.printStatus()