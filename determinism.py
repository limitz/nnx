import torch as _torch
import random as _random
import numpy as _np

class determinism:
    def __init__(self, seed=13):
        self._seed = seed
        self._state = None
        
    def __enter__(self):
        self._state = self.get_state()
        self.seed(self._seed)
        
    def __exit__(self, *exception):
        self.set_state(self._state)

    @staticmethod
    def seed(seed):
        _torch.manual_seed(seed)
        _random.seed(seed)
        _np.random.seed(seed)
        
        #torch.manual_seed also sets seed for devices, so this is not needed per se
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        
    @staticmethod
    def get_state():
        crng = []
        if _torch.cuda.is_available():
            for i in range(_torch.cuda.device_count()):
                crng.append(_torch.cuda.get_rng_state(device="cuda:" + str(i)))
        return dict(
            trng=_torch.get_rng_state(),
            crng=crng,
            rrng=_random.getstate(),
            nrng=_np.random.get_state())
    
    @staticmethod
    def set_state(state, strict=True):
        _torch.set_rng_state(state["trng"])
        _random.setstate(state["rrng"])
        _np.random.set_state(state["nrng"])
        if _torch.cuda.is_available():
            crng = state["crng"]
            dc = _torch.cuda.device_count()
            assert not strict or len(crng) == dc, "cuda device count doesn't match device count in RNG state"
            for i,c in zip(range(dc), crng):
                _torch.cuda.set_rng_state(c, device="cuda:"+str(i))
        