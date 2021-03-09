from __future__ import annotations
import numpy as np

class Schedule:
    
    def __init__(self, M, J):
        """ -----------about M and J"""
        self.M = M
        self.J = J
        
    def __repr__(self):
        repr_ = f'{self.__class__.__name__}( \
                M=np.{np.array_repr(self.M)}, \
                J=np.{np.array_repr(self.J)})'
        return repr_
        
    def __invert__(self) -> Schedule:
        "apply mutation"
        pass
        
    def __mul__(self, S: Schedule) -> List[Schedule]:
        """apply cross over"""
        pass
    