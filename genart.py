"""
Generative Art Module (3.2)
===========================

Implements quaternion-based expression trees for artifact
generation. Each artifact is a mathematical expression that maps
2D pixel coordinates to quaternion values, which are then
converted to RGB colors.

Key components:
    - QuaternionTensor: GPU-accelerated quaternion arithmetic
    - ExpressionNode: CPU-side expression tree (breed/mutate)
    - VectorizedImageGenerator: GPU stack-machine renderer

The operation set below corresponds to Tables 1 and 3 in the
thesis. Operations are classified as terminal (leaf nodes),
unary (one child), or binary (two children).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union, Callable, Tuple, Any, Dict
from PIL import Image
import random
import operator
from enum import Enum, auto, IntEnum
import logging
import os
from datetime import datetime
import json
import torch
import time
from timing_utils import time_it

logger = logging.getLogger(__name__)

# --- Quaternion Math (GPU-Optimized, 3.2.1) ---
# Quaternions extend complex numbers with four components [w,x,y,z].
# Using quaternion algebra for image generation produces richer
# color interactions than scalar or complex math alone, because
# the non-commutative Hamilton product couples all four channels.
@dataclass
class QuaternionTensor:
    """Quaternion implementation using PyTorch tensors"""
    data: torch.Tensor  # shape [..., 4] for [w,x,y,z]

    @property
    def device(self):
        return self.data.device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'QuaternionTensor':
        if tensor.shape[-1] != 4:
            raise ValueError("Last dimension must be 4 for quaternion values")
        return cls(tensor)

    def to(self, device: torch.device) -> 'QuaternionTensor':
        return QuaternionTensor(self.data.to(device))

    @property
    def w(self) -> torch.Tensor: return self.data[..., 0]
    @property
    def x(self) -> torch.Tensor: return self.data[..., 1]
    @property
    def y(self) -> torch.Tensor: return self.data[..., 2]
    @property
    def z(self) -> torch.Tensor: return self.data[..., 3]

    def __add__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        return QuaternionTensor(self.data + other.data)

    def __sub__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        return QuaternionTensor(self.data - other.data)

    def __mul__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return QuaternionTensor(torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1))

    def __truediv__(self, other: 'QuaternionTensor') -> 'QuaternionTensor':
            denom = other.norm_squared().unsqueeze(-1)
            mask = denom > 1e-6
            recip = other.conjugate()
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            result = self * QuaternionTensor(recip.data / safe_denom)
            result.data = torch.where(mask, result.data, torch.zeros_like(result.data))
            return result

    def __neg__(self) -> 'QuaternionTensor':
        return QuaternionTensor(-self.data)

    def rotate(self, angle: float) -> 'QuaternionTensor':
        c = torch.cos(torch.tensor(angle/2, device=self.device, dtype=self.data.dtype))
        s = torch.sin(torch.tensor(angle/2, device=self.device, dtype=self.data.dtype))
        return QuaternionTensor(torch.stack([
            c*self.w - s*self.x, c*self.x + s*self.w,
            c*self.y - s*self.z, c*self.z + s*self.y
        ], dim=-1))

    def floor(self) -> 'QuaternionTensor':
        return QuaternionTensor(torch.floor(self.data))

    def modulo(self, mod: float) -> 'QuaternionTensor':
        return QuaternionTensor(torch.remainder(self.data, mod))

    def conjugate(self) -> 'QuaternionTensor':
        return QuaternionTensor(torch.stack([self.w, -self.x, -self.y, -self.z], dim=-1))

    def norm_squared(self) -> torch.Tensor:
        return torch.sum(self.data * self.data, dim=-1)

    def normalize(self) -> 'QuaternionTensor':
            norm = torch.sqrt(self.norm_squared()).unsqueeze(-1)
            mask = norm > 1e-6
            safe_norm = torch.where(mask, norm, torch.ones_like(norm))
            result = QuaternionTensor(self.data / safe_norm)
            result.data = torch.where(mask, result.data, torch.zeros_like(result.data))
            return result

    def to_rgb(self) -> torch.Tensor:
        values = torch.clamp(self.data[..., 1:4].to(torch.float32), -10.0, 10.0)
        rgb = 255.0 / (1.0 + torch.exp(-values))
        return rgb.to(torch.uint8)

# --- Vectorized Quaternion Operations (Tables 1 & 3, 3.2) ---
# These functions form the operation vocabulary for expression
# trees. Each maps QuaternionTensor → QuaternionTensor.
# Terminal ops: coord, Q_I, Q_J, Q_K, golden_ratio
# Unary ops: sin, cos, tan, exp, log, sqrt, abs, inv, cube,
#            sinh, cosh, conj, norm, rot45, floor, mod2,
#            spiral, wave, ripple, swirl, iexp, ilog, isin,
#            rolR, power
# Binary ops: add, sub, mul, div, blend, imin, imax
GOLDEN_RATIO = (1 + torch.sqrt(torch.tensor(5.0))).clone().detach() / 2
def q_sin(q: QuaternionTensor) -> QuaternionTensor: return QuaternionTensor(torch.sin(q.data))
def q_cos(q: QuaternionTensor) -> QuaternionTensor: return QuaternionTensor(torch.cos(q.data))
def q_exp(q: QuaternionTensor) -> QuaternionTensor: 
    clamped_data = torch.clamp(q.data, -10.0, 10.0)
    return QuaternionTensor(torch.exp(clamped_data))
def q_log(q: QuaternionTensor) -> QuaternionTensor:
    abs_data = torch.abs(q.data)
    safe_data = torch.where(abs_data > 1e-6, abs_data, torch.tensor(1e-6, device=q.device, dtype=q.data.dtype))
    return QuaternionTensor(torch.log(safe_data))
def q_sqrt(q: QuaternionTensor) -> QuaternionTensor:
    return QuaternionTensor(torch.sqrt(torch.abs(q.data)))
def q_tan(q: QuaternionTensor) -> QuaternionTensor: return q_sin(q) * q_cos(q).conjugate()
def q_abs(q: QuaternionTensor) -> QuaternionTensor: return QuaternionTensor(torch.abs(q.data))
def q_floor(q: QuaternionTensor) -> QuaternionTensor: return q.floor()
def q_power(q: QuaternionTensor) -> QuaternionTensor:
    """Raises quaternion to the 3rd power via exp/log (scalar exponentiation).
    
    Distinct from q_cube which uses Hamilton product (q * q * q).
    The exp/log path computes scalar exponentiation, which produces 
    different results for non-unit quaternions.
    """
    p_tensor = torch.tensor(3.0, device=q.device, dtype=q.data.dtype)
    return q_exp(q_log(q) * QuaternionTensor(torch.full_like(q.data, p_tensor)))
def q_inverse(q: QuaternionTensor) -> QuaternionTensor:
    norm_squared = q.norm_squared().unsqueeze(-1)
    mask = norm_squared > 1e-6
    recip = q.conjugate()
    safe_norm = torch.where(mask, norm_squared, torch.ones_like(norm_squared))
    return QuaternionTensor(recip.data / safe_norm)
def q_cube(q: QuaternionTensor) -> QuaternionTensor: return q * q * q
def q_sinh(q: QuaternionTensor) -> QuaternionTensor: return (q_exp(q) - q_exp(-q)) * QuaternionTensor(torch.full_like(q.data, 0.5))
def q_cosh(q: QuaternionTensor) -> QuaternionTensor: return (q_exp(q) + q_exp(-q)) * QuaternionTensor(torch.full_like(q.data, 0.5))
def coord_to_quaternion(x: torch.Tensor, y: torch.Tensor) -> QuaternionTensor:
    zeros = torch.zeros_like(x)
    return QuaternionTensor(torch.stack([zeros, x, y, zeros], dim=-1))
def q_add(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor: return x + y
def q_sub(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor: return x - y
def q_mul(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor: return x * y
def q_div(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor: return x / y
def q_conjugate(q: QuaternionTensor) -> QuaternionTensor: return q.conjugate()
def q_normalize(q: QuaternionTensor) -> QuaternionTensor: return q.normalize()
def q_rotate45(q: QuaternionTensor) -> QuaternionTensor: return q.rotate(torch.pi/4)
def q_mod2(q: QuaternionTensor) -> QuaternionTensor: return q.modulo(2.0)
def q_spiral(q: QuaternionTensor) -> QuaternionTensor:
    clamped_data = torch.clamp(q.data, -10.0, 10.0)
    return QuaternionTensor(clamped_data * torch.exp(clamped_data * 0.5))
def q_wave(q: QuaternionTensor) -> QuaternionTensor: return q_sin(q) * q_cos(q.rotate(torch.pi/3))
def q_blend(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    t = 0.5 * (1 + torch.sin(x.norm_squared()))
    return QuaternionTensor(t.unsqueeze(-1) * x.data + (1-t).unsqueeze(-1) * y.data)
def q_ripple(q: QuaternionTensor) -> QuaternionTensor:
    r = torch.sqrt(q.x**2 + q.y**2)
    return q_sin(QuaternionTensor(r.unsqueeze(-1) * torch.ones_like(q.data)))
def q_swirl(q: QuaternionTensor) -> QuaternionTensor:
    r = torch.sqrt(q.x**2 + q.y**2)
    theta = torch.atan2(q.y, q.x) + r
    return QuaternionTensor(torch.stack([torch.zeros_like(r), r * torch.cos(theta), r * torch.sin(theta), torch.zeros_like(r)], dim=-1))
def q_rolR(q: QuaternionTensor) -> QuaternionTensor: return QuaternionTensor(torch.roll(q.data, 1, dims=-1))
Q_I_func = lambda: QuaternionTensor(torch.tensor([0., 1., 0., 0.]))
Q_J_func = lambda: QuaternionTensor(torch.tensor([0., 0., 1., 0.]))
Q_K_func = lambda: QuaternionTensor(torch.tensor([0., 0., 0., 1.]))
def q_iexp(q: QuaternionTensor) -> QuaternionTensor: return q_exp(Q_I_func().to(q.device) * q)
def q_ilog(q: QuaternionTensor) -> QuaternionTensor: return q_log(Q_I_func().to(q.device) * q)
def q_isin(q: QuaternionTensor) -> QuaternionTensor: return q_sin(Q_I_func().to(q.device) * q)
def q_imin(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    norm_x = x.norm_squared().unsqueeze(-1)
    norm_y = y.norm_squared().unsqueeze(-1)
    return QuaternionTensor(torch.where(norm_x < norm_y, x.data, y.data))
def q_imax(x: QuaternionTensor, y: QuaternionTensor) -> QuaternionTensor:
    norm_x = x.norm_squared().unsqueeze(-1)
    norm_y = y.norm_squared().unsqueeze(-1)
    return QuaternionTensor(torch.where(norm_x > norm_y, x.data, y.data))


# --- Expression Encoder ---
# Maps Python functions ↔ integer opcodes for the GPU stack
# machine. This indirection lets expression trees be serialized
# as integer sequences (RPN) for batched GPU evaluation.
class OpType(Enum):
    TERMINAL = auto()
    UNARY = auto()
    BINARY = auto()

class OpCode(IntEnum):
    NO_OP = -1
    COORD = 0
    Q_I = auto()
    Q_J = auto()
    Q_K = auto()
    GOLDEN_RATIO = auto()
    SIN = auto()
    COS = auto()
    TAN = auto()
    EXP = auto()
    LOG = auto()
    SQRT = auto()
    ABS = auto()
    INV = auto()
    CUBE = auto()
    SINH = auto()
    COSH = auto()
    CONJ = auto()
    NORM = auto()
    ROT45 = auto()
    FLOOR = auto()
    MOD2 = auto()
    SPIRAL = auto()
    WAVE = auto()
    RIPPLE = auto()
    SWIRL = auto()
    IEXP = auto()
    ILOG = auto()
    ISIN = auto()
    ROLR = auto()
    POWER = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    BLEND = auto()
    IMIN = auto()
    IMAX = auto()

class ExpressionEncoder:
    def __init__(self, include_floor_mod2: bool = True):
        self.op_map: Dict[Callable, Tuple[OpCode, OpType]] = {
            coord_to_quaternion: (OpCode.COORD, OpType.TERMINAL),
            Q_I_func: (OpCode.Q_I, OpType.TERMINAL),
            Q_J_func: (OpCode.Q_J, OpType.TERMINAL),
            Q_K_func: (OpCode.Q_K, OpType.TERMINAL),
            (lambda: QuaternionTensor(torch.tensor([GOLDEN_RATIO, 0., 0., 0.]))): (OpCode.GOLDEN_RATIO, OpType.TERMINAL),
            q_sin: (OpCode.SIN, OpType.UNARY), q_cos: (OpCode.COS, OpType.UNARY),
            q_tan: (OpCode.TAN, OpType.UNARY), q_exp: (OpCode.EXP, OpType.UNARY),
            q_log: (OpCode.LOG, OpType.UNARY), q_sqrt: (OpCode.SQRT, OpType.UNARY),
            q_abs: (OpCode.ABS, OpType.UNARY), q_inverse: (OpCode.INV, OpType.UNARY),
            q_cube: (OpCode.CUBE, OpType.UNARY), q_sinh: (OpCode.SINH, OpType.UNARY),
            q_cosh: (OpCode.COSH, OpType.UNARY), q_conjugate: (OpCode.CONJ, OpType.UNARY),
            q_normalize: (OpCode.NORM, OpType.UNARY), q_rotate45: (OpCode.ROT45, OpType.UNARY),
            q_spiral: (OpCode.SPIRAL, OpType.UNARY), q_wave: (OpCode.WAVE, OpType.UNARY),
            q_ripple: (OpCode.RIPPLE, OpType.UNARY), q_swirl: (OpCode.SWIRL, OpType.UNARY),
            q_iexp: (OpCode.IEXP, OpType.UNARY), q_ilog: (OpCode.ILOG, OpType.UNARY),
            q_isin: (OpCode.ISIN, OpType.UNARY), q_rolR: (OpCode.ROLR, OpType.UNARY),
            q_power: (OpCode.POWER, OpType.UNARY),
            q_add: (OpCode.ADD, OpType.BINARY), q_sub: (OpCode.SUB, OpType.BINARY),
            q_mul: (OpCode.MUL, OpType.BINARY), q_div: (OpCode.DIV, OpType.BINARY),
            q_blend: (OpCode.BLEND, OpType.BINARY), q_imin: (OpCode.IMIN, OpType.BINARY),
            q_imax: (OpCode.IMAX, OpType.BINARY),
        }
        if include_floor_mod2:
            self.op_map[q_floor] = (OpCode.FLOOR, OpType.UNARY)
            self.op_map[q_mod2] = (OpCode.MOD2, OpType.UNARY)

        self.func_to_op = {func: op_tuple[0] for func, op_tuple in self.op_map.items()}
        self.op_to_type = {op_tuple[0]: op_tuple[1] for op_tuple in self.op_map.values()}
        self.op_to_func: Dict[OpCode, Callable] = {op_tuple[0]: func for func, op_tuple in self.op_map.items()}

        self.func_to_name: Dict[Callable, str] = {func: op_tuple[0].name.lower() for func, op_tuple in self.op_map.items()}
        self.func_to_name[coord_to_quaternion] = 'coord'
        self.func_to_name[Q_I_func] = 'i'
        self.func_to_name[Q_J_func] = 'j'
        self.func_to_name[Q_K_func] = 'k'
        phi_func = next(f for f, t in self.op_map.items() if t[0] == OpCode.GOLDEN_RATIO)
        self.func_to_name[phi_func] = 'phi'
        self.func_to_name[q_rotate45] = 'rot45'
        self.func_to_name[q_rolR] = 'rolR'
        if q_mod2 in self.op_map:
            self.func_to_name[q_mod2] = 'mod2'
        if q_floor in self.op_map:
            self.func_to_name[q_floor] = 'floor'

        self.TERMINAL_OPS: List[Tuple[Callable, OpType]] = [
            (func, op_tuple[1]) for func, op_tuple in self.op_map.items()
            if op_tuple[1] == OpType.TERMINAL
        ]
        
        NON_TERMINAL_OPS: List[Tuple[Callable, OpType]] = [
            (func, op_tuple[1]) for func, op_tuple in self.op_map.items()
            if op_tuple[1] != OpType.TERMINAL
        ]
        self.ALL_OPERATORS = self.TERMINAL_OPS + NON_TERMINAL_OPS
        self.NON_TERMINAL_OPS = NON_TERMINAL_OPS

DEFAULT_ENCODER = ExpressionEncoder(include_floor_mod2=True)

def configure_expression_operators(enable_floor_mod2: bool = False):
    """Configure which optional operators are available for expression trees."""
    global DEFAULT_ENCODER
    DEFAULT_ENCODER = ExpressionEncoder(include_floor_mod2=enable_floor_mod2)

# --- Expression Tree (CPU-based, 3.2.2) ---
# Trees are manipulated on CPU (breeding, mutation) and only
# converted to RPN integer programs for GPU rendering.
@dataclass
class ExpressionNode:
    operator: Callable
    op_type: OpType
    left: Optional['ExpressionNode'] = None
    right: Optional['ExpressionNode'] = None

    def to_rpn(self) -> List[int]:
        rpn_sequence = []
        def post_order_traverse(node):
            if node.left: post_order_traverse(node.left)
            if node.right: post_order_traverse(node.right)
            rpn_sequence.append(DEFAULT_ENCODER.func_to_op[node.operator])
        post_order_traverse(self)
        return rpn_sequence

    def to_string(self) -> str:
        op_name = DEFAULT_ENCODER.func_to_name.get(self.operator, "unknown")
        if self.op_type == OpType.TERMINAL:
            return op_name
        elif self.op_type == OpType.UNARY:
            left_str = self.left.to_string() if self.left else ""
            return f"({op_name} {left_str})"
        elif self.op_type == OpType.BINARY:
            left_str = self.left.to_string() if self.left else ""
            right_str = self.right.to_string() if self.right else ""
            return f"({op_name} {left_str} {right_str})"
        return "invalid_expression"

    def mutate(self, rate: float = 0.1, max_depth: Optional[int] = None, _current_depth: int = 0):
        """Mutate this expression tree in-place (3.2.2).
        
        Each node has `rate` probability of being replaced with a
        random operator. max_depth prevents unbounded tree growth
        by forcing terminal nodes at the depth limit.
        """
        if random.random() < rate:
            # At max_depth, only allow operators that do not add child nodes,
            # preventing expression trees from growing unboundedly over many steps.
            if max_depth is not None and _current_depth >= max_depth:
                terminal_ops = [(op, ot) for op, ot in DEFAULT_ENCODER.ALL_OPERATORS
                                if ot == OpType.TERMINAL]
                new_op, new_op_type = random.choice(terminal_ops if terminal_ops
                                                     else DEFAULT_ENCODER.ALL_OPERATORS)
            else:
                new_op, new_op_type = random.choice(DEFAULT_ENCODER.ALL_OPERATORS)

            self.operator = new_op
            self.op_type = new_op_type

            # Ensure structural validity after operator change
            if new_op_type == OpType.BINARY:
                if not self.left:  self.left  = ExpressionNode.create_random(depth=1)
                if not self.right: self.right = ExpressionNode.create_random(depth=1)
            elif new_op_type == OpType.UNARY:
                if not self.left: self.left = ExpressionNode.create_random(depth=1)
                self.right = None
            elif new_op_type == OpType.TERMINAL:
                self.left = None
                self.right = None

        if self.left:  self.left.mutate(rate,  max_depth, _current_depth + 1)
        if self.right: self.right.mutate(rate, max_depth, _current_depth + 1)

    def breed(self, other: 'ExpressionNode') -> 'ExpressionNode':
        """Subtree crossover between two expression trees (3.2.2).
        
        Randomly selects a base (self or other), picks a random
        node in the base, and grafts a random subtree from the
        donor. This is the primary artifact variation mechanism.
        """
        new_expr = self._copy() if random.random() < 0.5 else other._copy()
        donor = other._copy() if random.random() < 0.5 else self._copy()
        
        target_node = new_expr._random_node()
        donor_node = donor._random_node()

        if random.random() < 0.5 and target_node.left:
            target_node.left = donor_node
        elif target_node.right:
            target_node.right = donor_node
        else:
             target_node.left = donor_node        
        
        return new_expr

    def _copy(self) -> 'ExpressionNode':
        return ExpressionNode(
            operator=self.operator,
            op_type=self.op_type,
            left=self.left._copy() if self.left else None,
            right=self.right._copy() if self.right else None
        )

    def _random_node(self) -> 'ExpressionNode':
        nodes = []
        def collect(node):
            nodes.append(node)
            if node.left: collect(node.left)
            if node.right: collect(node.right)
        collect(self)
        return random.choice(nodes)

    @classmethod
    def create_random(cls, depth: int = 3) -> 'ExpressionNode':
        """Create a random expression tree of given depth (3.2.2).
        
        Recursively builds a tree: non-terminal ops at inner
        nodes, coord terminals at leaves (depth=0).
        
        Note: The `depth` parameter here is set by agent.gen_depth.
        See scheduler._initialize_agents for the actual range used.
        """
        if depth <= 0:
            op, op_type = coord_to_quaternion, OpType.TERMINAL
        else:
            op, op_type = random.choice(DEFAULT_ENCODER.NON_TERMINAL_OPS)

        node = cls(op, op_type)
        if op_type in [OpType.UNARY, OpType.BINARY]:
            node.left = cls.create_random(depth - 1)
        if op_type == OpType.BINARY:
            node.right = cls.create_random(depth - 1)
        return node

class VectorizedImageGenerator:
    """GPU stack-machine that renders expression trees to images.
    
    Converts RPN-encoded expression trees into (B, 3, H, W) RGB
    tensors. All expressions in a batch are evaluated in lockstep
    across a shared stack, enabling massive GPU parallelism.
    
    Paper (3.2): Images are 32x32 pixels.
    """
    def __init__(self, width: int = 64, height: int = 64, 
                 device: Optional[torch.device] = None, 
                 use_static_noise: bool = False):
        self.width = width
        self.height = height
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_static_noise = use_static_noise
        
        # Pre-compute coordinate grids (H, W, 4)
        x = (torch.linspace(0, width - 1, width, device=self.device) - width / 2) / (min(width, height) / 4)
        y = (torch.linspace(0, height - 1, height, device=self.device) - height / 2) / (min(width, height) / 4)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        self.coords_q_data = coord_to_quaternion(X, Y).data.unsqueeze(0)

        # Pre-allocate constants tensor for fast indexing
        self.constants_map = {}
        for code in [OpCode.Q_I, OpCode.Q_J, OpCode.Q_K]:
            val = DEFAULT_ENCODER.op_to_func[code]().data
            self.constants_map[code.value] = val.to(self.device).view(1, 1, 1, 4).expand(1, height, width, 4)
        
        phi_val = torch.tensor([GOLDEN_RATIO, 0., 0., 0.], device=self.device)
        self.constants_map[OpCode.GOLDEN_RATIO.value] = phi_val.view(1, 1, 1, 4).expand(1, height, width, 4)

        self.op_type_lookup = {op.value: type_ for op, type_ in DEFAULT_ENCODER.op_to_type.items()}
        self.func_lookup = {op.value: func for op, func in DEFAULT_ENCODER.op_to_func.items()}

    @torch.no_grad()
    @time_it
    def generate_batch(self, expressions: List[ExpressionNode], use_amp=True) -> torch.Tensor:
        """
        Generates images using a 'Vertical' Stack Machine approach.
        Dynamically allocates stack memory to prevent overflows.
        """
        batch_size = len(expressions)
        if batch_size == 0:
            return torch.empty(0, 3, self.height, self.width, device=self.device)

        if self.use_static_noise:
            random_rgb = torch.randint(0, 256, (batch_size, 3, self.height, self.width), 
                                      device=self.device, dtype=torch.uint8)
            return random_rgb.float() / 255.0

        rpn_lists = [expr.to_rpn() for expr in expressions]
        max_len = max(len(p) for p in rpn_lists)
        
        # --- Dynamic Stack Sizing ---
        # Calculate exactly how deep the stack needs to be to avoid crashing
        max_stack_depth = 0
        for prog in rpn_lists:
            current_depth = 0
            for op_val in prog:
                op_type = self.op_type_lookup[OpCode(op_val).value]
                if op_type == OpType.TERMINAL:
                    current_depth += 1
                elif op_type == OpType.BINARY:
                    current_depth -= 1
                # Unary ops don't change depth
                max_stack_depth = max(max_stack_depth, current_depth)
        
        # Safety buffer to prevent stack overflows
        safe_stack_depth = max_stack_depth + 5

        program_tensor = torch.full((batch_size, max_len), -1, device=self.device, dtype=torch.long)
        for i, prog in enumerate(rpn_lists):
            program_tensor[i, :len(prog)] = torch.tensor(prog, device=self.device)

        # Allocate dynamically sized stack
        stack = torch.zeros(batch_size, safe_stack_depth, self.height, self.width, 4, device=self.device)
        sp = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        batch_indices = torch.arange(batch_size, device=self.device)

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp and self.device.type=='cuda'):
            for t in range(max_len):
                current_ops = program_tensor[:, t]
                unique_ops = torch.unique(current_ops)
                
                for op_val in unique_ops:
                    op_val_item = op_val.item()
                    if op_val_item == -1: continue
                        
                    mask = (current_ops == op_val)
                    active_indices = batch_indices[mask]
                    if len(active_indices) == 0: continue

                    op_type = self.op_type_lookup[op_val_item]

                    if op_type == OpType.TERMINAL:
                        current_sp = sp[active_indices]
                        if op_val_item == 0: # OpCode.COORD
                            val = self.coords_q_data.expand(len(active_indices), -1, -1, -1)
                            stack[active_indices, current_sp] = val
                        else:
                            val = self.constants_map[op_val_item].expand(len(active_indices), -1, -1, -1)
                            stack[active_indices, current_sp] = val
                        sp[active_indices] += 1

                    elif op_type == OpType.UNARY:
                        current_idx = sp[active_indices] - 1
                        operand = QuaternionTensor(stack[active_indices, current_idx])
                        func = self.func_lookup[op_val_item]
                        res = func(operand)
                        stack[active_indices, current_idx] = res.data

                    elif op_type == OpType.BINARY:
                        right_idx = sp[active_indices] - 1
                        left_idx = sp[active_indices] - 2
                        right = QuaternionTensor(stack[active_indices, right_idx])
                        left = QuaternionTensor(stack[active_indices, left_idx])
                        func = self.func_lookup[op_val_item]
                        res = func(left, right)
                        stack[active_indices, left_idx] = res.data
                        sp[active_indices] -= 1

        final_idx = sp - 1
        # Gather results safely
        final_results = stack[batch_indices, final_idx]

        final_quaternions = QuaternionTensor(final_results)
        rgb_batch = final_quaternions.to_rgb()
        
        return rgb_batch.permute(0, 3, 1, 2).float() / 255.0