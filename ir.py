"""
FLUX IR — Intermediate Representation between source and bytecode.

Provides a structured, human-readable IR that's higher-level than bytecode
but lower-level than natural language. Think LLVM IR for FLUX.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class IROp(Enum):
    CONST = "const"       # load constant
    LOAD = "load"         # load register
    STORE = "store"       # store to register
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MOD = "mod"
    NEG = "neg"
    CMP_EQ = "cmp_eq"
    CMP_LT = "cmp_lt"
    CMP_GT = "cmp_gt"
    JUMP = "jump"
    JUMP_IF = "jump_if"
    JUMP_IF_NOT = "jump_if_not"
    CALL = "call"
    RET = "ret"
    PUSH = "push"
    POP = "pop"
    LABEL = "label"
    NOP = "nop"
    HALT = "halt"


@dataclass
class IRInstruction:
    op: IROp
    dest: Optional[str] = None   # destination (virtual register)
    args: List = field(default_factory=list)  # source args
    label: str = ""
    comment: str = ""
    
    def to_string(self) -> str:
        if self.op == IROp.LABEL:
            return f"{self.label}:"
        parts = []
        if self.dest:
            parts.append(self.dest)
            parts.append("=")
        parts.append(self.op.value)
        if self.args:
            parts.append(", ".join(str(a) for a in self.args))
        line = " ".join(parts)
        if self.comment:
            line += f"  ; {self.comment}"
        return line


@dataclass
class IRProgram:
    name: str
    instructions: List[IRInstruction]
    constants: Dict[str, int] = field(default_factory=dict)
    labels: Dict[str, int] = field(default_factory=dict)
    
    def to_string(self) -> str:
        lines = [f"; IR Program: {self.name}", ""]
        if self.constants:
            lines.append("; Constants:")
            for k, v in self.constants.items():
                lines.append(f"  {k} = {v}")
            lines.append("")
        for inst in self.instructions:
            lines.append("  " + inst.to_string())
        return "\n".join(lines)
    
    def to_bytecode(self) -> List[int]:
        """Compile IR to FLUX bytecode."""
        # Map virtual registers to real registers
        vreg_map = {}
        next_reg = 0
        
        def get_reg(name):
            nonlocal next_reg
            if isinstance(name, int):
                return name
            if name not in vreg_map:
                vreg_map[name] = next_reg
                next_reg += 1
            return vreg_map[name]
        
        # Two passes: first resolve labels, then emit
        label_offsets = {}
        offset = 0
        for inst in self.instructions:
            if inst.op == IROp.LABEL:
                label_offsets[inst.label] = offset
                continue
            if inst.op in (IROp.CONST, IROp.NOP):
                offset += 3  # MOVI
            elif inst.op in (IROp.NEG, IROp.PUSH, IROp.POP):
                offset += 2
            elif inst.op in (IROp.ADD, IROp.SUB, IROp.MUL, IROp.DIV, IROp.MOD,
                           IROp.CMP_EQ, IROp.CMP_LT, IROp.CMP_GT, IROp.STORE):
                offset += 4
            elif inst.op in (IROp.JUMP_IF, IROp.JUMP_IF_NOT):
                offset += 4
            elif inst.op == IROp.HALT:
                offset += 1
            else:
                offset += 1
        
        # Emit bytecode
        bc = []
        for inst in self.instructions:
            if inst.op == IROp.LABEL:
                continue
            elif inst.op == IROp.CONST:
                rd = get_reg(inst.dest)
                val = inst.args[0] if inst.args else 0
                val = val & 0xFF
                bc.extend([0x18, rd, val])
            elif inst.op == IROp.ADD:
                rd = get_reg(inst.dest)
                rs1 = get_reg(inst.args[0])
                rs2 = get_reg(inst.args[1])
                bc.extend([0x20, rd, rs1, rs2])
            elif inst.op == IROp.SUB:
                rd = get_reg(inst.dest)
                rs1 = get_reg(inst.args[0])
                rs2 = get_reg(inst.args[1])
                bc.extend([0x21, rd, rs1, rs2])
            elif inst.op == IROp.MUL:
                rd = get_reg(inst.dest)
                rs1 = get_reg(inst.args[0])
                rs2 = get_reg(inst.args[1])
                bc.extend([0x22, rd, rs1, rs2])
            elif inst.op == IROp.CMP_EQ:
                rd = get_reg(inst.dest)
                rs1 = get_reg(inst.args[0])
                rs2 = get_reg(inst.args[1])
                bc.extend([0x2C, rd, rs1, rs2])
            elif inst.op == IROp.NEG:
                rd = get_reg(inst.dest)
                bc.extend([0x0B, rd])
            elif inst.op == IROp.PUSH:
                rs = get_reg(inst.args[0])
                bc.extend([0x0C, rs])
            elif inst.op == IROp.POP:
                rd = get_reg(inst.args[0])
                bc.extend([0x0D, rd])
            elif inst.op == IROp.JUMP_IF_NOT:
                rs = get_reg(inst.args[0])
                target = label_offsets.get(inst.args[1], 0)
                offset_from_here = target - len(bc)
                off_byte = offset_from_here & 0xFF
                bc.extend([0x3D, rs, off_byte, 0])
            elif inst.op == IROp.JUMP_IF:
                rs = get_reg(inst.args[0])
                target = label_offsets.get(inst.args[1], 0)
                offset_from_here = target - len(bc)
                off_byte = offset_from_here & 0xFF
                bc.extend([0x3C, rs, off_byte, 0])
            elif inst.op == IROp.HALT:
                bc.append(0x00)
        
        return bc


def from_factorial(n) -> IRProgram:
    """Generate IR for factorial computation."""
    return IRProgram(
        name="factorial",
        constants={"n": n, "one": 1, "zero": 0},
        instructions=[
            IRInstruction(IROp.CONST, "n_reg", [n], comment="load n"),
            IRInstruction(IROp.CONST, "result", [1], comment="acc = 1"),
            IRInstruction(IROp.LABEL, label="loop"),
            IRInstruction(IROp.CMP_EQ, "done", ["n_reg", "zero"], comment="n == 0?"),
            IRInstruction(IROp.JUMP_IF, args=["done", "exit"]),
            IRInstruction(IROp.MUL, "result", ["result", "n_reg"], comment="result *= n"),
            IRInstruction(IROp.CONST, "tmp", [1]),
            IRInstruction(IROp.SUB, "n_reg", ["n_reg", "tmp"], comment="n--"),
            IRInstruction(IROp.JUMP, args=["loop"]),
            IRInstruction(IROp.LABEL, label="exit"),
            IRInstruction(IROp.HALT),
        ]
    )


def from_fibonacci(n) -> IRProgram:
    """Generate IR for fibonacci."""
    return IRProgram(
        name="fibonacci",
        constants={"n": n},
        instructions=[
            IRInstruction(IROp.CONST, "a", [0]),
            IRInstruction(IROp.CONST, "b", [1]),
            IRInstruction(IROp.CONST, "count", [n]),
            IRInstruction(IROp.CONST, "zero", [0]),
            IRInstruction(IROp.LABEL, label="loop"),
            IRInstruction(IROp.CMP_EQ, "done", ["count", "zero"]),
            IRInstruction(IROp.JUMP_IF, args=["done", "exit"]),
            IRInstruction(IROp.ADD, "c", ["a", "b"]),
            IRInstruction(IROp.CONST, "tmp", [0]),
            IRInstruction(IROp.STORE, "a", ["b"]),
            IRInstruction(IROp.STORE, "b", ["c"]),
            IRInstruction(IROp.CONST, "one", [1]),
            IRInstruction(IROp.SUB, "count", ["count", "one"]),
            IRInstruction(IROp.JUMP, args=["loop"]),
            IRInstruction(IROp.LABEL, label="exit"),
            IRInstruction(IROp.HALT),
        ]
    )


# ── Tests ──────────────────────────────────────────────

import unittest


class TestIR(unittest.TestCase):
    def test_ir_to_string(self):
        prog = IRProgram("test", [
            IRInstruction(IROp.CONST, "x", [42]),
            IRInstruction(IROp.HALT),
        ])
        s = prog.to_string()
        self.assertIn("const", s)
        self.assertIn("42", s)
    
    def test_const_to_bytecode(self):
        prog = IRProgram("test", [
            IRInstruction(IROp.CONST, "x", [42]),
            IRInstruction(IROp.HALT),
        ])
        bc = prog.to_bytecode()
        self.assertEqual(bc[0], 0x18)  # MOVI
        self.assertEqual(bc[2], 42)
    
    def test_add_to_bytecode(self):
        prog = IRProgram("test", [
            IRInstruction(IROp.CONST, "a", [10]),
            IRInstruction(IROp.CONST, "b", [20]),
            IRInstruction(IROp.ADD, "c", ["a", "b"]),
            IRInstruction(IROp.HALT),
        ])
        bc = prog.to_bytecode()
        self.assertIn(0x20, bc)  # ADD opcode
    
    def test_factorial_ir(self):
        prog = from_factorial(6)
        self.assertGreater(len(prog.instructions), 5)
        self.assertIn("factorial", prog.name)
    
    def test_fibonacci_ir(self):
        prog = from_fibonacci(10)
        s = prog.to_string()
        self.assertIn("fibonacci", s)
    
    def test_labels_tracked(self):
        prog = IRProgram("test", [
            IRInstruction(IROp.LABEL, label="start"),
            IRInstruction(IROp.HALT),
        ])
        bc = prog.to_bytecode()
        self.assertGreater(len(bc), 0)
    
    def test_mul_to_bytecode(self):
        prog = IRProgram("test", [
            IRInstruction(IROp.CONST, "a", [5]),
            IRInstruction(IROp.CONST, "b", [7]),
            IRInstruction(IROp.MUL, "c", ["a", "b"]),
            IRInstruction(IROp.HALT),
        ])
        bc = prog.to_bytecode()
        self.assertIn(0x22, bc)
    
    def test_neg_to_bytecode(self):
        prog = IRProgram("test", [
            IRInstruction(IROp.CONST, "x", [42]),
            IRInstruction(IROp.NEG, "x", ["x"]),
            IRInstruction(IROp.HALT),
        ])
        bc = prog.to_bytecode()
        self.assertIn(0x0B, bc)
    
    def test_push_pop(self):
        prog = IRProgram("test", [
            IRInstruction(IROp.CONST, "x", [42]),
            IRInstruction(IROp.PUSH, args=["x"]),
            IRInstruction(IROp.POP, args=["y"]),
            IRInstruction(IROp.HALT),
        ])
        bc = prog.to_bytecode()
        self.assertIn(0x0C, bc)
        self.assertIn(0x0D, bc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
