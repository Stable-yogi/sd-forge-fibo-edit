# Read tensor names, types and shapes from a GGUF header. No tensor data is read. Author: Stable Yogi.
import struct

TYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
              12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16"}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("GGUF header ends early")
        return struct.unpack(fmt, data)[0]

    def text(self):
        n = self.take("<Q")
        if n > 1 << 24:
            raise ValueError("GGUF string too long")
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype in _SCALAR:
            return self.take(_SCALAR[vtype])
        if vtype == 8:
            return self.text()
        if vtype == 9:
            itype, count = self.take("<I"), self.take("<Q")
            return [self.value(itype) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_header(path: str) -> dict:
    """Returns {"version", "metadata": {key: value}, "tensors": {name: (torch_shape, type_name)}}."""
    with open(path, "rb") as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = r.take("<I")
        n_tensors, n_kv = r.take("<Q"), r.take("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.text()
            meta[key] = r.value(r.take("<I"))
        tensors = {}
        for _ in range(n_tensors):
            name = r.text()
            dims = [r.take("<Q") for _ in range(r.take("<I"))]
            ttype = r.take("<I")
            r.take("<Q")  # data offset, unused
            tensors[name] = (tuple(reversed(dims)), TYPE_NAMES.get(ttype, str(ttype)))
    return {"version": version, "metadata": meta, "tensors": tensors}
