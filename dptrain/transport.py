"""Real socket transport for pipeline stages.

Everything else in dptrain simulates the network by counting bytes. This
module actually sends them: separate OS processes, real TCP, real
serialization. The same code runs across two machines by changing --host;
nothing else differs.

Binds 127.0.0.1 by default. Listening on a routable interface requires
passing --host explicitly, and there is no authentication -- do not expose
this to an untrusted network.
"""

import socket
import struct
import time

import torch

HEADER = struct.Struct("!BBHQ")   # dtype_code, ndim, flags, payload_len
DTYPES = {0: torch.float32, 1: torch.bfloat16, 2: torch.int8}
CODES = {v: k for k, v in DTYPES.items()}


class Wire:
    """A socket plus byte/time counters, optionally rate-limited.

    The throttle makes a loopback run predictive of a real link: loopback is
    ~50 Gb/s, so without it the measurement says nothing about a 110 Mb/s
    uplink.
    """

    def __init__(self, sock: socket.socket, mbps: float = 0.0, rtt_ms: float = 0.0):
        self.sock = sock
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.mbps, self.rtt_ms = mbps, rtt_ms
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.send_seconds = 0.0
        self.recv_seconds = 0.0

    def _throttle(self, nbytes: int):
        if self.mbps <= 0:
            return
        time.sleep(nbytes * 8 / (self.mbps * 1e6) + self.rtt_ms / 2000.0)

    def send(self, t: torch.Tensor):
        t = t.contiguous()
        payload = t.view(torch.int8).numpy().tobytes() if t.dtype == torch.bfloat16 \
            else t.numpy().tobytes()
        head = HEADER.pack(CODES[t.dtype], t.dim(), 0, len(payload))
        shape = struct.pack(f"!{t.dim()}I", *t.shape)
        t0 = time.perf_counter()
        # Throttle BEFORE sendall. Sleeping afterwards leaves the bytes already
        # delivered, so the peer runs on while the sender waits -- that models
        # sender occupancy, not a slow link, and understates step latency.
        self._throttle(len(payload))
        self.sock.sendall(head + shape + payload)
        self.send_seconds += time.perf_counter() - t0
        self.bytes_sent += len(head) + len(shape) + len(payload)

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("peer closed mid-tensor")
            buf += chunk
        return bytes(buf)

    def recv(self) -> torch.Tensor:
        t0 = time.perf_counter()
        code, ndim, _, plen = HEADER.unpack(self._recv_exact(HEADER.size))
        shape = struct.unpack(f"!{ndim}I", self._recv_exact(4 * ndim))
        payload = self._recv_exact(plen)
        self.recv_seconds += time.perf_counter() - t0
        self.bytes_recv += HEADER.size + 4 * ndim + plen
        dt = DTYPES[code]
        if dt == torch.bfloat16:
            flat = torch.frombuffer(bytearray(payload), dtype=torch.int8)
            return flat.view(torch.bfloat16).view(shape).clone()
        flat = torch.frombuffer(bytearray(payload), dtype=dt)
        return flat.view(shape).clone()

    def stats(self) -> dict:
        return {"bytes_sent": self.bytes_sent, "bytes_recv": self.bytes_recv,
                "send_s": round(self.send_seconds, 4),
                "recv_s": round(self.recv_seconds, 4)}


def serve(host: str, port: int, timeout: float = 60.0) -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(timeout)
    conn, _ = srv.accept()
    srv.close()
    return conn


def connect(host: str, port: int, timeout: float = 60.0) -> socket.socket:
    deadline = time.time() + timeout
    while True:
        try:
            s = socket.create_connection((host, port), timeout=10)
            return s
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.1)
