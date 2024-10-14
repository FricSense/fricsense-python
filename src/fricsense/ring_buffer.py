import numpy as np
from numba.experimental import jitclass
from numba.types import uint8, int32
import unittest


class BufferFullException(Exception):
    pass


class BufferDataUnavailableException(Exception):
    pass


spec = [
    ("size", int32),
    ("buffer", uint8[:]),
    ("ri", int32),
    ("wi", int32),
    ("used", int32),
]


@jitclass(spec)
class RingBuffer:
    def __init__(self, size: int32):
        self.size = size
        self.buffer = np.empty(size, uint8)
        self.ri = 0
        self.wi = 0
        self.used = 0

    @property
    def capacity(self) -> int:
        return self.size

    @property
    def free(self) -> int:
        return self.size - self.used

    def add(self, buffer: bytes):
        free: int = self.free
        buffer_len: int = len(buffer)
        if buffer_len > free:
            raise BufferFullException()
        for i in range(buffer_len):
            self.buffer[self.wi % self.size] = buffer[i]
            self.wi += 1
        self.wi %= self.size
        self.used += buffer_len

    def peek(self, length: int) -> bytes:
        used: int = self.used
        if length > used:
            raise BufferDataUnavailableException()

        data = np.empty(length, uint8)
        si: int = self.ri
        ei: int = min(self.ri + length, self.size)
        N: int = ei - si
        data[0:N] = self.buffer[si:ei]
        if N < length:
            data[ei - si :] = self.buffer[0 : length - N]
        return data

    def consume(self, length: int) -> int:
        used: int = self.used
        if length > used:
            length = used
        self.ri += length
        self.ri %= self.size
        self.used -= length
        return length

    def find(self, substr: bytes) -> int:
        used: int = self.used
        ss_len: int = len(substr)
        if (ss_len <= 0) or (ss_len > used):
            return -1
        buffer_si: int = self.ri
        buffer_ei: int = buffer_si + used
        substr_i: int = 0

        for buffer_i in range(buffer_si, buffer_ei):
            if self.buffer[buffer_i % self.size] == substr[substr_i]:
                substr_i += 1
                if substr_i >= ss_len:
                    return buffer_i - len(substr) - self.ri + 1
            else:
                substr_i = 0

        return -1


class RingBufferTestCase(unittest.TestCase):
    def setUp(self):
        self.cb = RingBuffer(10)

    def test_empty(self):
        self.assertEqual(self.cb.capacity, 10)
        self.assertEqual(self.cb.used, 0)
        self.assertEqual(self.cb.free, 10)
        with self.assertRaises(BufferDataUnavailableException):
            self.cb.peek(1)

    def test_fill(self):
        self.cb.add(b"1234567890")
        self.assertEqual(self.cb.used, 10)
        self.assertEqual(self.cb.free, 0)
        np.testing.assert_array_equal(self.cb.peek(10).tobytes(), b"1234567890")
        with self.assertRaises(BufferFullException):
            self.cb.add(b"1")
        self.cb.consume(10)
        self.cb.add(b"1234")
        self.assertEqual(self.cb.used, 4)
        self.assertEqual(self.cb.free, 6)
        np.testing.assert_array_equal(self.cb.peek(4).tobytes(), b"1234")
        self.cb.consume(2)
        self.assertEqual(self.cb.used, 2)
        self.assertEqual(self.cb.free, 8)
        np.testing.assert_array_equal(self.cb.peek(2).tobytes(), b"34")
        self.cb.add(b"5678")
        self.assertEqual(self.cb.used, 6)
        self.assertEqual(self.cb.free, 4)
        np.testing.assert_array_equal(self.cb.peek(6).tobytes(), b"345678")
        self.cb.consume(6)
        self.assertEqual(self.cb.used, 0)
        self.assertEqual(self.cb.free, 10)
        with self.assertRaises(BufferDataUnavailableException):
            self.cb.peek(1)
        self.cb.add(b"1234567890")
        self.assertEqual(self.cb.used, 10)
        self.assertEqual(self.cb.free, 0)
        np.testing.assert_array_equal(self.cb.peek(10).tobytes(), b"1234567890")
        with self.assertRaises(BufferFullException):
            self.cb.add(b"1")

    def test_find_simple(self):
        self.cb.add(b"0123456789")
        np.testing.assert_array_equal(self.cb.buffer.tobytes(), b"0123456789")
        for i in range(10):
            self.assertEqual(self.cb.find(str(i).encode("ascii")), i)
        self.assertEqual(self.cb.find(b"01"), 0)
        self.assertEqual(self.cb.find(b"456"), 4)
        self.assertEqual(self.cb.find(b"6789"), 6)
        self.assertEqual(self.cb.find(b"0123456789"), 0)

        self.assertEqual(self.cb.find(b"01234567890"), -1)
        self.assertEqual(self.cb.find(b"0123456788"), -1)

    def test_find_wrapped(self):
        self.cb.add(b"XXXXXXX")
        self.cb.consume(7)
        self.cb.add(b"0123456789")
        np.testing.assert_array_equal(self.cb.buffer.tobytes(), b"3456789012")
        for i in range(10):
            self.assertEqual(self.cb.find(str(i).encode("ascii")), i)
        self.assertEqual(self.cb.find(b"01"), 0)
        self.assertEqual(self.cb.find(b"456"), 4)
        self.assertEqual(self.cb.find(b"6789"), 6)
        self.assertEqual(self.cb.find(b"0123456789"), 0)

        self.assertEqual(self.cb.find(b"23"), 2)
        self.assertEqual(self.cb.find(b"123"), 1)
        self.assertEqual(self.cb.find(b"0123"), 0)
        self.assertEqual(self.cb.find(b"234"), 2)
        self.assertEqual(self.cb.find(b"2345"), 2)
        self.assertEqual(self.cb.find(b"23458"), -1)

        self.assertEqual(self.cb.find(b"01234567890"), -1)
        self.assertEqual(self.cb.find(b"0123456788"), -1)


# Running the module will run some simple unit tests.
if __name__ == "__main__":
    unittest.main()
