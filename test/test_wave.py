import importlib.util
import numpy as np
import pytest

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
if cupy_spec is not None:
    import cupy as cp
    backend_libs = [np, cp]
else:
    cp = None
    backend_libs = [np]


class DummyFactor:
    def __init__(self):
        self.received = None

    def receive_message(self, wave, msg, block):
        self.received = (wave, msg)

    def get_input_precision_mode(self, wave):
        return "scalar"

    def get_output_precision_mode(self):
        return "scalar"


@pytest.mark.parametrize("xp", backend_libs)
def test_wave_sample_and_clear(xp):
    backend.set_backend(xp)
    w = Wave(event_shape=(4, 4), dtype=xp.complex64)
    sample = xp.ones((1, 4, 4), dtype=xp.complex64)
    w.set_sample(sample)
    assert xp.allclose(w.get_sample(), sample)
    w.clear_sample()
    assert w.get_sample() is None


@pytest.mark.parametrize("xp", backend_libs)
def test_receive_message_and_compute_belief_scalar(xp):
    backend.set_backend(xp)
    w = Wave(event_shape=(2, 2), batch_size=1, dtype=xp.complex64)
    w._set_precision_mode("scalar")
    parent = DummyFactor()
    child1 = DummyFactor()
    child2 = DummyFactor()

    w.set_parent(parent)
    w.add_child(child1)
    w.add_child(child2)

    msg1 = UncertainArray(xp.full((1, 2, 2), 1.0), precision=1.0)
    msg2 = UncertainArray(xp.full((1, 2, 2), 3.0), precision=1.0)
    parent_msg = UncertainArray(xp.full((1, 2, 2), 2.0), precision=2.0)

    w.receive_message(child1, msg1)
    w.receive_message(child2, msg2)
    w.receive_message(parent, parent_msg)

    belief = w.compute_belief()
    assert isinstance(belief, UncertainArray)
    assert belief.event_shape == (2, 2)
    assert xp.allclose(belief.data.shape, (1, 2, 2))


@pytest.mark.parametrize("xp", backend_libs)
def test_wave_to_backend_converts_all_messages(xp):
    backend.set_backend(xp)
    w = Wave(event_shape=(2, 2), batch_size=2, dtype=xp.complex64)
    w._set_precision_mode("array")
    child = DummyFactor()
    parent = DummyFactor()
    w.add_child(child)
    w.set_parent(parent)

    msg = UncertainArray(xp.full((2, 2, 2), 1.0), precision=xp.ones((2, 2, 2)))
    parent_msg = UncertainArray(xp.full((2, 2, 2), 1.0), precision=xp.ones((2, 2, 2)))
    w.receive_message(child, msg)
    w.receive_message(parent, parent_msg)
    belief = w.compute_belief()

    if cp is not None and xp.__name__ == "numpy":
        backend.set_backend(cp)
        w.to_backend()
        assert isinstance(w.belief.data, cp.ndarray)
        for m in w.child_messages.values():
            assert isinstance(m.data, cp.ndarray)
    elif cp is None and xp.__name__ == "numpy":
        pytest.skip("CuPy not available, skipping transfer-to-backend test")
    else:
        backend.set_backend(np)
        w.to_backend()
        assert isinstance(w.belief.data, np.ndarray)
        for m in w.child_messages.values():
            assert isinstance(m.data, np.ndarray)


@pytest.mark.parametrize("xp", backend_libs)
def test_wave_to_backend_converts_all_messages_with_parent(xp):
    backend.set_backend(xp)
    w = Wave(event_shape=(2, 2), batch_size=2, dtype=xp.complex64)
    w._set_precision_mode("array")
    parent = DummyFactor()
    child = DummyFactor()
    w.set_parent(parent)
    w.add_child(child)

    msg = UncertainArray(xp.full((2, 2, 2), 1.0), precision=xp.ones((2, 2, 2)))
    w.receive_message(child, msg)
    w.receive_message(parent, msg)

    w.compute_belief()

    if cp is not None and xp.__name__ == "numpy":
        backend.set_backend(cp)
        w.to_backend()
        assert isinstance(w.belief.data, cp.ndarray)
        for m in w.child_messages.values():
            assert isinstance(m.data, cp.ndarray)
    elif cp is None and xp.__name__ == "numpy":
        pytest.skip("CuPy not available, skipping transfer-to-backend test")
    else:
        backend.set_backend(np)
        w.to_backend()
        assert isinstance(w.belief.data, np.ndarray)
        for m in w.child_messages.values():
            assert isinstance(m.data, np.ndarray)

def test_set_label_and_repr():
    w = Wave((2,2))
    w.set_label("test")
    assert "test" in repr(w)


def test_set_precision_mode_conflict():
    w = Wave((2,2))
    w._set_precision_mode("scalar")
    with pytest.raises(ValueError):
        w._set_precision_mode("array")


def test_set_parent_conflict():
    w = Wave((2,2))
    f1, f2 = DummyFactor(), DummyFactor()
    w.set_parent(f1)
    with pytest.raises(ValueError):
        w.set_parent(f2)


def test_add_child_conflict():
    w = Wave((2,2))
    f = DummyFactor()
    w.add_child(f)
    with pytest.raises(ValueError):
        w.add_child(f)


def test_receive_message_dtype_mismatch_and_unregistered():
    w = Wave((2,2), dtype=np.float32)
    f = DummyFactor()
    w.set_parent(f)

    # mismatch: complex UA into float wave → allowed via .real
    msg = UncertainArray(np.ones((1,2,2),dtype=np.complex64), precision=1.0)
    w.receive_message(f, msg)
    assert w.parent_message is not None

    # mismatch: float UA into complex wave (ok: promote)
    w2 = Wave((2,2), dtype=np.complex64)
    f2 = DummyFactor()
    w2.set_parent(f2)
    msg2 = UncertainArray(np.ones((1,2,2),dtype=np.float32), precision=1.0)
    w2.receive_message(f2, msg2)
    assert w2.parent_message is not None

    # truly incompatible dtype
    w3 = Wave((2,2), dtype=np.int32)
    f3 = DummyFactor()
    w3.set_parent(f3)
    bad_msg = UncertainArray(np.ones((1,2,2),dtype=np.complex64), precision=1.0)
    with pytest.raises(TypeError):
        w3.receive_message(f3, bad_msg)

    # unregistered factor
    w4 = Wave((2,2))
    f4 = DummyFactor()
    with pytest.raises(ValueError):
        w4.receive_message(f4, msg)


def test_set_belief_shape_and_dtype_mismatch():
    w = Wave((2,2), batch_size=1, dtype=np.complex64)
    ua = UncertainArray(np.ones((2,2,2)), precision=1.0)  # wrong batch_size
    with pytest.raises(ValueError):
        w.set_belief(ua)

    ua2 = UncertainArray(np.ones((1,3,3)), precision=1.0)  # wrong shape
    with pytest.raises(ValueError):
        w.set_belief(ua2)

    ua3 = UncertainArray(np.ones((1,2,2),dtype=np.float32), precision=1.0)  # wrong dtype
    with pytest.raises(ValueError):
        w.set_belief(ua3)


def test_forward_branches_and_errors():
    w = Wave((2,2))
    f = DummyFactor()
    w.set_parent(f)
    # no parent message yet
    with pytest.raises(RuntimeError):
        w.forward()

    # case 1 child
    child = DummyFactor()
    w.add_child(child)
    msg = UncertainArray(np.ones((1,2,2)), precision=1.0)
    w.receive_message(f, msg)
    w.child_messages[child] = UncertainArray(np.ones((1,2,2)), precision=1.0)
    w.forward()
    assert child.received is not None

    # case >1 child
    w2 = Wave((2,2))
    f2 = DummyFactor()
    w2.set_parent(f2)
    c1, c2 = DummyFactor(), DummyFactor()
    w2.add_child(c1)
    w2.add_child(c2)
    mparent = UncertainArray(np.ones((1,2,2)), precision=1.0)
    w2.receive_message(f2, mparent)
    w2.child_messages[c1] = UncertainArray.zeros((2,2), batch_size=1, dtype=w2.dtype, precision=1.0)
    w2.child_messages[c2] = UncertainArray.zeros((2,2), batch_size=1, dtype=w2.dtype, precision=1.0)
    w2.forward()
    assert c1.received is not None and c2.received is not None


def test_combine_child_messages_no_children():
    w = Wave((2,2))
    with pytest.raises(RuntimeError):
        w.combine_child_messages()

def test_compute_belief_without_parent():
    w = Wave((2,2))
    # add a dummy child so combine_child_messages works
    class F: pass
    f = F()
    w.children.append(f)
    w.child_messages[f] = UncertainArray.zeros((2,2), batch_size=1)
    with pytest.raises(RuntimeError):
        w.compute_belief()

def test_generate_sample_cases():
    w = Wave((2,2))
    # case: already has sample
    w._sample = np.ones((1,2,2))
    w._generate_sample(rng=None)  # should just return, not error

    # case: no parent → nothing happens
    w2 = Wave((2,2))
    w2._generate_sample(rng=None)  # no exception, just silent

def test_set_sample_shape_mismatch():
    w = Wave((2,2))
    bad = np.ones((3,3))  # not broadcastable to (1,2,2)
    with pytest.raises(ValueError):
        w.set_sample(bad)

def test_clear_sample_sets_none():
    w = Wave((2,2))
    w._sample = np.ones((1,2,2))
    w.clear_sample()
    assert w._sample is None

def test_getitem_invalid_type():
    w = Wave((2,2))
    with pytest.raises(TypeError):
        _ = w[3.14, :]   # float is invalid index

def test_rmul_scalar_and_array():
    w = Wave((2,2))
    # both should delegate to __mul__, not error
    out1 = (2 * w)
    out2 = (np.ones((2,2)) * w)
    assert isinstance(out1, Wave)
    assert isinstance(out2, Wave)

def test_repr_batchsize_gt1():
    w = Wave((2,2), batch_size=5)
    s = repr(w)
    assert "batch_size=5" in s