import pytest
from deepreaction.utils.decorators.enforce_base_init import enforce_base_init

def test_subclass_calls_super_init():
    class Base:
        def __init__(self):
            self._initialized_by_base = True

        def __init_subclass__(cls):
            enforce_base_init(Base)(cls)
            return super().__init_subclass__()

    class GoodSub(Base):
        def __init__(self):
            super().__init__()
            self.value = 42

    # Should not raise
    obj = GoodSub()
    assert obj._initialized_by_base
    assert obj.value == 42

def test_subclass_missing_super_init_raises():
    class Base:
        def __init__(self):
            self._initialized_by_base = True

        def __init_subclass__(cls):
            enforce_base_init(Base)(cls)
            return super().__init_subclass__()

    class BadSub(Base):
        def __init__(self):
            # Missing super().__init__()
            self.value = 99

    with pytest.raises(RuntimeError) as excinfo:
        BadSub()
    assert "must call super().__init__()" in str(excinfo.value)

def test_grandchild_must_call_super_init():
    class Base:
        def __init__(self):
            self._initialized_by_base = True

        def __init_subclass__(cls):
            enforce_base_init(Base)(cls)
            return super().__init_subclass__()

    class Mid(Base):
        def __init__(self):
            super().__init__()

    class BadGrandchild(Mid):
        def __init__(self):
            # Missing super().__init__()
            pass

    with pytest.raises(RuntimeError):
        BadGrandchild()

    class GoodGrandchild(Mid):
        def __init__(self):
            super().__init__()

    # Should not raise
    GoodGrandchild()

def test_subclass_computes_arg_for_super_init():
    class Base:
        def __init__(self, value):
            self.value = value

        def __init_subclass__(cls):
            enforce_base_init(Base)(cls)
            return super().__init_subclass__()

    class SubWithComputedArg(Base):
        def __init__(self, x):
            computed_value = x * 2
            super().__init__(computed_value)
            self.x = x

    # Should not raise
    obj = SubWithComputedArg(10)
    assert obj.value == 20
    assert obj.x == 10