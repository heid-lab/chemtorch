from typing import Callable


def enforce_base_init(base_cls: type) -> Callable:
    """
    Decorator to enforce that subclasses of a given base class call the base class's __init__ method.
    This is useful for ensuring that the base class's initialization logic is always executed.

    Args:
        base_cls (type): The base class whose __init__ method must be called.

    Returns:
        Callable: A decorator that enforces the base class's __init__ method call.

    Raises:
        RuntimeError: If the subclass's __init__ method does not call the base class's __init__ method.

    Example:
        class Base:
            def __init__(self):
                print("Base init called")
                self._initialized_by_base = True

            def __init_subclass__(cls):
                enforce_base_init(Base)(cls)
                return super().__init_subclass__()

        class SubClass(Base):
            def __init__(self):
                print("SubClass init called")
                super().__init__()  # Must call base class's __init__

        sub = SubClass()  # This will work

    Example:
        >>> class Base:
        ...     def __init__(self):
        ...         print("Base init called")
        ...         self._initialized_by_base = True
        ...
        ...     def __init_subclass__(cls):
        ...         enforce_base_init(Base)(cls)
        ...         return super().__init_subclass__()
        ...
        >>> class GoodSubClass(Base):
        ...     def __init__(self):
        ...         super().__init__()  # Must call base class's __init__
        ...         print("SubClass init called")
        ...
        >>> class BadSubClass(Base):
        ...     def __init__(self):
        ...         print("OtherSubClass init called")
        ...         # super().__init__()  # This will raise an error
        ...
        >>> sub = GoodSubClass()  # This will work
        Base init called
        GoodSubClass init called
        >>> other_sub = BadSubClass()  # This will raise an error
        Traceback (most recent call last):
            ...
        RuntimeError: BadSubClass.__init__() must call super().__init__() from Base
    """

    def decorator(sub_cls):
        orig_init = sub_cls.__init__

        def wrapped_init(self, *args, **kwargs):
            self._initialized_by_base = False  # reset before init
            orig_init(self, *args, **kwargs)
            if not getattr(self, "_initialized_by_base", False):
                raise RuntimeError(
                    f"{sub_cls.__name__}.__init__() must call super().__init__() from {base_cls.__name__}"
                )

        sub_cls.__init__ = wrapped_init
        return sub_cls

    return decorator