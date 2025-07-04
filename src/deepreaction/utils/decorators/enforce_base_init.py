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
        """
        The actual decorator that will be applied to the subclass.
        """
        # Store the original __init__ method of the subclass
        orig_init = sub_cls.__init__

        def wrapped_init(self, *args, **kwargs):
            """
            Wrapper around the subclass's __init__ method to check if the base class's __init__ was called.
            """
            # Store the original value of the _initialized_by_base attribute, in case it was already set
            orig_initialized_by_base = getattr(self, "_initialized_by_base", False)
            try:
                # Call the original __init__ method of the subclass
                orig_init(self, *args, **kwargs)
            finally:
                # Check if the _initialized_by_base attribute is still False after calling the subclass's __init__
                # Also check if it was originally False to handle multiple inheritance correctly
                if not getattr(self, "_initialized_by_base", False) and not orig_initialized_by_base:
                    # If the base class's __init__ was not called, raise an error
                    raise RuntimeError(
                        f"{sub_cls.__name__}.__init__() must call super().__init__() from {base_cls.__name__}"
                    )

        # Replace the subclass's __init__ method with the wrapped version
        sub_cls.__init__ = wrapped_init

        # Modify the base class to set the flag
        orig_base_init = base_cls.__init__

        def wrapped_base_init(self, *args, **kwargs):
            """
            Wrapper around the base class's __init__ method to set the _initialized_by_base attribute to True.
            """
            # Call the original __init__ method of the base class
            orig_base_init(self, *args, **kwargs)
            # Set the _initialized_by_base attribute to True to indicate that the base class's __init__ was called
            self._initialized_by_base = True

        # Replace the base class's __init__ method with the wrapped version
        base_cls.__init__ = wrapped_base_init
        return sub_cls

    return decorator