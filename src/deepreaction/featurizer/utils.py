from typing import Any, Callable, List, Sequence, Tuple, Union

def one_hot_with_unknown(
    value: Any,
    options: Sequence[Any]
) -> List[int]:
    x = [0] * (len(options) + 1)
    option_dict = {j: i for i, j in enumerate(options)}
    idx = option_dict.get(value, len(option_dict))
    x[idx] = 1
    return x

def featurize(
        item: Any, 
        features: List[Union[Callable, Tuple[Callable, Sequence[Any]]]]
    ) -> List[float] | List[int]:
    """
    Applies a sequence of feature extractors to an item and concatenates their outputs into a single feature vector.

    Each element in `features` can be:
      - A callable: The callable is applied to `item`. The return value can be:
          * a numeric value (float or int): appended to the feature vector,
          * a bool: converted to int and appended,
          * a list of numerics/bools: extended into the feature vector.
      - A tuple (func, options): `func(item)` is interpreted as a categorical value and one-hot encoded
        using `options` (with an extra slot for unknowns), then extended into the feature vector.

    If `item` is None, returns a zero vector of the appropriate length.

    Args:
        item: The object to featurize (e.g., an Atom or Bond).
        features: A list of callables or (callable, options) tuples specifying how to extract each feature.

    Returns:
        List[float]: The concatenated feature vector.
    
    Raises:
        TypeError: If `features` is not a list or if any feature is not callable or a (callable, options) tuple.
        TypeError: If the return type of a feature function is not supported (not float, int, bool, or list of these).
        ValueError: If any feature tuple does not have exactly two elements.
        RuntimeError: If an error occurs during feature extraction.
    """
    # Validate features is a list
    if not isinstance(features, list):
        raise TypeError(f"`features` must be a list, got {type(features)}")

    # Validate each feature
    for idx, feature in enumerate(features):
        if isinstance(feature, tuple):
            if len(feature) != 2:
                raise ValueError(f"Feature tuple at index {idx} must have length 2 (func, options), got {len(feature)}")
            func, options = feature
            if not callable(func):
                raise TypeError(f"First element of tuple at index {idx} must be callable, got {type(func)}")
            if not isinstance(options, (list, tuple, Sequence)):
                raise TypeError(f"Second element of tuple at index {idx} must be a sequence, got {type(options)}")
        elif not callable(feature):
            raise TypeError(f"Feature at index {idx} must be callable or a (callable, options) tuple, got {type(feature)}")

    if item is None:
        # return a list of zeros with the size of the feature vector
        dim = 0
        for feature in features:
            if isinstance(feature, tuple):
                _, options = feature
                dim += (len(options) + 1) if options else 1
            else:
                dim += 1
        return [0] * dim

    feature_vector = []
    for idx, feature in enumerate(features):
        try:
            if isinstance(feature, tuple):
                func, options = feature
                val = func(item)
                feature_vector.extend(one_hot_with_unknown(val, options))
            else:
                res = feature(item)
                if isinstance(res, list):
                    if not all(isinstance(x, (float, int, bool)) for x in res):
                        raise TypeError(
                            f"Feature function at index {idx} returned a list with unsupported types. "
                            "Supported: float, int, bool."
                        )
                    feature_vector.extend(res)
                elif isinstance(res, (float, int)):
                    feature_vector.append(res)
                elif isinstance(res, bool):
                    feature_vector.append(int(res))
                else:
                    raise TypeError(
                        f"Feature function at index {idx} returned unsupported type {type(res)}. "
                        "Supported: float, int, bool, or list of these."
                    )
        except Exception as e:
            feature_name = getattr(feature, '__name__', str(feature))
            raise RuntimeError(
                f"Error extracting feature at index {idx} ({feature_name}) for item {item}: {e}"
            ) from e
    return feature_vector
