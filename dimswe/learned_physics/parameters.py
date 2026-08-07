"""Float64 JAX-pytree operations for learned-physics parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp


class Float64TreeError(TypeError):
    """A pytree contains a leaf outside the J4A float64 contract."""


_FLOAT64 = jnp.dtype(jnp.float64)


def _require_x64() -> None:
    if not bool(jax.config.read("jax_enable_x64")):
        raise Float64TreeError(
            "learned-physics operations require jax_enable_x64=True"
        )


def _validated_leaf(name: str, leaf):
    try:
        value = jnp.asarray(leaf)
    except (TypeError, ValueError) as exc:
        raise Float64TreeError(
            f"{name} contains a non-array leaf of type {type(leaf).__name__}"
        ) from exc
    if value.dtype != _FLOAT64:
        raise Float64TreeError(
            f"{name} leaves must have dtype float64, got {value.dtype}"
        )
    return value


def validate_float64_tree(tree, *, name: str = "tree"):
    """Validate and return an owned float64 copy with the same pytree shape."""
    _require_x64()
    return jax.tree_util.tree_map(
        lambda leaf: jnp.array(_validated_leaf(name, leaf), copy=True), tree
    )


def tree_copy(tree):
    """Return an owned float64 copy of an arbitrary JAX pytree."""
    return validate_float64_tree(tree)


def tree_zeros(tree):
    """Return float64 zeros with the same structure and leaf shapes."""
    owned = validate_float64_tree(tree)
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros_like(leaf, dtype=jnp.float64), owned
    )


def _paired_leaves(left, right):
    left_owned = validate_float64_tree(left, name="left")
    right_owned = validate_float64_tree(right, name="right")
    left_leaves, left_definition = jax.tree_util.tree_flatten(left_owned)
    right_leaves, right_definition = jax.tree_util.tree_flatten(right_owned)
    if left_definition != right_definition:
        raise ValueError("pytree structures differ")
    for index, (left_leaf, right_leaf) in enumerate(
        zip(left_leaves, right_leaves)
    ):
        if left_leaf.shape != right_leaf.shape:
            raise ValueError(
                f"pytree leaf {index} shapes differ: "
                f"{left_leaf.shape} != {right_leaf.shape}"
            )
    return left_owned, right_owned, tuple(zip(left_leaves, right_leaves))


def tree_dot(left, right):
    """Return the Euclidean dot product over all corresponding leaves."""
    _, _, pairs = _paired_leaves(left, right)
    if not pairs:
        return jnp.float64(0.0)
    return sum(
        (jnp.vdot(left_leaf, right_leaf) for left_leaf, right_leaf in pairs),
        start=jnp.float64(0.0),
    )


def tree_norm(tree):
    """Return the Euclidean norm over every float64 leaf."""
    return jnp.sqrt(tree_dot(tree, tree))


def tree_axpy(base, scale, increment):
    """Return ``base + scale * increment`` without mutating either input."""
    base_owned, increment_owned, _ = _paired_leaves(base, increment)
    scale_value = _validated_leaf("scale", scale)
    if scale_value.shape != ():
        raise ValueError("scale must be scalar")
    return jax.tree_util.tree_map(
        lambda x, y: jnp.array(x + scale_value * y, copy=True),
        base_owned,
        increment_owned,
    )


def tree_all_finite(tree):
    """Return a scalar JAX boolean indicating whether every leaf is finite."""
    owned = validate_float64_tree(tree)
    leaves = jax.tree_util.tree_leaves(owned)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(
        jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves))
    )


__all__ = (
    "Float64TreeError",
    "tree_all_finite",
    "tree_axpy",
    "tree_copy",
    "tree_dot",
    "tree_norm",
    "tree_zeros",
    "validate_float64_tree",
)
