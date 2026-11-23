"""Boundary handling for elliptic problems on DIMSWE's box-shaped meshes (TODO-elliptic §4.2).

Logical face names (independent of the Firedrake numbering):

| direction | side 0 (low coordinate) | side 1 (high coordinate) |
|-----------|-------------------------|--------------------------|
| x         | left                    | right                    |
| y         | bottom                  | top                      |
| z         | back                    | front                    |

Note that in 3D the logical 'bottom'/'top' are the y faces; the z faces are 'back'/'front' even
though, on extruded meshes, Firedrake calls them "bottom"/"top" (ds_b/ds_t).

Firedrake subdomain ids, for the meshes built by dimswe.meshes.build_mesh:
- 1D/2D ("consecutive" numbering): ids are numbered 1, 2, ... over the *bounded*
  (non-periodic) directions in x, y order, low side first. This matches IntervalMesh
  (1: x0, 2: x1), RectangleMesh (1-4: x0, x1, y0, y1) and PeriodicRectangleMesh(direction='x')
  (1: y=0, 2: y=Ly) / direction='y' (1: x=0, 2: x=Lx), for quadrilaterals and triangles.
- 3D non-extruded ("fixed" numbering: BoxMesh for tets and genuine hexahedra, and
  PeriodicBoxMesh(directions=..., hexahedral=True) for partially periodic genuine hexahedra): ids 1-6 = x0, x1, y0, y1, z0, z1 always; faces of periodic directions
  are interior and simply absent.
- 3D extruded (Q-family hexahedra): the base mesh ids (consecutive rule over x, y) are used with
  ds_v, and the z faces are ds_b/ds_t ("bottom"/"top" for DirichletBC).
- fully periodic meshes: no boundary.
These rules (build_face_registry) are only the expected layout: Firedrake has changed the
numbering of partially periodic meshes between versions, so get_boundary_registry detects the
actual marker of every face geometrically (detect_face_ids). verify_boundary_registry checks the numbering
geometrically against the installed Firedrake.

Each face carries exactly one BC type, 'essential' or 'natural' (TODO-elliptic §1.1, §1.3).
"""
from dataclasses import dataclass
from typing import Tuple, Union

from firedrake import Constant, DirichletBC, Function, SpatialCoordinate, assemble, ds, ds_b, ds_t, ds_v, sqrt

from ufl import zero

from .differential_forms import boundary_pairing
from .meshes import DIRECTIONS, get_mesh_type_dimension, is_box_mesh_type
from .ufl_helpers import ufl_attribute

# Logical face name -> (direction, side). Ordered by direction, low side first.
FACE_LOCATIONS = {
    'left': ('x', 0), 'right': ('x', 1),
    'bottom': ('y', 0), 'top': ('y', 1),
    'back': ('z', 0), 'front': ('z', 1),
}

# Logical faces of non-box ('radial') meshes: disk/ball 'outer', annulus 'inner' and 'outer'.
RADIAL_FACES = ('inner', 'outer')

# Allowed boundary condition types.
BC_TYPES = ('essential', 'natural')

# Firedrake facet measures by name (see measure_for).
MEASURES = {'ds': ds, 'ds_v': ds_v, 'ds_t': ds_t, 'ds_b': ds_b}


@dataclass(frozen=True)
class BoundaryFace:
    """One logical face of a box-shaped mesh and how Firedrake addresses it.

    Attributes:
        name: logical name ('left', 'right', 'bottom', 'top', 'back', 'front').
        direction: coordinate direction normal to the face ('x', 'y' or 'z').
        side: 0 for the low-coordinate face, 1 for the high-coordinate face.
        subdomain_ids: Firedrake boundary ids for the facet measure (empty for ds_t/ds_b).
        measure: name of the facet measure, one of 'ds', 'ds_v', 'ds_t', 'ds_b'.
        dirichlet_subdomain: the sub_domain argument for DirichletBC (an int id, a tuple of ids
            when a face carries several markers, or "top"/"bottom" on extruded meshes).
    """
    name: str
    direction: str
    side: int
    subdomain_ids: Tuple[int, ...]
    measure: str
    dirichlet_subdomain: Union[int, str, Tuple[int, ...]]


def build_face_registry(dim, periodic_directions, extruded=False):
    """Pure-Python construction of the face registry (see the module docstring for the rules).

    Args:
        dim: spatial dimension (1, 2 or 3).
        periodic_directions: iterable of periodic directions ('x', 'y', 'z'); their faces are
            omitted.
        extruded: True for ExtrudedMesh boxes (3D only), whose z faces use ds_b/ds_t.

    1D/2D meshes and the base of extruded meshes use consecutive ids over the bounded directions;
    non-extruded 3D meshes (BoxMesh/PeriodicBoxMesh) use the fixed ids 1-6.

    Returns:
        dict logical name -> BoundaryFace, ordered by direction and side.
    """
    if dim not in (1, 2, 3):
        raise ValueError("dimension " + str(dim) + " is not supported (1, 2 or 3)")
    if extruded and dim != 3:
        raise ValueError("extruded meshes are only used for 3D boxes")
    periodic = set(periodic_directions)
    fixed_ids = dim == 3 and not extruded
    registry = {}
    next_id = 1
    for axis, direction in enumerate(DIRECTIONS[:dim]):
        if direction in periodic:
            continue
        for name, (face_dir, side) in FACE_LOCATIONS.items():
            if face_dir != direction:
                continue
            if fixed_ids:
                # PeriodicBoxMesh keeps the BoxMesh ids for the non-periodic faces.
                next_id = 2 * axis + side + 1
            if extruded and direction == 'z':
                # Extruded z faces: ds_b (bottom layer) and ds_t (top layer).
                token = 'bottom' if side == 0 else 'top'
                registry[name] = BoundaryFace(name, direction, side, (), 'ds_b' if side == 0 else 'ds_t', token)
            else:
                measure = 'ds_v' if extruded else 'ds'
                registry[name] = BoundaryFace(name, direction, side, (next_id,), measure, next_id)
                next_id += 1
    return registry


def is_extruded(mesh):
    """True if mesh is an extruded mesh (cell dimension is a tuple such as (2, 1))."""
    return isinstance(ufl_attribute(mesh, 'cell_dimension'), tuple)


def get_boundary_registry(mesh, mesh_type, periodic_directions=None):
    """Return the face registry (dict name -> BoundaryFace) for a mesh built by DIMSWE.

    Args:
        mesh: the Firedrake mesh (used to detect extrusion).
        mesh_type: parameters['mesh']['type'].
        periodic_directions: periodic directions; defaults to mesh.periodic_directions as
            recorded by dimswe.meshes.annotate_mesh.

    'file' meshes have no logical faces; per-face BCs on them are deferred (TODO-elliptic §8).
    """
    dim = get_mesh_type_dimension(mesh_type)
    if dim is None:
        raise NotImplementedError("boundary registry for 'file' meshes is not implemented")
    if not is_box_mesh_type(mesh_type):
        return radial_face_registry(mesh, mesh_type)
    if periodic_directions is None:
        periodic_directions = getattr(mesh, 'periodic_directions', ())
    # The detection assembles a few small forms per marker; cache the result on the mesh, since
    # many models (and tests) are built on the same mesh.
    key = (mesh_type, tuple(periodic_directions))
    cache = getattr(mesh, '_dimswe_face_registries', None)
    if cache is None:
        cache = {}
        mesh._dimswe_face_registries = cache
    if key not in cache:
        registry = build_face_registry(dim, periodic_directions, extruded=is_extruded(mesh))
        bounds = getattr(mesh, 'domain_bounds', None) or _coordinate_bounds(mesh, dim)
        cache[key] = detect_face_ids(mesh, registry, bounds)
    return dict(cache[key])  # BoundaryFace entries are immutable; the dict is a fresh copy


def radial_face_registry(mesh, mesh_type):
    """Faces of a disk/annulus/ball (TODO-elliptic §4.14), detected by radius about the origin.

    Every non-empty boundary marker belongs to 'outer' (disk, ball), or, for the annulus, to
    'inner' or 'outer' depending on whether its mean radius is below or above the mean of the two
    radii (robust for polygonal approximations of the circles). Faces carry direction None and are
    integrated with ds over their markers. Cached on the mesh like the box registry.
    """
    from .meshes import get_mesh_spec
    spec = get_mesh_spec(mesh_type)
    cache = getattr(mesh, '_dimswe_face_registries', None)
    if cache is None:
        cache = {}
        mesh._dimswe_face_registries = cache
    key = (mesh_type, 'radial')
    if key in cache:
        return dict(cache[key])
    markers = _boundary_markers(mesh) or []
    X = SpatialCoordinate(mesh)
    radius = sqrt(sum(X[i] ** 2 for i in range(spec.dim)))
    means = {}
    for marker in markers:
        if _marker_facet_count(mesh, marker) == 0:
            continue
        dm = ds(marker, domain=mesh)
        area = assemble(Constant(1.0) * dm)
        if area > 0.0:
            means[marker] = assemble(radius * dm) / area
    if not means:
        raise ValueError("mesh type " + mesh_type + " has no boundary markers")
    groups = {name: [] for name in spec.faces}
    if 'inner' in spec.faces:
        split = 0.5 * (min(means.values()) + max(means.values()))
        for marker, mean in means.items():
            groups['inner' if mean < split else 'outer'].append(marker)
    else:
        groups['outer'] = list(means)
    registry = {}
    for side, name in enumerate(spec.faces):
        ids = tuple(sorted(groups[name]))
        if not ids:
            raise ValueError("no boundary marker of the " + mesh_type + " lies on face '" + name + "'")
        registry[name] = BoundaryFace(name, None, side, ids, 'ds', ids[0] if len(ids) == 1 else ids)
    cache[key] = registry
    return dict(registry)


def _coordinate_bounds(mesh, dim):
    """((x0, x1), ...) from the mesh coordinates (used when mesh.domain_bounds is not recorded)."""
    from mpi4py import MPI
    coords = mesh.coordinates.dat.data_ro.reshape(-1, dim)
    # Reduce each axis as a scalar (lowercase allreduce of lists would compare them lexicographically).
    return tuple((mesh.comm.allreduce(float(coords[:, i].min()), op=MPI.MIN),
                  mesh.comm.allreduce(float(coords[:, i].max()), op=MPI.MAX)) for i in range(dim))


def _boundary_markers(mesh):
    """All exterior facet markers of the mesh (side facets on extruded meshes), or None."""
    try:
        markers = mesh.exterior_facets.unique_markers
    except AttributeError:
        return None
    return sorted(int(m) for m in markers)


def _marker_facet_count(mesh, marker):
    """Global number of exterior facets carrying marker, or None if it cannot be determined.

    Used to skip empty labels (the faces of periodic directions in newer Firedrake) before any
    kernel is compiled for them: assembling over an empty facet set makes loopy emit an
    'overflow encountered in scalar negative' RuntimeWarning.
    """
    import logging
    logger = logging.getLogger('firedrake')
    level = logger.level
    # subset() itself warns "Subdomain ... is empty" for the labels we are trying to skip.
    logger.setLevel(logging.ERROR)
    try:
        subset = mesh.exterior_facets.subset(marker)
    except Exception:  # API differs between Firedrake versions: fall back to the area check
        return None
    finally:
        logger.setLevel(level)
    return mesh.comm.allreduce(int(subset.size))


def detect_face_ids(mesh, registry, bounds, rel_tol=1e-8):
    """Assign the mesh's boundary markers to the logical faces by geometry.

    Firedrake's numbering of the remaining faces of partially periodic meshes changed between
    versions (renumbered 1, 2, ... in some, RectangleMesh/BoxMesh ids kept in others), so the ids
    of build_face_registry are only the expected layout. Here every marker with non-zero area is
    assigned to the face whose normal coordinate is constant (equal to its bound) on it; extruded
    z faces (ds_b/ds_t) need no ids. Raises ValueError if a bounded face gets no marker.
    """
    markers = _boundary_markers(mesh)
    if markers is None or not any(face.measure in ('ds', 'ds_v') for face in registry.values()):
        return registry  # nothing to detect (e.g. fully periodic, or only extruded z faces)
    extruded = is_extruded(mesh)
    measure = MEASURES['ds_v' if extruded else 'ds']
    X = SpatialCoordinate(mesh)
    scale = max([1.0] + [hi - lo for lo, hi in bounds])
    found = {name: [] for name, face in registry.items() if face.measure in ('ds', 'ds_v')}
    for marker in markers:
        if _marker_facet_count(mesh, marker) == 0:
            continue  # empty label: no kernel needed
        dm = measure(marker, domain=mesh)
        area = assemble(Constant(1.0) * dm)
        if area <= 1e-14 * scale ** 2:
            continue  # empty label (e.g. the faces of a periodic direction)
        for name in found:
            face = registry[name]
            axis = DIRECTIONS.index(face.direction)
            spread = assemble((X[axis] - bounds[axis][face.side]) ** 2 * dm) / area
            if spread ** 0.5 <= rel_tol * scale:
                found[name].append(marker)
                break
    detected = dict(registry)
    for name, ids in found.items():
        if not ids:
            raise ValueError("no boundary marker of the mesh lies on face '" + name + "' (markers: " + str(markers) + ")")
        face = registry[name]
        ids = tuple(sorted(ids))
        detected[name] = BoundaryFace(face.name, face.direction, face.side, ids, face.measure,
                                      ids[0] if len(ids) == 1 else ids)
    return detected


def parse_boundary_conditions(cfg, registry):
    """Parse the per-face BC types from the 'elliptic: boundary_conditions' cfg section.

    Args:
        cfg: dict face name -> 'essential' | 'natural' (case-insensitive), or None when every
            direction is periodic.
        registry: the face registry of the mesh (get_boundary_registry).

    Returns:
        dict face name -> 'essential' | 'natural', ordered like the registry.

    Raises ValueError for unknown face names, BCs on faces of periodic directions, invalid BC
    types, and bounded faces without a BC.
    """
    cfg = {} if cfg is None else cfg
    if not isinstance(cfg, dict):
        raise ValueError("boundary_conditions must map face names to BC types, got " + repr(cfg))
    parsed = {}
    for key, value in cfg.items():
        name = str(key).strip().lower()
        if name not in FACE_LOCATIONS and name not in RADIAL_FACES:
            raise ValueError("unknown boundary face '" + str(key) + "' (valid: " + str(list(FACE_LOCATIONS) + list(RADIAL_FACES)) + ")")
        if name not in registry:
            raise ValueError("boundary face '" + name + "' does not exist on this mesh (it is periodic or beyond the mesh dimension)")
        if name in parsed:
            raise ValueError("boundary face '" + name + "' is given more than once")
        bc_type = str(value).strip().lower()
        if bc_type not in BC_TYPES:
            raise ValueError("boundary condition '" + str(value) + "' on face '" + name + "' must be one of " + str(list(BC_TYPES)))
        parsed[name] = bc_type
    missing = [name for name in registry if name not in parsed]
    if missing:
        raise ValueError("missing boundary conditions for faces " + str(missing))
    return {name: parsed[name] for name in registry}


def faces_of_type(registry, face_bcs, bc_type):
    """Return the BoundaryFaces (in registry order) whose BC type is bc_type."""
    return [face for name, face in registry.items() if face_bcs[name] == bc_type]


def face_bcs_by_direction(registry, face_bcs):
    """Group the face BC types by direction: {direction: (type on side 0, type on side 1)}.

    Periodic directions are absent. This is the form the harmonic-form rule (TODO-elliptic §1.4)
    needs.
    """
    grouped = {}
    for name, face in registry.items():
        if face.direction is None:
            raise ValueError("face_bcs_by_direction only applies to box meshes (face '" + name + "' is not axis-aligned)")
        grouped.setdefault(face.direction, [None, None])[face.side] = face_bcs[name]
    return {direction: tuple(types) for direction, types in grouped.items()}


def measure_for(face, domain=None, degree=None):
    """Facet measure restricted to face, valid on extruded and non-extruded meshes.

    Args:
        face: a BoundaryFace.
        domain: optional mesh (needed only if the integrand has no domain of its own).
        degree: optional quadrature degree.
    """
    measure = MEASURES[face.measure]
    metadata = {} if degree is None else {'quadrature_degree': degree}
    if face.subdomain_ids:
        return measure(face.subdomain_ids, domain=domain, metadata=metadata)
    return measure(domain=domain, metadata=metadata)


def _face_data(data, face):
    """Boundary data for face: data may be None, a single expression, or dict name -> expression.

    Returns None for homogeneous data (no entry, or None).
    """
    if isinstance(data, dict):
        return data.get(face.name, None)
    return data


def _block_space(W, index):
    """Return block index of W (W itself for a non-mixed space, which only has block 0)."""
    if len(W) > 1:
        return W.sub(index)
    if index != 0:
        raise ValueError("space has a single block, sub index " + str(index) + " is invalid")
    return W


def _zero_value():
    """Homogeneous essential data.

    DirichletBC special-cases a (scalar) UFL Zero and assigns it directly to the boundary dofs,
    for scalar and vector spaces alike. Any other expression, including a zero Constant, would be
    interpolated, which the Q-family Hcurl/Hdiv elements (RTCE/RTCF/NCE/NCF, enriched elements
    without a dual basis) do not support, so Firedrake would fall back to a global L2 projection.
    """
    return zero()


def build_essential_bcs(W, sub_index_sigma, sub_index_u, faces, g_sigma=None, g_u=None):
    """Build the DirichletBCs enforcing tr sigma = g_sigma and tr u = g_u on the essential faces.

    The trace that is constrained is the one natural to each space (value for CG, tangential
    trace for Hcurl, normal trace for Hdiv); full fields may be given as data. The data are first
    put into the space (_data_in_space): interpolated when the element supports it (CG, and the
    simplicial families), otherwise (RTCE/RTCF/NCE/NCF, trimmed serendipity, BDMCE/BDMCF) by a
    global L2 projection whose boundary dofs are then used. That is exact for data in the
    discrete space but in general not a face-wise trace projection.

    Args:
        W: the (mixed) function space of the problem.
        sub_index_sigma: block index of sigma in W, or None when sigma is absent (k = 0).
        sub_index_u: block index of u in W, or None when u has no trace (k = n, DG).
        faces: iterable of essential BoundaryFaces (see faces_of_type).
        g_sigma, g_u: boundary data; None (homogeneous), one UFL expression/Function for every
            face, or dict face name -> expression (missing faces are homogeneous).

    Returns:
        list of DirichletBC (sigma BCs first, then u BCs).
    """
    bcs = []
    for index, data in ((sub_index_sigma, g_sigma), (sub_index_u, g_u)):
        if index is None:
            continue
        V = _block_space(W, index)
        for face in faces:
            value = _face_data(data, face)
            if value is None:
                value = _zero_value()
            else:
                value = _data_in_space(value, V)
            bcs.append(DirichletBC(V, value, face.dirichlet_subdomain))
    return bcs


def _data_in_space(value, V):
    """Boundary data as a Function in V (interpolated, or L2-projected where V cannot interpolate).

    Giving DirichletBC a Function in V means Firedrake never needs its own interpolation or
    projection fallback, which fails for elements without a usable dual basis (trimmed
    serendipity, BDMCE/BDMCF). For interpolable elements (CG, N1curl, RT, ...) this is the same
    interpolation DirichletBC would do itself.
    """
    from .harmonic_forms import interpolate_or_project  # local import: harmonic_forms imports this module
    if isinstance(value, Function) and value.function_space() == V:
        return value
    return interpolate_or_project(value, V)


def build_natural_terms(tau, v, faces, g_u, g_du, k, dim, complex, n, degree=None):
    """Weak boundary terms of the natural BCs (TODO-elliptic §1.1, §1.6).

    Returns the pair (sigma_term, u_term) with
        sigma_term = - sum_faces int_face B_{k-1}(tau, g_u)   (rhs of the sigma equation)
        u_term     = + sum_faces int_face B_k(v, g_du)        (rhs of the u equation)
    where B_j is dimswe.differential_forms.boundary_pairing. Either entry is the integer 0 when
    it vanishes: homogeneous data, no natural faces, k = 0 (no sigma, tau is ignored) or k = n
    (d u = 0).

    Args:
        tau: test function for sigma ((k-1)-form), ignored when k = 0.
        v: test function for u (k-form).
        faces: iterable of natural BoundaryFaces.
        g_u: data for tr *u, given as the full k-form field u_D; None, one expression, or dict
            face name -> expression.
        g_du: data for tr *du, given as the full (k+1)-form field (du)_D; same conventions.
        k, dim, complex: form degree, dimension and 2D complex.
        n: outward unit normal (FacetNormal of the mesh).
        degree: optional quadrature degree for the facet integrals.
    """
    sigma_term = 0
    u_term = 0
    for face in faces:
        measure = measure_for(face, degree=degree)
        gu = _face_data(g_u, face)
        if k > 0 and gu is not None:
            sigma_term = sigma_term - boundary_pairing(tau, gu, k - 1, dim, complex, n) * measure
        gdu = _face_data(g_du, face)
        if k < dim and gdu is not None:
            u_term = u_term + boundary_pairing(v, gdu, k, dim, complex, n) * measure
    return sigma_term, u_term


def verify_boundary_registry(mesh, registry, bounds, tol=1e-10):
    """Check geometrically that every registry face sits where its logical name says it does.

    For each face, the mean of the normal coordinate over the face (computed with its measure)
    must equal the corresponding domain bound, and the face area must be positive.

    Args:
        mesh: the mesh.
        registry: its face registry.
        bounds: domain bounds ((x0, x1), ...), e.g. mesh.domain_bounds.
        tol: absolute tolerance on the mean coordinate.

    Raises ValueError describing the first mismatch.
    """
    X = SpatialCoordinate(mesh)
    for face in registry.values():
        axis = DIRECTIONS.index(face.direction)
        measure = measure_for(face, domain=mesh)
        # The measure carries domain=mesh, so a plain Constant integrand is fine.
        area = assemble(Constant(1.0) * measure)
        if area <= 0.0:
            raise ValueError("face " + face.name + " has zero area")
        mean = assemble(X[axis] * measure) / area
        expected = bounds[axis][face.side]
        if abs(mean - expected) > tol:
            raise ValueError("face " + face.name + " is at " + face.direction + " = " + str(mean) + ", expected " + str(expected))
