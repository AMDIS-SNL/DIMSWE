from firedrake import IntervalMesh, PeriodicIntervalMesh, RectangleMesh, PeriodicRectangleMesh, BoxMesh, PeriodicBoxMesh, Mesh, FacetNormal
from firedrake import FunctionSpace, VectorFunctionSpace, dx, ds, dS, ExtrudedMesh
from firedrake import Function, CellDiameter
import finat
import FIAT

from .ufl_helpers import ufl_attribute

# Coordinate directions, in order, for 1D/2D/3D meshes.
DIRECTIONS = ('x', 'y', 'z')


class MeshSpec:
    """Description of a mesh type (TODO-elliptic §4.14).

    Attributes:
        dim: topological = geometric dimension (immersed manifolds are not supported yet).
        topology: 'box' (axis-aligned box, possibly periodic: the face registry, Künneth harmonic
            forms and box problem data apply) or 'radial' (disk, annulus, ball: faces 'outer' and
            'inner' detected by radius, numerical harmonic forms). Firedrake's CylinderMesh and the
            sphere/torus meshes are immersed surfaces (stage 3 of TODO-elliptic §4.14).
        base: for box aliases, the canonical box type ('line', 'rectangle', 'box') it maps to.
        periodic: for box aliases, the default periodic directions ('all' or a tuple).
        unit: for box aliases, True if the domain is fixed to the unit box (Lx = Ly = Lz = 1,
            starting at the origin); False if the cfg/domain lengths are used.
        simplicial_only: the mesh only exists with simplicial cells.
        faces: for 'radial' meshes, the logical face names.
    """

    def __init__(self, dim, topology='box', base=None, periodic=(), unit=False, simplicial_only=False, faces=()):
        self.dim, self.topology, self.base = dim, topology, base
        self.periodic, self.unit, self.simplicial_only, self.faces = periodic, unit, simplicial_only, tuple(faces)


# Every mesh type DIMSWE knows. Box aliases reuse the canonical box builders (and their periodicity,
# registry and coordinate conventions); 'tensor-*' are graded boxes; 'radial' meshes are new.
MESH_TYPES = {
    'line': MeshSpec(1, base='line'),
    'line-periodic': MeshSpec(1, base='line', periodic='all'),
    'rectangle': MeshSpec(2, base='rectangle'),
    'rectangle-periodic': MeshSpec(2, base='rectangle', periodic='all'),
    'box': MeshSpec(3, base='box'),
    'box-periodic': MeshSpec(3, base='box', periodic='all'),
    # Firedrake's unit/square/cube convenience meshes, as box aliases.
    'unit-interval': MeshSpec(1, base='line', unit=True),
    'periodic-unit-interval': MeshSpec(1, base='line', periodic='all', unit=True),
    'unit-square': MeshSpec(2, base='rectangle', unit=True),
    'periodic-unit-square': MeshSpec(2, base='rectangle', periodic='all', unit=True),
    'square': MeshSpec(2, base='rectangle'),
    'periodic-square': MeshSpec(2, base='rectangle', periodic='all'),
    'unit-cube': MeshSpec(3, base='box', unit=True),
    'periodic-unit-cube': MeshSpec(3, base='box', periodic='all', unit=True),
    'cube': MeshSpec(3, base='box'),
    # Graded boxes from explicit coordinate lists (TensorRectangleMesh / TensorBoxMesh).
    'tensor-rectangle': MeshSpec(2, base='rectangle'),
    'tensor-box': MeshSpec(3, base='box'),
    # Non-box volume meshes (simplicial, curved boundaries approximated by the mesh).
    'unit-disk': MeshSpec(2, topology='radial', simplicial_only=True, faces=('outer',)),
    'annulus': MeshSpec(2, topology='radial', faces=('inner', 'outer')),
    'unit-ball': MeshSpec(3, topology='radial', simplicial_only=True, faces=('outer',)),
}

# Kept for backwards compatibility (code outside dimswe may use them).
LINE_MESH_TYPES = tuple(name for name, spec in MESH_TYPES.items() if spec.base == 'line')
RECTANGLE_MESH_TYPES = tuple(name for name, spec in MESH_TYPES.items() if spec.base == 'rectangle')
BOX_MESH_TYPES = tuple(name for name, spec in MESH_TYPES.items() if spec.base == 'box')


def get_mesh_spec(mesh_type):
    """The MeshSpec of a mesh type, or None for 'file' (ValueError for unknown types)."""
    if mesh_type == 'file':
        return None
    if mesh_type not in MESH_TYPES:
        raise ValueError("mesh " + str(mesh_type) + " is unknown (known: " + str(sorted(MESH_TYPES)) + ", file)")
    return MESH_TYPES[mesh_type]


def is_box_mesh_type(mesh_type):
    """True for axis-aligned box meshes (the face registry, Künneth rule and box problem data apply)."""
    spec = get_mesh_spec(mesh_type)
    return spec is not None and spec.topology == 'box'

# FEEC families that need genuine (non-extruded) hexahedral meshes in 3D (see TODO-elliptic §4.9).
SERENDIPITY_FAMILIES = ('S-', 'S')

def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result

def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())

def get_mesh_type_dimension(mesh_type):
    """Return the spatial dimension (1, 2 or 3) of a built-in mesh type, or None for 'file'.

    Raises ValueError for unknown mesh types.
    """
    spec = get_mesh_spec(mesh_type)
    return None if spec is None else spec.dim

def set_dimension(parameters, mesh=None):
    """Store the spatial dimension of the mesh in parameters['mesh']['dim'].

    For built-in mesh types it follows from the type name. For 'file' meshes it is read from the
    mesh (its geometric dimension), so mesh must be given in that case.
    """
    dim = get_mesh_type_dimension(parameters['mesh']['type'])
    if dim is None:
        if mesh is None:
            raise ValueError("the dimension of a 'file' mesh can only be set once the mesh is loaded")
        dim = ufl_attribute(mesh, 'geometric_dimension')
    parameters['mesh']['dim'] = dim

def get_periodic_directions(mesh_parameters):
    """Return the periodic directions of a mesh as a tuple drawn from ('x', 'y', 'z'), in order.

    mesh_parameters is the 'mesh' section of the parameters. If the optional key
    'periodic_directions' is present (and not None) it overrides the periodicity implied by the
    mesh type, so for example 'rectangle' with periodic_directions [x] is periodic in x only, and
    'rectangle-periodic' with periodic_directions [] is not periodic at all. It may be a list of
    direction names, a single name ('x'), 'both'/'all' (every direction) or 'none'/[] (bounded).
    Without the key, '*-periodic' types are periodic in every direction and the rest in none.
    'file' meshes are never treated as periodic here.

    Raises ValueError for unknown or out-of-range directions, and for partial periodicity in 1D.
    """
    mesh_type = mesh_parameters['type']
    spec = get_mesh_spec(mesh_type)
    requested = mesh_parameters.get('periodic_directions', None)
    if spec is None or spec.topology != 'box':
        if requested:
            raise ValueError("periodic_directions is only supported for box meshes, not '" + str(mesh_type) + "'")
        return ()
    dim = spec.dim
    valid = DIRECTIONS[:dim]
    if requested is None:
        return valid if spec.periodic == 'all' else tuple(spec.periodic)

    # Normalize the different accepted spellings to a list of direction names.
    if isinstance(requested, str):
        key = requested.strip().lower()
        if key in ('both', 'all'):
            requested = list(valid)
        elif key == 'none':
            requested = []
        else:
            requested = [key]
    requested = [str(d).strip().lower() for d in requested]
    for d in requested:
        if d not in valid:
            raise ValueError("periodic direction " + d + " is not valid for a " + str(dim) + "D mesh (valid: " + str(list(valid)) + ")")
    # Keep the canonical x, y, z order and drop duplicates.
    return tuple(d for d in valid if d in requested)

def get_domain_bounds(mesh_parameters, domain, periodic_directions):
    """Return ((x0, x1), ...) for the box-shaped domain of a built-in mesh, one pair per direction.

    These are the bounds used when the mesh is constructed (see build_mesh), so this function is
    the single place that defines DIMSWE's coordinate conventions:
    - non-periodic line and rectangle meshes are centered at (domain.xc, domain.yc);
    - meshes with at least one periodic direction start at the origin, since Firedrake's
      periodic meshes do;
    - box meshes always occupy [0, Lx] x [0, Ly] x [0, Lz].
    Returns None for 'file' meshes.
    """
    spec = get_mesh_spec(mesh_parameters['type'])
    if spec is None or spec.topology != 'box':
        return None
    dim = spec.dim
    if spec.unit:
        return tuple((0., 1.) for _ in range(dim))
    if mesh_parameters['type'].startswith('tensor-'):
        return tuple((float(min(c)), float(max(c))) for c in _tensor_coordinates(mesh_parameters, dim))
    lengths = [domain.Lx, getattr(domain, 'Ly', None), getattr(domain, 'Lz', None)][:dim]
    if dim < 3 and len(periodic_directions) == 0:
        centers = [domain.xc, getattr(domain, 'yc', None)][:dim]
        return tuple((c - L / 2., c + L / 2.) for c, L in zip(centers, lengths))
    return tuple((0., L) for L in lengths)

def mesh_size(msh):
    """Return the maximum cell diameter of msh (computed in parallel-safe fashion via PETSc)."""
    DG0 = FunctionSpace(msh, "DG", 0)
    h = Function(DG0).interpolate(CellDiameter(msh))
    with h.dat.vec as hvec:
        _, maxh = hvec.max()
    return maxh

def set_mesh_spacing(mesh, parameters, domain):
    """Attach the uniform cell spacings (mesh.dx, mesh.dy, mesh.dz) used by forcing terms.

    These attributes are set when DIMSWE builds a periodic mesh itself, but are missing on a
    mesh loaded from a checkpoint, so they are (re)derived here from the parameters.
    The behaviour for fully periodic meshes is unchanged; partially periodic meshes (see
    get_periodic_directions) get the same attributes.
    """
    mesh_type = parameters['mesh']['type']
    if not is_box_mesh_type(mesh_type) or mesh_type.startswith('tensor-'):
        return  # spacings are only defined for uniform box meshes
    domain = _box_domain(parameters['mesh'], domain)
    periodic = get_periodic_directions(parameters['mesh'])
    simplicial = parameters['mesh'].get('simplicial_cells', False)
    # nx, ny, nz are the coarsest level of a refinement hierarchy (finest = n * 2**levels).
    refine = 2 ** get_refinement_levels(parameters['mesh'])
    if mesh_type in RECTANGLE_MESH_TYPES and len(periodic) > 0:
        mesh.dx = domain.Lx / (parameters['mesh']['nx'] * refine)
        mesh.dy = domain.Ly / (parameters['mesh']['ny'] * refine)
    elif mesh_type in BOX_MESH_TYPES and len(periodic) > 0 and not simplicial:
        mesh.dx = domain.Lx / (parameters['mesh']['nx'] * refine)
        mesh.dy = domain.Ly / (parameters['mesh']['ny'] * refine)
        mesh.dz = domain.Lz / (parameters['mesh']['nz'] * refine)

def annotate_mesh(mesh, parameters, domain):
    """Record mesh metadata needed downstream (boundary registry, harmonic forms, problem data).

    Sets mesh.periodic_directions (tuple of 'x'/'y'/'z'), mesh.domain_bounds (see
    get_domain_bounds; None for non-box meshes), mesh.mesh_topology ('box', 'radial' or 'file')
    and the cell spacings of set_mesh_spacing.
    """
    periodic = get_periodic_directions(parameters['mesh'])
    mesh.periodic_directions = periodic
    mesh.domain_bounds = get_domain_bounds(parameters['mesh'], domain, periodic)
    spec = get_mesh_spec(parameters['mesh']['type'])
    mesh.mesh_topology = 'file' if spec is None else spec.topology
    set_mesh_spacing(mesh, parameters, domain)

def _diagonal_kwargs(mp):
    """Return {'diagonal': value} if mesh parameter 'diagonal' is set, else {}.

    Firedrake's mesh constructors have their own (constructor-specific) default diagonal and
    reject diagonal=None for simplicial cells, so an unset/blank cfg entry must not be passed on.
    Quadrilateral/hexahedral meshes ignore the argument.
    """
    diagonal = mp.get('diagonal', None)
    return {} if diagonal is None else {'diagonal': diagonal}

def _build_line_mesh(mp, domain, periodic):
    """Interval mesh: fully periodic ([0, Lx]) or bounded (centered at domain.xc)."""
    if periodic:
        return PeriodicIntervalMesh(mp['nx'], domain.Lx)
    left = domain.xc - domain.Lx/2.
    right = domain.xc + domain.Lx/2.
    return IntervalMesh(mp['nx'], left, right)

def _build_rectangle_mesh(mp, domain, periodic):
    """Rectangle mesh with any periodicity (none, x, y or both).

    Bounded rectangles are centered at (domain.xc, domain.yc); (partially) periodic ones start at
    the origin. Boundary ids follow Firedrake: RectangleMesh uses 1-4 (x=x0, x=x1, y=y0, y=y1);
    PeriodicRectangleMesh with direction 'x' or 'y' keeps ids 1, 2 for the two remaining faces.
    """
    quadrilateral = not mp.get('simplicial_cells', False)
    diagonal = _diagonal_kwargs(mp)
    if len(periodic) == 0:
        left_x = domain.xc - domain.Lx/2.
        right_x = domain.xc + domain.Lx/2.
        left_y = domain.yc - domain.Ly/2.
        right_y = domain.yc + domain.Ly/2.
        return RectangleMesh(mp['nx'], mp['ny'], right_x, right_y, originX=left_x, originY=left_y, quadrilateral=quadrilateral, **diagonal)
    if periodic == ('x', 'y'):
        # Identical to the historic 'rectangle-periodic' construction.
        return PeriodicRectangleMesh(mp['nx'], mp['ny'], domain.Lx, domain.Ly, quadrilateral=quadrilateral, **diagonal)
    return PeriodicRectangleMesh(mp['nx'], mp['ny'], domain.Lx, domain.Ly, direction=periodic[0], quadrilateral=quadrilateral, **diagonal)

def _build_base_rectangle(mp, domain, periodic_xy):
    """Quadrilateral base mesh on [0, Lx] x [0, Ly] for extruded boxes, periodic in periodic_xy."""
    if len(periodic_xy) == 0:
        return RectangleMesh(mp['nx'], mp['ny'], domain.Lx, domain.Ly, quadrilateral=True)
    if periodic_xy == ('x', 'y'):
        return PeriodicRectangleMesh(mp['nx'], mp['ny'], domain.Lx, domain.Ly, quadrilateral=True)
    return PeriodicRectangleMesh(mp['nx'], mp['ny'], domain.Lx, domain.Ly, direction=periodic_xy[0], quadrilateral=True)

def _build_box_mesh(mp, domain, periodic, family):
    """Box mesh on [0, Lx] x [0, Ly] x [0, Lz] with any periodicity.

    - hexahedral cells, family 'Q' (default): ExtrudedMesh of a quadrilateral (periodic) rectangle,
      periodic in z if requested. H(curl)/H(div) elements (NCE/NCF) on hexahedra are only
      supported on extruded meshes, and this covers all 8 periodicity combinations. Faces: base
      ids via ds_v, z faces via ds_b/ds_t ("bottom"/"top" for DirichletBC).
    - hexahedral cells, serendipity families (TODO-elliptic §4.9): genuine BoxMesh when bounded,
      PeriodicBoxMesh(directions=..., hexahedral=True) otherwise, which covers all 8
      periodicity combinations too. PeriodicBoxMesh keeps the fixed boundary ids 1-6 (x0, x1,
      y0, y1, z0, z1) for the remaining faces, and needs >= 3 cells per direction.
    - simplicial (tetrahedral) cells: BoxMesh or fully periodic PeriodicBoxMesh only, unchanged
      from the historic construction; Firedrake's PeriodicBoxMesh supports `directions` only
      with hexahedral=True, so partial periodicity raises NotImplementedError.
    """
    simplicial = mp.get('simplicial_cells', False)
    if simplicial or family in SERENDIPITY_FAMILIES:
        hexahedral = not simplicial
        if len(periodic) == 0:
            return BoxMesh(mp['nx'], mp['ny'], mp['nz'], domain.Lx, domain.Ly, domain.Lz, hexahedral=hexahedral, **_diagonal_kwargs(mp))
        if periodic == DIRECTIONS:
            return PeriodicBoxMesh(mp['nx'], mp['ny'], mp['nz'], domain.Lx, domain.Ly, domain.Lz, hexahedral=hexahedral)
        if simplicial:
            raise NotImplementedError("partially periodic tetrahedral box meshes are not supported by Firedrake's PeriodicBoxMesh (directions requires hexahedral cells); use hexahedral cells or periodicity in all/no directions")
        directions = tuple(d in periodic for d in DIRECTIONS)
        return PeriodicBoxMesh(mp['nx'], mp['ny'], mp['nz'], domain.Lx, domain.Ly, domain.Lz, directions=directions, hexahedral=hexahedral)
    base = _build_base_rectangle(mp, domain, tuple(d for d in periodic if d != 'z'))
    return ExtrudedMesh(base, mp['nz'], layer_height=domain.Lz / mp['nz'], extrusion_type='uniform', periodic='z' in periodic)

class _UnitDomain:
    """The unit box as a domain object (Lx = Ly = Lz = 1, centers at 1/2), for unit-* mesh types."""
    Lx = Ly = Lz = 1.0
    xc = yc = zc = 0.5


def _box_domain(mesh_parameters, domain):
    """The domain the canonical box builders use: the unit box for unit-* aliases, else domain."""
    return _UnitDomain() if get_mesh_spec(mesh_parameters['type']).unit else domain


def _tensor_coordinates(mesh_parameters, dim):
    """Coordinate lists of a tensor-rectangle/tensor-box mesh (mesh.x_coords, y_coords, z_coords).

    Each list needs >= 2 strictly increasing values; ValueError otherwise.
    """
    lists = []
    for name in ('x_coords', 'y_coords', 'z_coords')[:dim]:
        values = mesh_parameters.get(name)
        if values is None or len(values) < 2:
            raise ValueError("mesh type " + mesh_parameters['type'] + " needs mesh." + name + " with >= 2 values")
        values = [float(v) for v in values]
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError("mesh." + name + " must be strictly increasing")
        lists.append(values)
    return lists


def _build_tensor_mesh(mp, dim):
    """Graded box from explicit coordinates: TensorRectangleMesh (2D) or TensorBoxMesh (3D)."""
    import numpy
    from firedrake import TensorBoxMesh, TensorRectangleMesh
    coordinates = [numpy.array(c) for c in _tensor_coordinates(mp, dim)]
    if dim == 2:
        return TensorRectangleMesh(*coordinates, quadrilateral=not mp.get('simplicial_cells', False),
                                   **_diagonal_kwargs(mp))
    if not mp.get('simplicial_cells', False):
        raise NotImplementedError("tensor-box only exists with simplicial_cells: true (hexahedral TensorBoxMesh "
                                  "has no NCE/NCF spaces); use box with an extruded mesh for family Q")
    return TensorBoxMesh(*coordinates)


def _radial_value(mp, key, default):
    """A positive float cfg entry of a radial mesh."""
    value = float(mp.get(key, default))
    if value <= 0.0:
        raise ValueError("mesh." + key + " must be positive")
    return value


def _build_radial_mesh(mp):
    """Disk, annulus or ball (TODO-elliptic §4.14).

    cfg entries (all optional):
    - unit-disk, unit-ball: refinement (Firedrake refinement_level, default 3 for the disk, 2 for
      the ball);
    - annulus: inner_radius (default 0.5), outer_radius (1), nr (radial cells, default nx or 4),
      nt (angular cells, default 4 * nr); quadrilateral unless simplicial_cells: true.
    The faces are detected by radius (boundary_conditions.radial_face_registry).
    """
    from firedrake import UnitBallMesh, UnitDiskMesh
    mesh_type = mp['type']
    if mesh_type == 'unit-disk':
        return UnitDiskMesh(refinement_level=int(mp.get('refinement', 3)))
    if mesh_type == 'unit-ball':
        return UnitBallMesh(refinement_level=int(mp.get('refinement', 2)))
    if mesh_type == 'annulus':
        from firedrake import AnnulusMesh
        inner = _radial_value(mp, 'inner_radius', 0.5)
        outer = _radial_value(mp, 'outer_radius', 1.0)
        if not inner < outer:
            raise ValueError("annulus needs inner_radius < outer_radius")
        nr = int(mp.get('nr', mp.get('nx', 4)))
        nt = int(mp.get('nt', 4 * nr))
        quadrilateral = not mp.get('simplicial_cells', False)
        try:
            return AnnulusMesh(outer, inner, nr=nr, nt=nt, quadrilateral=quadrilateral)
        except TypeError:
            if not quadrilateral:
                raise NotImplementedError("this Firedrake's AnnulusMesh has no quadrilateral option; it only "
                                          "builds its default cells (use simplicial_cells: false)")
            return AnnulusMesh(outer, inner, nr=nr, nt=nt)
    raise ValueError("no builder for radial mesh " + str(mesh_type))


def get_refinement_levels(mesh_parameters):
    """mesh.refinement_levels (default 0): number of uniform refinements of a multigrid hierarchy."""
    levels = mesh_parameters.get('refinement_levels', 0) or 0
    if not isinstance(levels, int) or levels < 0:
        raise ValueError("mesh.refinement_levels must be a non-negative integer, got " + repr(levels))
    if levels > 0 and mesh_parameters['type'] == 'file':
        raise ValueError("mesh.refinement_levels is not supported for 'file' meshes")
    return levels


def _build_hierarchy_mesh(parameters, domain, dim, periodic, levels):
    """Finest mesh of a uniform refinement hierarchy for geometric multigrid (TODO-elliptic §4.12).

    nx, ny, nz describe the coarsest mesh; every level halves the cells (so the finest mesh has
    nx * 2**levels cells per direction). Box meshes of the Q family use an ExtrudedMeshHierarchy
    of the refined base rectangles with the layers doubled per level (periodic in z if requested);
    all other meshes a MeshHierarchy of the coarse mesh. The hierarchy is kept alive on the finest
    mesh (mesh.mesh_hierarchy), which is what Firedrake's PCMG/PMGPC look up.
    """
    from firedrake import ExtrudedMeshHierarchy, MeshHierarchy
    mp = parameters['mesh']
    family = parameters['spatial-discretization']['family']
    extruded_box = dim == 3 and not mp.get('simplicial_cells', False) and family not in SERENDIPITY_FAMILIES
    if extruded_box:
        from functools import partial
        base = _build_base_rectangle(mp, domain, tuple(d for d in periodic if d != 'z'))
        base_hierarchy = MeshHierarchy(base, levels)
        builder = partial(ExtrudedMesh, periodic='z' in periodic)
        hierarchy = ExtrudedMeshHierarchy(base_hierarchy, domain.Lz, base_layer=mp['nz'], refinement_ratio=2,
                                          extrusion_type='uniform', mesh_builder=builder)
    else:
        if dim == 1:
            coarse = _build_line_mesh(mp, domain, periodic)
        elif dim == 2:
            coarse = _build_rectangle_mesh(mp, domain, periodic)
        else:
            coarse = _build_box_mesh(mp, domain, periodic, family)
        hierarchy = MeshHierarchy(coarse, levels)
    mesh = hierarchy[-1]
    mesh.mesh_hierarchy = hierarchy
    mesh.refinement_levels = levels
    return mesh


def build_mesh(parameters, domain):
    """Construct the mesh described by parameters['mesh'] (not yet annotated, see annotate_mesh).

    domain is any object describing the box-shaped domain (duck-typed): it needs Lx (and Ly in
    2D/3D, Lz in 3D) and, for bounded line/rectangle meshes, the centers xc (and yc). Both the
    initial-condition objects of the time-dependent models and the elliptic problem-data objects
    satisfy this contract. It is not used for 'file' meshes.
    """
    mp = parameters['mesh']
    mesh_type = mp['type']
    spec = get_mesh_spec(mesh_type)
    periodic = get_periodic_directions(mp)
    levels = get_refinement_levels(mp)
    if spec is not None and spec.simplicial_only and not mp.get('simplicial_cells', False):
        raise ValueError("mesh type " + mesh_type + " only exists with simplicial cells: set simplicial_cells: true")
    if spec is not None and spec.topology == 'radial':
        if levels > 0:
            raise NotImplementedError("mesh.refinement_levels on curved meshes is not supported yet "
                                      "(TODO-elliptic §4.14, stage 4)")
        return _build_radial_mesh(mp)
    if mesh_type.startswith('tensor-'):
        if levels > 0:
            raise NotImplementedError("mesh.refinement_levels is not supported for tensor-* meshes")
        return _build_tensor_mesh(mp, spec.dim)
    dim = None if spec is None else spec.dim
    box_domain = None if spec is None else _box_domain(mp, domain)
    if levels > 0 and dim is not None:
        return _build_hierarchy_mesh(parameters, box_domain, dim, periodic, levels)
    if dim == 1:
        return _build_line_mesh(mp, box_domain, periodic)
    if dim == 2:
        return _build_rectangle_mesh(mp, box_domain, periodic)
    if dim == 3:
        family = parameters['spatial-discretization']['family']
        return _build_box_mesh(mp, box_domain, periodic, family)
    # 'file' mesh
    mesh = Mesh(mp['meshfile'])
    if mp.get('scale_center_meshfile_coords', False):
#ADD THIS
        pass
#POSSIBLY NEED TO SCALE+CENTER MESH COORDINATES
    return mesh

def get_mesh_and_spaces(parameters, domain, mesh=None):
    """Build (or reuse) the mesh and the de Rham complex of spaces on it.

    domain describes the box-shaped domain; see build_mesh for the duck-typed contract (Lx, Ly,
    Lz, xc, yc). It can be an initial-condition object or an elliptic problem-data object.

    If mesh is given (for example a mesh loaded from a truth checkpoint), it is reused instead of
    constructing a new one, so functions loaded from that checkpoint live on the model's mesh.
    The caller is responsible for the mesh matching the mesh parameters.

    The optional parameters['mesh']['periodic_directions'] selects partial periodicity (see
    get_periodic_directions). The resulting periodic directions and domain bounds are recorded on
    the mesh and on the returned spaces (spaces.periodic_directions, spaces.domain_bounds).
    """
    if mesh is None:
        mesh = build_mesh(parameters, domain)
    annotate_mesh(mesh, parameters, domain)
    set_dimension(parameters, mesh)
    spaces = DeRhamComplex(mesh, parameters)
    return [mesh, spaces]

#LOTS OF CHOICES HERE- QUADS VS TRIANGLES, DIFFERENT FAMILIES, ETC.

# FEEC families supported by DeRhamComplex (TODO-elliptic §4.9).
FEEC_FAMILIES = ('Q', 'P-', 'P+', 'S-', 'S')

# Cells each family can be built on (cellname of the mesh; 'extruded' = quad x interval,
# 'extruded2d' = interval x interval).
FAMILY_CELLS = {
    'Q': ('interval', 'quadrilateral', 'extruded', 'extruded2d'),
    'P-': ('interval', 'triangle', 'tetrahedron'),
    'P+': ('interval', 'triangle', 'tetrahedron'),
    'S-': ('interval',),
    'S': ('interval', 'quadrilateral'),
}


def family_elements(family, dim, r):
    """Element table of the discrete de Rham complexes: {space: (element, degree, variant)}.

    Keys are 'CG' (0-forms), 'Hcurl' (1-forms), 'Hdiv' ((n-1)-forms) and 'DG' (n-forms); a value
    of None means the space does not exist. Degrees follow the UFL/FIAT convention in which
    N1curl/RT/RTCE/RTCF/NCE/NCF/Sminus* of degree r are the trimmed spaces containing all
    polynomials of degree r - 1, and N2curl/BDM/BDMCE/BDMCF of degree s contain all of degree s.

    | family | 0-forms | 1-forms (Hcurl)            | (n-1)-forms (Hdiv)           | n-forms   |
    |--------|---------|----------------------------|------------------------------|-----------|
    | Q      | CG r    | RTCE r / NCE r (extruded)  | RTCF r / NCF r (extruded)    | DG r-1    |
    | P-     | CG r    | N1curl r                   | RT r                         | DG r-1    |
    | P+     | CG r    | N2curl r-1                 | BDM r-(n-1)                  | DG r-n    |
    | S-     | S r     | SminusCurl r (unusable)    | SminusDiv r (unusable)       | DPC r-1   |
    | S      | S r     | BDMCE r-1 (2D only)        | BDMCF r-1 (2D only)          | DPC r-2   |

    (2D / 3D where they differ.) In 1D every family reduces to CG r -> DG r-1 and, as for the
    existing models, Hdiv is the CG space and Hcurl does not exist. Q keeps variant 'spectral'
    for CG/DG (unchanged from the original implementation); the other families use the default
    variants. The serendipity families raise NotImplementedError in 3D (Firedrake supports only
    Q/DQ elements on hexahedra), and S- also in 2D (FIAT's trimmed serendipity elements lack
    dual_basis(), which even assembly needs); S- is only the CG/DG complex in 1D. Several of these elements (RTCE/RTCF/NCE/NCF, Sminus*, BDMCE/F)
    cannot be interpolated into; use harmonic_forms.interpolate_or_project to put data in them.

    Raises ValueError for unknown families, out-of-range dimensions, and orders below the
    family's minimum (P+: r >= n; S: r >= 2 in 2D; all: r >= 1).
    """
    if family not in FEEC_FAMILIES:
        raise ValueError("FEEC family " + str(family) + " is unknown (known: " + str(list(FEEC_FAMILIES)) + ")")
    if dim not in (1, 2, 3):
        raise ValueError("dimension " + str(dim) + " is not supported")
    r = int(r)
    minimum = {'P+': dim, 'S': 2 if dim > 1 else 1}.get(family, 1)
    if r < minimum:
        raise ValueError("family " + family + " in " + str(dim) + "D needs order >= " + str(minimum) + ", got " + str(r))
    spectral = 'spectral' if family == 'Q' or dim == 1 else None
    if dim == 1:
        # Same complex for every family (Q builder); Hdiv = CG as in the existing models.
        return {'CG': ('CG', r, 'spectral'), 'Hcurl': None, 'Hdiv': ('CG', r, 'spectral'), 'DG': ('DG', r - 1, 'spectral')}
    if family == 'Q':
        curl, div = ('RTCE', 'RTCF') if dim == 2 else ('NCE', 'NCF')
        return {'CG': ('CG', r, 'spectral'), 'Hcurl': (curl, r, None), 'Hdiv': (div, r, None), 'DG': ('DG', r - 1, 'spectral')}
    if family == 'P-':
        return {'CG': ('CG', r, None), 'Hcurl': ('N1curl', r, None), 'Hdiv': ('RT', r, None), 'DG': ('DG', r - 1, None)}
    if family == 'P+':
        return {'CG': ('CG', r, None), 'Hcurl': ('N2curl', r - 1, None), 'Hdiv': ('BDM', r - (dim - 1), None),
                'DG': ('DG', r - dim, None)}
    if dim == 3 and family in ('S-', 'S'):
        # Firedrake only supports Q, DQ and Real elements on (non-extruded) hexahedral meshes, so
        # neither serendipity family exists in 3D (the trimmed SminusE/SminusF would be the 3D
        # S- spaces if that changes).
        raise NotImplementedError("serendipity families (S-, S) are not available in 3D: Firedrake only supports "
                                  "Q/DQ elements on hexahedral meshes; use Q (extruded) or P-/P+ (tetrahedra)")
    if family == 'S-':
        # The trimmed serendipity spaces would be S r -> SminusCurl r / SminusDiv r -> DPC r-1, but
        # FIAT's TrimmedSerendipityCurl/Div do not implement dual_basis(), which FIAT also uses for
        # space_dimension(), so no form on them can even be assembled.
        raise NotImplementedError("family S- (trimmed serendipity) is not usable with the installed FIAT: "
                                  "TrimmedSerendipityCurl/Div lack dual_basis() (needed even for assembly); "
                                  "use S, Q or P-/P+")
    # Full serendipity S_r (2D).
    return {'CG': ('S', r, None), 'Hcurl': ('BDMCE', r - 1, None), 'Hdiv': ('BDMCF', r - 1, None), 'DG': ('DPC', r - 2, None)}


def mesh_cell_name(mesh):
    """'interval' | 'triangle' | 'quadrilateral' | 'tetrahedron' | 'hexahedron' | 'extruded' | 'extruded2d'.

    'extruded' is a 3D extruded mesh (quadrilateral x interval: NCE/NCF); 'extruded2d' a 2D one
    (interval x interval, e.g. Firedrake's quadrilateral AnnulusMesh: RTCE/RTCF, which Firedrake
    builds on TensorProductCell(interval, interval)). Prism meshes (triangle x interval) are
    reported as 'extruded-prism' and have no Q complex.
    """
    cell_dimension = ufl_attribute(mesh, 'cell_dimension')
    if isinstance(cell_dimension, tuple):
        if sum(cell_dimension) == 2:
            return 'extruded2d'
        base = ufl_attribute(mesh.ufl_cell().sub_cells[0], 'cellname') if hasattr(mesh.ufl_cell(), 'sub_cells') \
            else 'quadrilateral'
        return 'extruded' if base == 'quadrilateral' else 'extruded-prism'
    return ufl_attribute(mesh.ufl_cell(), 'cellname')


def _space(mesh, element, vector=False):
    """FunctionSpace (or VectorFunctionSpace) for an (element, degree, variant) table entry."""
    name, degree, variant = element
    kwargs = {} if variant is None else {'variant': variant}
    if vector:
        return VectorFunctionSpace(mesh, name, degree, **kwargs)
    return FunctionSpace(mesh, name, degree, **kwargs)


class DeRhamComplex():
    """Discrete de Rham complex CG -> Hcurl -> Hdiv -> DG (plus CGV, DGV) of a FEEC family.

    Families (spatial-discretization.family): Q (tensor-product, quads/extruded hexes), P-
    and P+ (simplices), S- and S (serendipity, non-extruded quads/hexes); see family_elements
    for the element table. Every family sets the same attributes (CG, Hcurl, Hdiv, DG, CGV, DGV,
    dx, dS, ds, n), so get_diff_form_space and the existing models work unchanged; HdivV/HcurlV
    stay unset as before. Only Q supports lump_mass (GLL quadrature).
    """

    def __init__(self, mesh, parameters):
        """Build the discrete de Rham complex (CG, Hcurl, Hdiv, DG and vector spaces) on mesh.

        Also exposes the mesh metadata recorded by annotate_mesh: periodic_directions (tuple of
        'x'/'y'/'z', empty if unknown) and domain_bounds (None if unknown), which the boundary
        registry and the harmonic-form rule of the elliptic problems need.
        """
        disc = parameters['spatial-discretization']
        self.n = FacetNormal(mesh)
        self.order = disc['order']
        self.family = disc['family']
        self.mesh = mesh
        self.periodic_directions = getattr(mesh, 'periodic_directions', ())
        self.domain_bounds = getattr(mesh, 'domain_bounds', None)
        if self.family not in FEEC_FAMILIES:
            raise ValueError("FEEC family " + str(self.family) + " is unknown")
        self.cell = mesh_cell_name(mesh)
        self.dim = {'extruded': 3, 'extruded2d': 2, 'extruded-prism': 3}.get(
            self.cell, None) or ufl_attribute(mesh, 'topological_dimension')
        if self.cell not in FAMILY_CELLS[self.family]:
            raise ValueError("FEEC family " + self.family + " is not available on " + self.cell + " cells "
                             "(supported: " + str(list(FAMILY_CELLS[self.family])) + "; P-/P+ need "
                             "simplicial_cells: true, Q/S-/S need simplicial_cells: false, 3D Q needs the extruded box)")
        if disc.get('lump_mass', False) and self.family != 'Q':
            raise ValueError("lump_mass (GLL quadrature) is only available for family Q")
        builders = {'Q': self._build_Q, 'P-': self._build_Pminus, 'P+': self._build_P,
                    'S-': self._build_Sminus, 'S': self._build_S}
        # Serendipity and simplicial families reduce to the Q complex on intervals.
        builder = self._build_Q if self.dim == 1 else builders[self.family]
        builder()
        self._build_measures(parameters)

    # --- one builder per family (they share _build_from_table) --------------------------------

    def _build_from_table(self, family):
        """Create CG, Hcurl, Hdiv, DG, CGV, DGV from family_elements(family, dim, order)."""
        table = family_elements(family, self.dim, self.order)
        self.elements = table
        self.CG = _space(self.mesh, table['CG'])
        self.DG = _space(self.mesh, table['DG'])
        if self.dim == 1:
            # As in the original implementation: 1-form "Hdiv" is CG, vectors are scalars.
            self.DGV = self.DG
            self.Hdiv = self.CG
            self.Hcurl = None
            self.CGV = self.CG
            return
        self.DGV = _space(self.mesh, table['DG'], vector=True)
        self.CGV = _space(self.mesh, table['CG'], vector=True)
        self.Hdiv = _space(self.mesh, table['Hdiv'])
        self.Hcurl = _space(self.mesh, table['Hcurl'])

    def _build_Q(self):
        """Tensor-product Q_r^- family (CG/RTCE/RTCF/DG, NCE/NCF on extruded hexes); unchanged."""
        self._build_from_table('Q')

    def _build_Pminus(self):
        """Simplicial P_r^- family (CG, N1curl, RT, DG of the same degree r)."""
        self._build_from_table('P-')

    def _build_P(self):
        """Simplicial full P_r family (CG r, N2curl r-1, BDM r-(n-1), DG r-n)."""
        self._build_from_table('P+')

    def _build_Sminus(self):
        """Trimmed serendipity S_r^- family (S, SminusCurl/E, SminusDiv/F, DPC r-1)."""
        self._build_from_table('S-')

    def _build_S(self):
        """Full serendipity S_r family in 2D (S r, BDMCE r-1, BDMCF r-1, DPC r-2)."""
        self._build_from_table('S')

    def _build_measures(self, parameters):
        """dx, dS, ds: GLL rules for Q with lump_mass, the default rules otherwise."""
        if parameters['spatial-discretization'].get('lump_mass', False):
            gll_rule = gauss_lobatto_legendre_cube_rule(dimension=parameters['mesh']['dim'], degree=ufl_attribute(self.CG.ufl_element(), 'degree'))
            gll_surf_rule = gauss_lobatto_legendre_cube_rule(dimension=parameters['mesh']['dim']-1, degree=ufl_attribute(self.CG.ufl_element(), 'degree'))
            self.dx = dx(scheme=gll_rule)
            self.dS = dS(scheme=gll_surf_rule)
            self.ds = ds(scheme=gll_surf_rule)
        else:
            self.dx = dx
            self.dS = dS
            self.ds = ds
