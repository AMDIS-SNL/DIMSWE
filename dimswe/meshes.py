from firedrake import IntervalMesh, PeriodicIntervalMesh, RectangleMesh, PeriodicRectangleMesh, BoxMesh, PeriodicBoxMesh, Mesh, FacetNormal
from firedrake import FunctionSpace, VectorFunctionSpace, dx, ds, dS, ExtrudedMesh
from firedrake import Function, CellDiameter
import finat
import FIAT

# Coordinate directions, in order, for 1D/2D/3D meshes.
DIRECTIONS = ('x', 'y', 'z')

# Mesh types built by DIMSWE, grouped by dimension.
LINE_MESH_TYPES = ('line', 'line-periodic')
RECTANGLE_MESH_TYPES = ('rectangle', 'rectangle-periodic')
BOX_MESH_TYPES = ('box', 'box-periodic')

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
    if mesh_type in LINE_MESH_TYPES:
        return 1
    if mesh_type in RECTANGLE_MESH_TYPES:
        return 2
    if mesh_type in BOX_MESH_TYPES:
        return 3
    if mesh_type == 'file':
        return None
    raise ValueError("mesh " + str(mesh_type) + " is unknown")

def set_dimension(parameters, mesh=None):
    """Store the spatial dimension of the mesh in parameters['mesh']['dim'].

    For built-in mesh types it follows from the type name. For 'file' meshes it is read from the
    mesh (its geometric dimension), so mesh must be given in that case.
    """
    dim = get_mesh_type_dimension(parameters['mesh']['type'])
    if dim is None:
        if mesh is None:
            raise ValueError("the dimension of a 'file' mesh can only be set once the mesh is loaded")
        dim = mesh.geometric_dimension()
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
    dim = get_mesh_type_dimension(mesh_type)
    requested = mesh_parameters.get('periodic_directions', None)
    if dim is None:
        if requested:
            raise ValueError("periodic_directions is not supported for mesh type 'file'")
        return ()
    valid = DIRECTIONS[:dim]
    if requested is None:
        return valid if mesh_type.endswith('-periodic') else ()

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
    dim = get_mesh_type_dimension(mesh_parameters['type'])
    if dim is None:
        return None
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
    periodic = get_periodic_directions(parameters['mesh'])
    simplicial = parameters['mesh'].get('simplicial_cells', False)
    if mesh_type in RECTANGLE_MESH_TYPES and len(periodic) > 0:
        mesh.dx = domain.Lx / parameters['mesh']['nx']
        mesh.dy = domain.Ly / parameters['mesh']['ny']
    elif mesh_type in BOX_MESH_TYPES and len(periodic) > 0 and not simplicial:
        mesh.dx = domain.Lx / parameters['mesh']['nx']
        mesh.dy = domain.Ly / parameters['mesh']['ny']
        mesh.dz = domain.Lz / parameters['mesh']['nz']

def annotate_mesh(mesh, parameters, domain):
    """Record mesh metadata needed downstream (boundary registry, harmonic forms, problem data).

    Sets mesh.periodic_directions (tuple of 'x'/'y'/'z'), mesh.domain_bounds (see
    get_domain_bounds; None for 'file' meshes) and the cell spacings of set_mesh_spacing.
    """
    periodic = get_periodic_directions(parameters['mesh'])
    mesh.periodic_directions = periodic
    mesh.domain_bounds = get_domain_bounds(parameters['mesh'], domain, periodic)
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

def build_mesh(parameters, domain):
    """Construct the mesh described by parameters['mesh'] (not yet annotated, see annotate_mesh).

    domain is any object describing the box-shaped domain (duck-typed): it needs Lx (and Ly in
    2D/3D, Lz in 3D) and, for bounded line/rectangle meshes, the centers xc (and yc). Both the
    initial-condition objects of the time-dependent models and the elliptic problem-data objects
    satisfy this contract. It is not used for 'file' meshes.
    """
    mp = parameters['mesh']
    mesh_type = mp['type']
    dim = get_mesh_type_dimension(mesh_type)
    periodic = get_periodic_directions(mp)
    if dim == 1:
        return _build_line_mesh(mp, domain, periodic)
    if dim == 2:
        return _build_rectangle_mesh(mp, domain, periodic)
    if dim == 3:
        family = parameters['spatial-discretization']['family']
        return _build_box_mesh(mp, domain, periodic, family)
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

class DeRhamComplex():

    def __init__(self, mesh, parameters):
        """Build the discrete de Rham complex (CG, Hcurl, Hdiv, DG and vector spaces) on mesh.

        Also exposes the mesh metadata recorded by annotate_mesh: periodic_directions (tuple of
        'x'/'y'/'z', empty if unknown) and domain_bounds (None if unknown), which the boundary
        registry and the harmonic-form rule of the elliptic problems need.
        """

        self.n = FacetNormal(mesh)
        self.order = parameters['spatial-discretization']['order']
        self.family = parameters['spatial-discretization']['family']
        self.mesh = mesh
        self.periodic_directions = getattr(mesh, 'periodic_directions', ())
        self.domain_bounds = getattr(mesh, 'domain_bounds', None)
        if parameters['spatial-discretization']['family'] == 'Q':
            self.CG = FunctionSpace(mesh, 'CG', parameters['spatial-discretization']['order'], variant="spectral") #TRY Q??
            self.DG = FunctionSpace(mesh, 'DG', parameters['spatial-discretization']['order']-1, variant="spectral") #TRY DQ?
#PLOTTING FOR DG L2 IS BROKEN
#UNCLEAR IF WE REALLY NEED DG L2?
            if mesh.cell_dimension() == 1:
                self.DGV = self.DG
                self.Hdiv = self.CG
                self.Hcurl = None
                self.CGV = self.CG
            if mesh.cell_dimension() == 2:
                self.DGV = VectorFunctionSpace(mesh, 'DG', parameters['spatial-discretization']['order']-1, variant="spectral") #TRY DQ?
                self.CGV = VectorFunctionSpace(mesh, 'CG', parameters['spatial-discretization']['order'], variant="spectral") #TRY Q?
                self.Hdiv = FunctionSpace(mesh, 'RTCF', parameters['spatial-discretization']['order'])
                self.Hcurl = FunctionSpace(mesh, 'RTCE', parameters['spatial-discretization']['order'])
#THIS IS NEEDED SINCE 3D HEXAHEDRA ARE NOT SUPPORTED YET
            if mesh.cell_dimension() == (2,1):
                self.DGV = VectorFunctionSpace(mesh, 'DG', parameters['spatial-discretization']['order']-1, variant="spectral") #TRY DQ?
                self.CGV = VectorFunctionSpace(mesh, 'CG', parameters['spatial-discretization']['order'], variant="spectral") #TRY DQ?
                self.Hdiv = FunctionSpace(mesh, 'NCF', parameters['spatial-discretization']['order'])
                self.Hcurl = FunctionSpace(mesh, 'NCE', parameters['spatial-discretization']['order'])
            if parameters['spatial-discretization']['lump_mass']:
                gll_rule = gauss_lobatto_legendre_cube_rule(dimension=parameters['mesh']['dim'], degree=self.CG.ufl_element().degree())
                gll_surf_rule = gauss_lobatto_legendre_cube_rule(dimension=parameters['mesh']['dim']-1, degree=self.CG.ufl_element().degree())
                self.dx = dx(scheme=gll_rule)
                self.dS = dS(scheme=gll_surf_rule)
                self.ds = ds(scheme=gll_surf_rule)
            else:
                self.dx = dx
                self.dS = dS
                self.ds = ds

        #elif parameters['family'] == 'P-':
        #    pass
        #elif parameters['family'] == 'P+':
        #    pass
        else:
            raise ValueError("FEEC family " + parameters['spatial-discretization']['family'] + " is unknown")
