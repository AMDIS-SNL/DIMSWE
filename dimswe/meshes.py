from firedrake import IntervalMesh, PeriodicIntervalMesh, RectangleMesh, PeriodicRectangleMesh, BoxMesh, PeriodicBoxMesh, Mesh, FacetNormal
from firedrake import FunctionSpace, VectorFunctionSpace, dx, ds, dS, ExtrudedMesh
from firedrake import Function, CellDiameter
import finat
import FIAT

from .ufl_helpers import ufl_attribute

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

# FEEC families supported by DeRhamComplex (TODO-elliptic §4.9).
FEEC_FAMILIES = ('Q', 'P-', 'P+', 'S-', 'S')

# Cells each family can be built on (cellname of the mesh; 'extruded' = quad x interval).
FAMILY_CELLS = {
    'Q': ('interval', 'quadrilateral', 'extruded'),
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
        return {'CG': ('CG', r, spectral), 'Hcurl': (curl, r, None), 'Hdiv': (div, r, None), 'DG': ('DG', r - 1, spectral)}
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
    """'interval' | 'triangle' | 'quadrilateral' | 'tetrahedron' | 'hexahedron' | 'extruded'."""
    if isinstance(ufl_attribute(mesh, 'cell_dimension'), tuple):
        return 'extruded'
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
        self.dim = 3 if self.cell == 'extruded' else ufl_attribute(mesh, 'topological_dimension')
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
