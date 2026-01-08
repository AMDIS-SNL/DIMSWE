from firedrake import inner, sign, dot, grad

def SVLieDerivative(degree, dim, u, a, ahat, alpha_s, n, order):
    #0-forms
    if degree == 0:
        return
    #volume forms
    elif degree == dim:
        alpha = alpha_s * sign(dot(u('+'),n('+')))
#MISSING BOUNDARY TERMS- ds
        atilde = 0.5 * ((1. + alpha) * a('+') + (1. - alpha)*a('-'))
        expr = (ahat('+')*inner(u('+'), n('+')) + ahat('-')*inner(u('-'), n('-')))*atilde*dS
        if order > 1:
            rhs_expr = rhs_expr - inner(grad(ahat), a * u   )*dx
        return expr
#PROBABLY NEED TO DISTINGUISH BETWEEN 1-FORMS AND N-1 FORMS HERE!
    #1-forms in 2D
    elif degree == 1 and dim == 2:
        return
    #1-forms in 3D
    elif degree == 1 and dim == 3:
        return
    #2-forms in 3D
    elif degree == 2 and dim == 3:
        return

#MISSING LOTS OF THESE EXPRESSIONS...
def VVLieDerivative(degree, dim, u, a, ahat):
    #0-forms
    if degree == 0:
        return
    #volume forms
    elif degree == dim:
        return
#PROBABLY NEED TO DISTINGUISH BETWEEN 1-FORMS AND N-1 FORMS HERE!
    #1-forms in 2D
    elif degree == 1 and dim == 2:
        return
    #1-forms in 3D
    elif degree == 1 and dim == 3:
        return
    #2-forms in 3D
    elif degree == 2 and dim == 3:
        return

#MISSING LOTS OF THESE EXPRESSIONS...
def CVLieDerivative(degree, dim, u, a, ahat):
    #0-forms
    if degree == 0:
        return
    #volume forms
    elif degree == dim:
        return
#PROBABLY NEED TO DISTINGUISH BETWEEN 1-FORMS AND N-1 FORMS HERE!
    #1-forms in 2D
    elif degree == 1 and dim == 2:
        return
    #1-forms in 3D
    elif degree == 1 and dim == 3:
        return
    #2-forms in 3D
    elif degree == 2 and dim == 3:
        return

#EVENTUALLY ADD SOME TENSOR-VALUED BUNDLES ALSO? Unclear...
