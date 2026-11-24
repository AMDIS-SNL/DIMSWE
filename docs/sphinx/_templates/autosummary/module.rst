{#-
  Page template for one module of the API reference (used by the autosummary tables in
  api/index.rst through ``:template: module.rst``).

  Each module gets a single page: the module docstring, summary tables of its classes, functions
  and exceptions (linking to the entries further down the same page), then the full autodoc
  entries with inheritance ("Bases: ...") shown for every class. Only objects defined in the
  module are listed; names imported from elsewhere (for example the ``from .x import *``
  re-exports) are documented on their own module's page.
-#}
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

{%- block classes %}
{%- if classes %}

   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
{% for item in classes %}
      {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}

{%- block functions %}
{%- if functions %}

   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
{% for item in functions %}
      {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}

{%- block exceptions %}
{%- if exceptions %}

   .. rubric:: Exceptions

   .. autosummary::
      :nosignatures:
{% for item in exceptions %}
      {{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}

   .. rubric:: Reference
