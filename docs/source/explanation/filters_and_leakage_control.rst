Filters and leakage control
===========================

Taxonomy include/exclude lists restrict reference eligibility before distance
computation and match exact IDs. Include-only takes precedence. Automatic
descendant expansion is disabled because it depended on an external local
taxonomy state.

Optional MMseqs2 redundancy masking removes references assigned to the query's
cluster; it does not guarantee exclusion of every high-identity donor. A
benchmark should retrieve multiple neighbours and apply explicit global
identity filtering afterward. These controls change donor availability and
therefore transferred GO terms; they do not by themselves measure accuracy.
