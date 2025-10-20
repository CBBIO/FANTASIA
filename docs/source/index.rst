FANTASIA
========

**Functional ANnoTAtion based on embedding space SImilArity**

FANTASIA is an advanced pipeline designed for automatic functional annotation of protein sequences
using state-of-the-art protein language models. It integrates deep learning embeddings and similarity
searches in vector databases to associate Gene Ontology (GO) terms with proteins.

.. raw:: html

   <div style="margin: 1rem 0;"></div>

.. grid:: 1 2 2 3
   :gutter: 2
   :margin: 2 0 2 0

   .. grid-item-card:: Quickstart
      :link: appendix/installation_and_quickstart
      :link-type: doc
      :shadow: md
      :text-align: left

      Set up FANTASIA and run your first annotation. :bdg:`5 min` :bdg-primary:`Beginner`
      See :doc:`Start here → <introduction>`.

   .. grid-item-card:: Method
      :link: method/index
      :link-type: doc
      :shadow: md
      :text-align: left

      Embeddings, lookups, and evaluation details. :bdg-info:`PLMs` :bdg:`ESM` :bdg:`ProtT5` :bdg:`ProSTT5`
      Explore :doc:`method/index`.

   .. grid-item-card:: Results
      :link: results/index
      :link-type: doc
      :shadow: md
      :text-align: left

      Performance, metrics, and species panels. :bdg-success:`Benchmarks`
      See :doc:`results/index`.

   .. grid-item-card:: API Reference
      :link: reference/index
      :link-type: doc
      :shadow: md
      :text-align: left

      Autodoc-driven reference for modules and classes. :bdg-secondary:`autodoc`
      Browse :doc:`reference/index`.

   .. grid-item-card:: How to Cite
      :link: references
      :link-type: doc
      :shadow: md
      :text-align: left

      Citation info, datasets and reproducibility. :bdg:`BibTeX`
      See :doc:`references`.

   .. grid-item-card:: Contact
      :link: contact
      :link-type: doc
      :shadow: md
      :text-align: left

      Questions, issues, and collaborations. :bdg-warning:`Get in touch`
      Go to :doc:`contact`.

.. raw:: html

   <div style="margin: 1.5rem 0;"></div>

.. admonition:: What is FANTASIA?
   :class: tip

   A vector-search based functional annotation pipeline leveraging protein language models and Gene
   Ontology mappings.

.. toctree::
   :caption: Paper-like structure
   :maxdepth: 2

   abstract
   introduction
   method/index
   results/index
   discussion
   conclusions
   appendix/index
   references
   acknowledgments
   contact

.. toctree::
   :caption: API Reference
   :maxdepth: 2
   :hidden:

   reference/index
