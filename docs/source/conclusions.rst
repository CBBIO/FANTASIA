Conclusions
===========

Our work leverages protein language models (PLMs) to explore functional annotation within FANTASIA.

- **Validity of the approach**
  Our results confirm that PLMs provide real value in recovering annotations with precision. The use of taxonomy and redundancy filters, together with additional metrics such as global and local identity and reliability index, strengthens the robustness of the method.

- **Advances in the reimplementation**
  The modular reimplementation has yielded a more robust and maintainable framework, supporting newer models and flexible filtering pipelines.

- **Limitations**
  Amplifying the noise introduced by **IEA annotations** will not lead to reliable results in the future. For this reason, our method considers as *certain* only annotations with **experimental evidence**, which inevitably introduces a **bias toward well-studied model organisms**.
  However, by relying on annotations that we can be confident are real and individually validated by the scientific community, we avoid **erroneous or redundant transfers**. Looking ahead, we expect that the annotation databases will continue to grow, allowing us to keep updating our method as scientific knowledge in this area advances.

- **Future perspectives**
  Beyond benchmarking, FANTASIA is already being used to provide a first approximation of the functions of proteins that remain unannotated. This exploratory capacity opens the door to its integration into a wide variety of bioinformatics pipelines, where preliminary functional hints can guide downstream analyses.
  In parallel, extending the evaluation to upcoming challenges such as **CAFA6** will further strengthen benchmarking and provide continuity with the international community efforts in protein function prediction.