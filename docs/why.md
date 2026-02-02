# Why MLArray?

MLArray addresses a gap I repeatedly ran into over the last few years: we have excellent storage formats optimized for machine learning workloads, but no widely usable *image* format that combines **efficient array storage** with **standardized, software-friendly metadata**.

Projects like **Zarr** and **Blosc2** already solve the “store large arrays efficiently” problem extremely well. However, they do not provide a standardized metadata layer for imaging. As a result, it’s difficult to integrate their file formats into common analysis and visualization tools in a meaningful and consistent way.

MLArray is designed to bridge that gap: a machine-learning-friendly array format that preserves metadata and enables a broader ecosystem of tooling around it.

---

## How does MLArray address this gap?

* **A standardized, extensible metadata schema**
  MLArray defines a metadata schema that balances *standardization* and *flexibility*: software that supports MLArray has a consistent way to access relevant metadata, while users can still attach arbitrary custom metadata when needed.

* **Preserve source metadata across conversions**
  Users can convert images from arbitrary formats to MLArray while preserving the source metadata *in a structured and reproducible way*. Tools that integrate MLArray can still access metadata according to the source format’s conventions, which makes MLArray a practical alternative for ML pipelines without breaking downstream analysis or visualization workflows.

* **Machine learning–specific metadata support**
  In addition to format-preserving metadata, MLArray includes a dedicated schema for machine-learning-relevant information, and it also supports storing dynamic metadata outside predefined schemas.

---

## What type of images can I store as MLArray?

In short: **any array data**.

MLArray was designed with very large *N*-dimensional images in mind, including:

* medical imaging (radiology, histopathology, etc.)
* satellite and remote sensing data
* general scientific imaging
* segmentation masks and label maps

Natural image data can also be stored in MLArray, but it is often unnecessary—formats like **JPEG** and **PNG** are already a strong default for many ML training pipelines.

MLArray can also store **metadata-only** or **non-array data**, such as:

* bounding boxes
* regression targets
* classification results

This can be useful when you want a standardized interface for accessing these annotations and results, enabling simpler analysis and visualization in software that supports MLArray.

---

## How is MLArray optimized for Machine Learning / Deep Learning?

MLArray uses **Blosc2** as its storage backend, which provides several properties that are particularly well-suited for machine learning and deep learning workloads.

For details, see: [ML Optimization](https://MIC-DKFZ.github.io/mlarray/optimization/)
