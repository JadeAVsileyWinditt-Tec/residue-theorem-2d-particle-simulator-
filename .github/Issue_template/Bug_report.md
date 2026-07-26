name: Bug report
description: File a bug or numerical discrepancy issue in the TBU simulation engine.
title: "[BUG]: "
labels: ["bug"]
body:
  - type: textarea
    attributes:
      label: Description of the Issue
      description: Clear description of what the bug is.
      placeholder: The Keplerian tarpit simulation collapses when...
    validations:
      required: true
  - type: textarea
    attributes:
      label: Hardware & Environment
      description: GPU model, CUDA version, PyTorch version.
      placeholder: RTX 4090, CUDA 12.x, PyTorch 2.x
    validations:
      required: true
