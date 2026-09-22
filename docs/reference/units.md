# Units

Unless a public API explicitly documents another convention:

- coordinates: **Bohr**
- energies: **Hartree**
- forces: **Hartree/Bohr**
- nuclear gradients: **Hartree/Bohr**, with force equal to minus gradient

Unit conversion belongs at an explicit API boundary; do not infer units from input magnitude.
