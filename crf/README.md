CRF on FIPCO
=====

To run CRF toolbox on the FIPCO dataset you need to do a couple of things.

- Use `<root_dir>/utils/matlab/create_crf_sequence.m` to create Training and Test data for CRF. Note the format of the input data.
- Start matlab with `matlab -nodesktop` on the server.
- Run `addpath('<path to data>')`.
- Run `addpath('HCRF2.0b/apps/matHCRF')` to add the HCRF code to MATLAB path.
- Run `MS_runOnZFaceMAT(...)` with appropriate arguments.
