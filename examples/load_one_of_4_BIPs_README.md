BIPs are loadable by running

`model, true_m_np = load_one_of_4_BIPs(PDEproblem_dir, BIPproblem_dir, BIP)` from the `load_one_from_4_BIPS.py` file

`model`: a hippylib BIP model
`true_m_np`: a numpy array of the `true` parameter that generated this particular model's synthetic data. In case its needed for prior/posterior predictive checks, or for any sort of results/plotting/etc. One would of course have to run `m.set_local(true_m_np)`, where m is a dolfin parameter space vector, if one wants to do this.

One way to do run this code is from the command line:

`python load_one_of_4_BIPs.py -BIP 1 -BIP_problem_dir "/storage/joshua/nonlinear_diffusion_reaction/problem/rel_noise_0.01_noise_stdev_0.009724453175183136/"
-PDEproblem_base_dir "/workspace/josh/dinox/examples/"

Since all of the problem data is stored in `/storage/joshua/nonlinear_diffusion_reaction/problem/`, you will have to pick one (or all) of the folders in this directory to run

For `-PDEproblem_base_dir`, one could either copy the examples folder to your own working directory, or just operate from there. 

The examples have been modified to work sepcifically with this code, for example, settings are modified for each BIP before instantiating a hippylib.model
Therefore, really the only way to currently interface with/load up a BIP model is with this loader. One can also just copy load_one_of_4_BIPs to wherever one would like and run it within your own code, rather than using the command line interface.


To run MCMC on ALL of the BIPS, one needs to vary BIP between [0,1,2,3] for each probelm in a PDE problem, e.g. each folder in `/storage/joshua/nonlinear_diffusion_reaction/problem/`.

Please let Joshua Chen know if any permissions need to be changed. I plan to have everything be read-only, but write permissions within the folders allowed, so that one can define `MCMC_results` in `/storage/joshua/nonlinear_diffusion_reaction/problem/rel_noise_0.01_noise_stdev_0.009724453175183136/` for example.
