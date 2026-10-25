# control-in-biophysics

Python notebooks accompanying the review *Control of biophysical dynamics across scales*. Each folder implements one control paradigm on a biophysical case study:

| Folder | Framework | Case study |
| --- | --- | --- |
| `FeedbackControl/` | PID feedback | Insulin regulation in an artificial pancreas |
| `OptimalControl/` | Pontryagin / bang-bang | Protein aggregation kinetics |
| `ModelPredictiveControl/` | MPC | Optogenetic control of GFP expression |
| `ReinforcementLearning/` | Deep RL | Adaptive drug scheduling in cancer therapy |

Run the notebooks with a standard scientific Python stack (`numpy`, `scipy`, `matplotlib`; `torch` for the RL example, and `casadi` for the optimal control example).

Web-based visualiser is contained in the `docs/` folder, accessible [here](https://dq219.github.io/control-in-biophysics/).

Released under the [MIT License](LICENSE).