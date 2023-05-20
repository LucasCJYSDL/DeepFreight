# DeepFreight
Codebase for my paper: DeepFreight: A Model-free Deep-reinforcement-learning-based Algorithm for Multi-transfer Freight Delivery

Language: Python

The following components are included:
- A multi-hop freight delivery simulator.
- Implementations of SOTA MARL algorithms: MAVEN, QMIX, Weighted QMIX, MSAC, COMA.
- Implementation of a MILP solver based on Gurobi, which is used in conjunction with the MARL output. 

Please cite the paper:

```bash
@inproceedings{DBLP:conf/aips/ChenULA21,
  author       = {Jiayu Chen and
                  Abhishek K. Umrawal and
                  Tian Lan and
                  Vaneet Aggarwal},
  title        = {DeepFreight: {A} Model-free Deep-reinforcement-learning-based Algorithm
                  for Multi-transfer Freight Delivery},
  booktitle    = {Proceedings of the Thirty-First International Conference on Automated
                  Planning and Scheduling, {ICAPS} 2021, Guangzhou, China (virtual),
                  August 2-13, 2021},
  pages        = {510--518},
  publisher    = {{AAAI} Press},
  year         = {2021}
}
```
