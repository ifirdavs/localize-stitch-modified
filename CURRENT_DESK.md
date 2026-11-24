Checks for notebooks:
0. Check each dataset notebook to have proper `preprocess` function
0. Is dataset in Single fine-tuned model CORRECT for each notebook?


Features:
1. Correct the README.md fully and rewrite the parts if necessary. New things done in this research in my mind:
- Applied LaS to Federated Learning domain.
- Using the previous stitched model as a reference model between rounds (original LaS uses a fixed pretrained reference in both learned LaS and dataless LaS variants).
- K and $\mu$ parameters are dynamically updated based on the previous round's validation accuracy.

2. Can this whole work be viable for a bachelor thesis?

3. (Optional, won't implement): Consider using **sparse masks** instead of dense masks to **reduce** the transmission or file size in communication rounds.




Further experiments to try:
0. TRY LaS **without dynamic K** between rounds.
0. TRY LaS **with learned localization masks**.
0. TRY **keeping the original pretrained reference** which would localize cumulative changes since pretraining. That would be a different adaptation. Original LaS uses a fixed pretrained reference in both variants (learned LaS and dataless LaS).

Reffering to your previous output: 
```md
Each method maintains its own global model. Every round, five client copies train for one epoch; their results are aggregated into that method’s next global model. For LaS, the reference now becomes the previous stitched global model:
\[ \Delta_i^{(t)}=\theta_i^{(t)}-\theta^{(t)}. \]
The retention fraction is:
\[ K_t=1-0.035\,A_{t-1}, \]
where accuracy is a fraction between 0 and 1.
```


