
Checks for notebooks:
0. Check each dataset notebook to have proper `preprocess` function
0.a Is dataset in Single fine-tuned model CORRECT for each notebook?


Features:
0. TRY LaS **without dynamic K** between rounds.
0. TRY LaS **with learned localization masks**.

1. Correct the README.md fully and rewrite the parts if necessary. New things done in this research in my mind:
- Applied LaS to Federated Learning domain.
- Using the previous stitched model as a reference model between rounds (original LaS uses a fixed pretrained reference in both learned LaS and dataless LaS variants).
- K and $\mu$ parameters are dynamically updated based on the previous round's validation accuracy.

2. Reffering your previous output: 
   ```md
Each method maintains its own global model. Every round, five client copies train for one epoch; their results are aggregated into that method’s next global model. For LaS, the reference now becomes the previous stitched global model:
\[ \Delta_i^{(t)}=\theta_i^{(t)}-\theta^{(t)}. \]
The retention fraction is:
\[ K_t=1-0.035\,A_{t-1}, \]
where accuracy is a fraction between 0 and 1.
   ```

   Is it okay that, in terms of LaS, the reference becomes the previous stitched global model?

3. Can this whole work be viable for a bachelor thesis?

 . (Optional, won't implement): Consider using **sparse masks** instead of dense masks to **reduce** the transmission or file size in communication rounds.