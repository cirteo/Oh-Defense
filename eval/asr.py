import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('evaluation/LlamaGuard/attack/VLSafe/LLaVA/vlsafe.csv')

success_rate = (df['jailbroken_llamaguard_ours'] == True).sum() / df['jailbroken_stirng'].notna().sum()
print(f"ASR: {success_rate:.2%}")

