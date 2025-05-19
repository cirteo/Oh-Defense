import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/path/to/your/jailbreakv_28k.csv')


df['format'] = df['format'].replace(['SD', 'SD_typo', 'typo'], 'Query-Relevant')
df['format'] = df['format'].replace(['Template','Logic','Persuade'],'LLM_Transfer')

format_counts = df['format'].value_counts()
format_proportions = format_counts / len(df)


def calculate_asr(group):
    return ((group['jailbroken_llamaguard_ours'] == True)).sum() / len(group)

asr_by_format = df.groupby('format').apply(calculate_asr)

custom_order = ['figstep','Query-Relevant','LLM_Transfer']


asr_dict = dict(asr_by_format)

ordered_values = [asr_dict[format_type] for format_type in custom_order]


plt.figure(figsize=(12, 6))
bars = plt.bar(custom_order, ordered_values)
plt.title('ASR')
plt.xlabel('Formats')
plt.ylabel('ASR')
plt.ylim(0, 1)

# Add specific values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.5f}',
             ha='center', va='bottom')

plt.savefig('jail.png')
plt.close()
